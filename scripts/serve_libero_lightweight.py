"""Serve a pi0.5 LIBERO policy with CPU prefix encoding and selectable denoising."""

import argparse
import dataclasses
import json
import logging
import math
import os
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from openpi import transforms
from openpi.models import gemma
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies.policy import Policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared.download import get_cache_dir
from pruning.run_structured_ffn import MANIFEST_NAME
from pruning.run_structured_ffn import get_train_config
from pruning.run_structured_ffn import load_original_bf16
from pruning.run_structured_ffn import load_pruned_model
from pruning.structured_ffn import _aligned_kept

DENOISE_STEPS = 10
LIBERO_VIEWS = ("base_0_rgb", "left_wrist_0_rgb")
KEEP_FRACTION = 0.85


def attach_split_sampler(policy, model, final, prefix_info, denoise_device):
    root = model.paligemma_with_expert
    root.paligemma.language_model.config._attn_implementation = "eager"
    denoise_modules = (
        root.gemma_expert.model,
        model.action_in_proj,
        model.time_mlp_in,
        model.time_mlp_out,
        model.action_out_proj,
    )
    for module in denoise_modules:
        module.to(device=denoise_device, dtype=torch.float32)
    for device, modules in ((torch.device("cpu"), (root.paligemma,)), (denoise_device, denoise_modules)):
        if any(p.device != device or p.dtype != torch.float32 for module in modules for p in module.parameters()):
            raise RuntimeError(f"Expected FP32 parameters on {device}")

    @torch.inference_mode()
    def sample_actions(_device, observation, *, noise, num_steps):
        images, masks, tokens, token_mask, state = model._preprocess_observation(observation, train=False)
        if len(images) != 3 or [bool(mask.any()) for mask in masks] != [True, True, False]:
            raise RuntimeError("Expected base, left wrist and masked right wrist in LIBERO input")
        images, masks = images[:2], masks[:2]
        prefix, pad, att = model.embed_prefix(
            images, masks, tokens, token_mask, image_keys=LIBERO_VIEWS, inference=True,
        )
        valid = token_mask[0].nonzero()
        language_span = int(valid[-1, 0]) + 1 if valid.numel() else 0
        vision_tokens = 384 if model.vision_token_pooling == "max" else 512
        language_tokens = language_span if final else tokens.shape[1]
        if prefix.shape[1] != vision_tokens + language_tokens or prefix.dtype != torch.float32:
            raise RuntimeError(f"Unexpected prefix shape or dtype: {prefix.shape}, {prefix.dtype}")
        prefix_info.update(
            input_views=3,
            encoded_views=len(images),
            vision_tokens=vision_tokens,
            language_tokens=language_tokens,
            language_span=language_span,
            prefix_tokens=prefix.shape[1],
        )
        _, cpu_cache = root.forward(
            attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        denoise_cache = DynamicCache()
        for layer, (key, value) in enumerate(cpu_cache):
            denoise_cache.update(key.to(denoise_device), value.to(denoise_device), layer)
        state, pad = state.to(denoise_device), pad.to(denoise_device)
        x = noise.to(denoise_device)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=denoise_device)
        t = torch.tensor(1.0, dtype=torch.float32, device=denoise_device)
        for _ in range(num_steps):
            v = model.denoise_step(state, pad, denoise_cache, x, t.expand(x.shape[0]))
            x = x + dt * v
            t += dt
        # Include the completed denoising work in the policy timing.
        return x.cpu()

    policy._sample_actions = sample_actions


class SeededPolicy:
    def __init__(self, policy, noise_shape, prefix_info):
        self.policy = policy
        self.noise_shape = noise_shape
        self.prefix_info = prefix_info
        self.generator = torch.Generator(device="cpu").manual_seed(123)

    @torch.inference_mode()
    def infer(self, observation):
        observation = dict(observation)
        if (seed := observation.pop("benchmark_noise_seed", None)) is not None:
            self.generator.manual_seed(int(seed))
        noise = torch.randn(self.noise_shape, generator=self.generator, dtype=torch.float32).numpy()
        result = self.policy.infer(observation, noise=noise)
        result["comparison_prefix"] = dict(self.prefix_info)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("original", "final"))
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--pruned-dir", type=Path)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--device", default="cuda:0", help="Denoising device: cpu, cuda, or cuda:N")
    parser.add_argument("--vision-pooling", choices=("none", "max"), default="none")
    args = parser.parse_args()
    final = args.model == "final"
    if final != (args.pruned_dir is not None):
        parser.error("--pruned-dir is required for final and invalid for original")

    logging.basicConfig(level=logging.INFO)
    if not final and args.vision_pooling != "none":
        parser.error("--vision-pooling max requires the final model")
    try:
        denoise_device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if denoise_device.type == "cuda":
        index = denoise_device.index if denoise_device.index is not None else 0
        if index >= torch.cuda.device_count():
            parser.error(f"CUDA device {index} is unavailable")
        denoise_device = torch.device("cuda", index)
    elif denoise_device.type == "cpu":
        denoise_device = torch.device("cpu")
    else:
        parser.error("--device must select CPU or CUDA")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dense_dir = args.dense_dir.resolve(strict=True)
    tokenizer = get_cache_dir() / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"Expected the pre-cached tokenizer at {tokenizer}")
    os.environ["OPENPI_VISION_TOKEN_POOLING"] = args.vision_pooling
    prune_metadata = {}
    if final:
        checkpoint = args.pruned_dir.resolve(strict=True)
        manifest = json.loads((checkpoint / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest["target"] != "language" or not math.isclose(manifest["requested_keep_fraction"], KEEP_FRACTION):
            raise ValueError("Expected a 15% Gemma language-only FFN pruning checkpoint")
        config = get_train_config()
        language = gemma.get_config(config.model.paligemma_variant)
        width = _aligned_kept(language.mlp_dim, KEEP_FRACTION, manifest["multiple_of"])
        layers = {
            (layer["part"], layer["layer"], layer["original_intermediate"], layer["new_intermediate"])
            for layer in manifest["layers"]
        }
        expected = {("language", index, language.mlp_dim, width) for index in range(language.depth)}
        if width == language.mlp_dim or layers != expected or len(manifest["layers"]) != len(expected):
            raise ValueError("The pruning manifest does not describe every expected language FFN layer")
        prune_metadata = {
            "prune_target": manifest["target"],
            "requested_keep_fraction": manifest["requested_keep_fraction"],
            "actual_prune_fraction": 1 - width / language.mlp_dim,
        }
        model = load_pruned_model(checkpoint, device="cpu")
    else:
        model, config, checkpoint = load_original_bf16(str(dense_dir))
    model.config = dataclasses.replace(model.config, trim_language_padding=final)
    model.float()
    data = config.data.create(dense_dir / "assets", config.model)
    if data.norm_stats is None:
        raise RuntimeError("Could not load the original LIBERO normalization statistics")

    policy = Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": DENOISE_STEPS},
        is_pytorch=True,
        pytorch_device="cpu",
    )
    prefix_info = {}
    attach_split_sampler(policy, model, final, prefix_info, denoise_device)
    language_ffn_widths = [
        layer.mlp.gate_proj.out_features for layer in model.paligemma_with_expert.paligemma.language_model.layers
    ]
    if final and language_ffn_widths != [width] * language.depth:
        raise RuntimeError(f"Unexpected loaded language FFN widths: {language_ffn_widths}")
    metadata = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "normalization_source": str(dense_dir / "assets"),
        "vision_token_pooling": model.vision_token_pooling,
        "skip_right_wrist": True,
        "trim_language_padding": model.config.trim_language_padding,
        "prefix_device": "cpu",
        "denoise_device": str(denoise_device),
        "dtype": "float32",
        "denoise_steps": DENOISE_STEPS,
        "language_ffn_widths": language_ffn_widths,
        **prune_metadata,
    }
    print("COMPARE_SETTINGS " + json.dumps(metadata), flush=True)
    WebsocketPolicyServer(
        SeededPolicy(policy, (model.config.action_horizon, model.config.action_dim), prefix_info),
        host="127.0.0.1",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()

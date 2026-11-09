# Vision token pooling for PyTorch π0.5 LIBERO

`Pi0Config.vision_token_pooling` accepts `none` (default), `avg`, or `max`.
Set `OPENPI_VISION_TOKEN_POOLING` before model creation to override the config.
The dense PyTorch loader and the [FFN-pruned checkpoint loader](../pruning/README.md)
both construct `PI0Pytorch` and use its shared `embed_prefix` method. Pooling
adds no checkpoint parameters. Use the same checkpoint loader and assets as before.

## Existing split evaluator

With the new model code on the evaluator's Python import path, add the two
variables to the existing `serve_compare.py` command. Keep the evaluator's
existing `PYTHONPATH` and device settings.

```sh
OPENPI_SKIP_MASKED_VIEWS=1 OPENPI_VISION_TOKEN_POOLING=avg \
  "$model_python" -u "$eval_dir/serve_compare.py" dense --t400 --port "$port"
```

Set `OPENPI_VISION_TOKEN_POOLING` to `none`, `avg`, or `max`. Use `dense` for the
original checkpoint or `pruned` for the language FFN 10% checkpoint. Select
each mode in a fresh process. `OPENPI_SKIP_MASKED_VIEWS=1`
is the split evaluator's existing switch for dropping fully masked views. For
Python callers, `dataclasses.replace(model_config, vision_token_pooling="avg")`
also sets the mode when the environment override is unset.

## Camera identity and prefix length

LIBERO names its base camera `base_0_rgb`. The split evaluator filters the
preprocessed image tensors into a new list and passes them to `embed_prefix`.
Each tensor carries its camera key through that filter. A wrist tensor remains
a wrist tensor even when it becomes the first or only view. If calling
`embed_prefix` with newly constructed or copied tensors, supply the matching
`image_keys` sequence when pooling is enabled.

After SigLIP and the PaliGemma projector, `avg` and `max` reduce each adjacent
horizontal pair in the base camera's row-major 16×16 grid. Both produce a 16×8
grid. The wrist grid and the 200 padded language tokens, including π0.5's
discrete state, retain their values and lengths. With the fully masked right
wrist removed, the physical lengths are:

| Mode | Views | Base tokens | Wrist tokens | Prefix and KV cache |
| --- | ---: | ---: | ---: | ---: |
| `none` | 2 | 256 | 256 | 712 |
| `avg` | 2 | 128 | 256 | 584 |
| `max` | 2 | 128 | 256 | 584 |

Prefill uses the physical prefix length for the cache. Denoising uses that
length for its attention mask and starts its position IDs at the number of
**valid** prefix tokens, excluding language padding. The standard model path
retains all three camera slots, so its physical lengths are 968 and 840.

## CPU verification

With the project's dependencies and patched Transformers installed:

```sh
JAX_PLATFORMS=cpu uv run pytest -q \
  src/openpi/models_pytorch/pi0_vision_pooling_test.py \
  src/openpi/models_pytorch/pi0_vision_pooling_cache_test.py
```

The tests use deterministic projected image tokens and a tiny CPU Gemma with
the production cache and denoising paths. CPU–T400 latency and LIBERO success
rates require separate evaluation with the real checkpoints.

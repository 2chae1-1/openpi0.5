# Skipping inactive cameras in PyTorch pi05

`Pi0Config.pytorch_skip_inactive_images` defaults to `False`. Enable it explicitly
on the model config **before loading** a PyTorch pi05 checkpoint:

```python
import dataclasses

from openpi.policies import policy_config
from openpi.training import config

train_config = config.get_config("pi05_libero")
train_config = dataclasses.replace(
    train_config,
    model=dataclasses.replace(train_config.model, pytorch_skip_inactive_images=True),
)
policy = policy_config.create_trained_policy(train_config, "/path/to/pytorch/checkpoint")
```

The checkpoint must contain `model.safetensors` to select the PyTorch path. The
option does not change JAX execution. PyTorch pi0 rejects this option when enabled.
Model weights and checkpoint keys are unchanged.

## Selection and sequence lengths

Before `embed_prefix` calls SigLIP, the selector checks each camera's image mask
over the **entire batch**. Only a camera with no valid images is omitted. A mixed
batch retains that camera's full block and original per-example mask. Selection
is repeated for every call; camera names and image pixel values do not decide
validity. Even an all-zero image is retained when its mask is valid.

For LIBERO pi05, `LiberoInputs` supplies `right_wrist_0_rgb` with a False mask, so
its SigLIP call and prefix token block are omitted when this option is enabled.
The base and left-wrist cameras remain. This also works if all cameras are
inactive: language tokens form the prefix.

The existing attention and cache code uses the resulting prefix tensors:

- Padding and attention masks contain only retained token blocks.
- Position IDs use cumulative valid-token counts. Removing all-False blocks
  preserves positions for retained tokens and the suffix's starting offset.
- Prefill stores only the shorter prefix in the KV cache. Denoising constructs
  its mask from that prefix's actual width and reuses the cache across steps.

There is no image token pooling or token merging. Valid camera tokens are not
compressed. W8A8 and FFN pruning code are unchanged.

## Compilation

Mask selection uses `@torch.compiler.disable`, an intentional eager boundary
inside the compiled entry point. This supports the existing `torch.compile`
default `fullgraph=False`; the enabled path does not support `fullgraph=True`.
See the [PyTorch compiler API](https://docs.pytorch.org/docs/2.7/torch.compiler_api.html).

The mask reductions are collected with one `.tolist()` call. On a GPU this
requires a host synchronization once per prefix, outside the denoising loop.
SigLIP and subsequent tensor operations remain eligible for compilation.
Changing the active camera layout can cause specialization/recompilation because
prefix lengths change. The selector never freezes a first-seen mask layout.
Benchmark this boundary and recompilation costs on the deployment GPU.

## CPU verification

With project dependencies installed:

```sh
JAX_PLATFORMS=cpu uv run pytest src/openpi/models_pytorch/pi0_pytorch_test.py -q
```

The tests use a small deterministic backbone without downloading model weights.
They exercise the real PI0 embedding, mask, sampling, and denoising methods,
including baseline parity, mixed masks, language-only prefixes, changing masks
under Dynamo capture with an eager-equivalent backend, position IDs, tensor cache
lengths, and training loss/gradient parity.
These tests do not establish real-checkpoint numerical parity or GPU performance.

Local validation on 2026-11-02 (Windows CPU, Python 3.12.2, PyTorch 2.7.1+cpu):
18 new CPU cases and 4 existing `pi0_test.py` cases passed. Ruff 0.8.6 lint and
format checks passed for all three changed Python files. The local command was:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$env:JAX_PLATFORMS = "cpu"
.\.venv\Scripts\python.exe -m pytest src/openpi/models_pytorch/pi0_pytorch_test.py src/openpi/models/pi0_test.py -q
```

## GPU and LIBERO validation still required

Use the same checkpoint, observations, prompts, noise, denoising steps, precision,
and device for option-off/on comparisons. Check:

1. SigLIP calls fall from three to two for LIBERO inputs. Prefill/cache width
   decreases by one camera's actual image-token count; valid masks and suffix
   position IDs stay consistent. A mixed/valid right-wrist batch must keep it.
2. Actions and valid-token outputs agree within an appropriate measured tolerance
   in eager and deployment compile modes, including after mask-layout changes.
3. Warm-up/compilation time, graph breaks/recompilations, synchronized SigLIP and
   prefill latency, total policy latency (median/tail), throughput, and peak GPU
   memory are measured separately. Include the selector's synchronization cost.
4. LIBERO suites use matched seeds, episode counts, and evaluation settings for
   success-rate comparison. Report the counts and uncertainty with the results.

No GPU speedup or LIBERO success-rate improvement is claimed before these checks.

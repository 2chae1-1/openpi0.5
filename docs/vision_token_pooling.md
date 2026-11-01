# Vision token pooling baseline (PyTorch)

`Pi0Config.vision_token_pooling` defaults to `False`. Enable it for the PyTorch
`pi05_libero` model without changing or fine-tuning checkpoint weights:

```python
import dataclasses
import pathlib

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config

checkpoint = pathlib.Path("/path/to/pi05_libero_pytorch")
assert (checkpoint / "model.safetensors").is_file()

train_config = config.get_config("pi05_libero")
train_config = dataclasses.replace(
    train_config,
    model=dataclasses.replace(train_config.model, vision_token_pooling=True),
)
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint,
    pytorch_device="cuda:0",
    sample_kwargs={"num_steps": 10},
)
websocket_policy_server.WebsocketPolicyServer(
    policy=policy,
    host="0.0.0.0",
    port=8000,
    metadata=policy.metadata,
).serve_forever()
```

The checkpoint must be the PyTorch checkpoint with its matching normalization
assets. `create_trained_policy` selects PyTorch by the presence of
`model.safetensors`; a JAX checkpoint does not enable this feature. The existing
`serve_policy.py` CLI does not expose a model configuration override, so use the
programmatic setup above. Change only `vision_token_pooling` to `False` for the
comparison run. Install the repository's required Transformers replacements as
described in the main README before loading the PyTorch model.

## Operation and sequence lengths

In `PI0Pytorch.embed_prefix`, pooling runs immediately after each camera's
`embed_image` result, which already includes SigLIP and the PaliGemma projector.
For the row-major 16 by 16 patch grid, it computes
`pooled[b, row, col] = mean(original[b, row, 2*col:2*col+2])`.
The result contains 16 by 8 = 128 tokens instead of 256. Pooling is independent
for every camera and batch element and introduces no parameters. SigLIP and
projector computation still operate at their original resolution.

For `N` camera slots and `L` padded language tokens, physical prefix length is
`256*N + L` when disabled and `128*N + L` when enabled. All camera slots,
including masked slots, are embedded and retained. This baseline does not skip
inactive cameras. The standard LIBERO transform supplies base, left wrist, and a
masked right wrist slot. With the `pi05_libero` language length of 200:

| Quantity | Disabled | Enabled |
| --- | ---: | ---: |
| Tokens per camera slot | 256 | 128 |
| Image tokens across three slots | 768 | 384 |
| Physical prefix / per-layer KV cache length | 968 | 584 |
| Prefix plus 10 action tokens | 978 | 594 |

The prefix padding and attention masks are constructed from the pooled tensor's
actual length. Prefix position IDs use the cumulative valid-token mask, and
suffix position IDs start after the number of valid prefix tokens. Masked camera
and language tokens still occupy physical cache slots but do not advance valid
positions. Denoising attends to the shorter prefix cache plus the unchanged
action suffix; the output remains `[batch, 10, 32]` for `pi05_libero`.

Pooling uses static shape operations and a mean reduction with no tensor-value
dependent camera selection. The CPU compile test checks the prefix with
`torch.compile(..., backend="eager", fullgraph=True)`; test the actual deployment
compile mode separately on the target GPU.

## GPU latency, memory, and cache validation

Run disabled and enabled configurations in separate fresh processes on the same
GPU. Keep checkpoint, precision, batch size, observation tensors, prompt and
padding, action noise, denoising steps, compile mode, software versions, and power
settings identical. Use representative recorded observations and fixed noise.
Record the GPU, driver, CUDA, PyTorch, and Transformers versions and the source
commit. Measure eager and compiled configurations separately if both matter.

Warm up each configuration until compilation and timings stabilize (start with
at least 20 calls); exclude loading, compilation, and warmup from steady-state
latency. The following helper times a callable that returns its result without
retaining it, using at least 100 measured calls:

```python
import time

import numpy as np
import torch


@torch.inference_mode()
def measure_cuda(call, *, device="cuda:0", warmup=20, repeats=100):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    base_allocated = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        call()
        torch.cuda.synchronize(device)
        elapsed_ms.append((time.perf_counter() - start) * 1000)

    return {
        "median_ms": float(np.median(elapsed_ms)),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "base_allocated_bytes": base_allocated,
        "base_reserved_bytes": base_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
```

Measure these scopes separately, releasing outputs between calls:

1. **Prefix prefill:** image/language embedding and the PaliGemma forward that
   builds the prefix cache, using the masks and positions from `sample_actions`.
2. **One denoising step:** `denoise_step` with a precomputed prefix cache, fixed
   `x_t`, and fixed timestep. Include this resident cache in the memory baseline.
3. **Full action sampling:**
   `model.sample_actions(device, observation, noise=noise, num_steps=10)` with an
   already transformed, batched observation on the GPU.
4. **End-to-end policy inference:** `policy.infer(raw_observation, noise=noise)`
   to include preprocessing, transfers, and output transforms. Report websocket
   and simulation time separately if included.

The existing `policy_timing.infer_ms` stops its timer before the output's CPU
transfer and has no explicit CUDA synchronization. Use synchronized measurements
for GPU comparisons instead of relying on that field alone. For each scope,
report absolute peak allocated and reserved VRAM and their baselines, in bytes
or consistently converted MiB. A smaller KV cache does not by itself establish a
smaller whole-process peak because weights, temporary tensors, compilation, and
the allocator also contribute.

In a separate diagnostic run, inspect the real prefill cache:
`cache.get_seq_length()` and each layer's key/value tensor sequence dimension
must equal the physical prefix length above. Check prefix attention shape
`[B, 1, P, P]`, suffix attention shape `[B, 1, 10, P+10]`, and position IDs against
the masks. Recheck the cache after several denoising steps: it should retain
length `P` because suffix inference uses `use_cache=False`. Avoid hooks or
diagnostic logging in the timed runs.

## LIBERO success-rate validation

Use the environment setup in [the LIBERO example](../examples/libero/README.md)
and run its client against the programmatic server above. Evaluate
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` separately. For
example, in the LIBERO client environment:

```bash
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.seed 7 \
  --args.video-out-path data/libero/pooling_on/seed7/libero_spatial
```

Repeat each suite with pooling disabled and enabled, using identical task order,
initial states, seeds, 224-pixel resizing, five-step replanning, settling steps,
episode limits, checkpoint, normalization, precision, and denoising settings.
Use separate log/video directories. Repeat with several predetermined seeds.
The client seeds NumPy and the environment; also seed the server's Torch RNG
(and CUDA RNGs) identically before each run to control action noise. For strict
paired noise across diverging rollouts, use a harness that supplies recorded
per-task/per-episode/per-replan noise to `Policy.infer`; the stock websocket
client does not provide such pairing automatically.

Report successes and episode counts per task and suite, aggregate success rates,
and uncertainty intervals or seed-to-seed variation. Inspect failures and videos
for spatial degradation. Do not infer preserved success rate from output shapes
or CPU tests. No GPU speedup, peak-VRAM reduction, or success-rate preservation
is established until these experiments are completed.

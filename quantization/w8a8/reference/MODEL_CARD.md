---
license: apache-2.0
tags:
  - robotics
  - vision-language-model
  - quantization
  - w8a8
  - bitblas
  - pi0
  - libero
language:
  - en
pipeline_tag: robotics
library_name: transformers
---

# Pi0.5-LIBERO W8A8 Quantized Model

This is a **W8A8 (INT8 weights + INT8 activations)** quantized version of [Pi0.5-LIBERO](https://github.com/Physical-Intelligence/openpi), using [BitBLAS](https://github.com/microsoft/BitBLAS) for efficient INT8 Tensor Core computation.

## Model Description

- **Architecture**: Pi0.5 (PaliGemma 2B VLM + Gemma 300M Action Expert)
- **Quantization**: W8A8 (INT8 weights, INT8 activations) with per-channel weight scales and per-tensor activation scales
- **Backend**: BitBLAS for INT8 Tensor Core acceleration
- **W8A8 Layers**: 180 Linear layers replaced with BitBLASW8A8Linear
- **Model Size**: ~4.90GB (quantized) vs ~6.96GB (FP16)
- **Task**: Robot manipulation (LIBERO benchmark)

## Performance

Tested on LIBERO benchmark:

| Task Suite | Success Rate |
|------------|--------------|
| libero_spatial | 100% (10/10) |

Inference speed on NVIDIA A40:
- First inference: ~30s (BitBLAS kernel compilation/caching)
- Subsequent inference: ~150ms per step
- Memory usage: ~4GB VRAM

## Installation

### Prerequisites

- Python 3.11
- CUDA 12.1+ compatible GPU (tested on NVIDIA A40)
- Linux (Ubuntu 22.04 recommended)

### Step 1: Create Conda Environment

```bash
conda create -n openpi_w8a8 python=3.11 -y
conda activate openpi_w8a8
```

### Step 2: Install PyTorch

```bash
pip install torch==2.7.1 torchvision
pip install 'numpy<2.0.0'
```

### Step 3: Install HuggingFace Packages

```bash
pip install transformers==4.53.2 accelerate safetensors huggingface_hub einops
```

### Step 4: Install BitBLAS and Robot Simulation

```bash
pip install bitblas scipy mujoco matplotlib
pip install robosuite==1.4.1
pip install bddl easydict
pip install 'numpy<2.0.0'
```

### Step 5: Install JAX and Flax

```bash
pip install jax[cuda12]==0.5.3 flax==0.10.2 orbax-checkpoint==0.11.13
```

### Step 6: Install Other Dependencies

```bash
pip install sentencepiece draccus==0.10.0 tyro wandb polars numpydantic augmax \
    beartype==0.19.0 equinox jaxtyping==0.2.36 ml-collections==1.0.0 \
    imageio tqdm-loggable flatbuffers Pillow
```

### Step 7: Install LeRobot

```bash
pip install 'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5' --no-deps
pip install datasets h5py zarr diffusers hydra-core gym jsonlines av torchcodec
```

### Step 8: Clone and Install OpenPI

```bash
git clone https://github.com/JingxuanZhang77/openpi_duquant.git openpi
pip install -e openpi/packages/openpi-client --no-deps
pip install -e openpi --no-deps
```

### Step 9: Clone and Install LIBERO (for evaluation)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
touch LIBERO/libero/__init__.py
pip install -e LIBERO --no-deps
```

### Step 10: Copy Custom Transformers Files

```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r openpi/src/openpi/models_pytorch/transformers_replace/models/* $SITE_PACKAGES/transformers/models/
```

### Step 11: Set Environment Variables

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export OPENPI_DISABLE_TORCH_COMPILE=1
```

## Quick Start

```python
import os
os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"

from openpi.models_pytorch.bitblas_w8a8_layers import load_w8a8_policy

# Load model from HuggingFace (downloads automatically)
policy = load_w8a8_policy(
    "fatdove/pi05-libero-w8a8",
    policy_config_name="pi05_libero",
    enable_tuning=False,
)

print(f"Model loaded! W8A8 layers: {policy._w8a8_layer_count}")  # 180

# Run inference
import numpy as np
obs = {
    "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/state": np.random.randn(8).astype(np.float32),
    "prompt": "pick up the red cube and place it on the blue plate",
}
result = policy.infer(obs)
print(f"Actions shape: {result['actions'].shape}")  # (10, 7)
```

## Run LIBERO Evaluation

After completing the installation, you can run the LIBERO evaluation:

```bash
# Quick test (1 trial per task)
python run_libero_w8a8.py --task-suite libero_spatial --num-trials 1

# Full evaluation (20 trials per task)
python run_libero_w8a8.py --task-suite libero_spatial --num-trials 20
```

## Model Files

- `model.safetensors` - Quantized weights (includes both W8A8 and non-quantized layers)
- `w8a8_config.json` - Quantization configuration (layer names, scales info)
- `assets/` - Normalization statistics for input preprocessing

## Quantization Details

The W8A8 quantization uses:
- **Weight quantization**: Per-channel INT8 with symmetric quantization
- **Activation quantization**: Per-tensor INT8 with dynamic quantization
- **Backend**: BitBLAS Matmul kernels optimized for NVIDIA Tensor Cores

180 Linear layers are quantized, including:
- PaliGemma VLM: All attention (q_proj, k_proj, v_proj, o_proj) and MLP (gate_proj, up_proj, down_proj) layers
- Gemma Expert: All MLP layers

## Troubleshooting

### CUDA/BitBLAS Issues
Make sure CUDA 12.x is installed and LD_LIBRARY_PATH is set:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

### Import Errors
If you get import errors like `cannot import name 'ACT2FN'`, ensure you've copied the transformers_replace files:
```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r openpi/src/openpi/models_pytorch/transformers_replace/models/* $SITE_PACKAGES/transformers/models/
```

### NumPy Version Errors
OpenPI requires numpy<2.0:
```bash
pip install 'numpy<2.0.0'
```

## Citation

If you use this model, please cite:

```bibtex

```

## License

Apache 2.0. See the [OpenPI repository](https://github.com/Physical-Intelligence/openpi) for more details.

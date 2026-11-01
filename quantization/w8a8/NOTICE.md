# W8A8 runtime source and license record

- Official OpenPI baseline: `Physical-Intelligence/openpi`, commit
  `215abfb217dbac7d5f1273282331b9b1866c0479`. Existing `LICENSE` and
  `LICENSE_GEMMA.txt` remain applicable and unchanged.
- Adapted source: [JingxuanZhang77/openpi_duquant](https://github.com/JingxuanZhang77/openpi_duquant),
  commit `7d4d1582297ced3131a7b3f9403f7ce403fd59f9`,
  [bitblas_w8a8_layers.py](https://github.com/JingxuanZhang77/openpi_duquant/blob/7d4d1582297ced3131a7b3f9403f7ce403fd59f9/src/openpi/models_pytorch/bitblas_w8a8_layers.py).
  Its root Apache-2.0 license is retained unmodified in `LICENSE.openpi_duquant`.
  No separate W8A8 author/copyright notice was present in that source file.
  Repository ownership is not asserted to be authorship.
- Local adaptations: `src/openpi/models_pytorch/bitblas_w8a8_layers.py` and
  `src/openpi/policies/w8a8_policy.py`, modified 2026-10-31 and 2026-11-01.
  Retained W8A8 math, BitBLAS configuration, explicit Linear replacement and
  complete checkpoint loading. Added strict tensor/statistics validation,
  verified tied-weight alias restoration, separate raw/packed weights and
  integer kernel probes. The lightweight benchmark restores per-layer operator
  call/dtype counters and target/probe diagnostics from development branch
  `quantization/w8a8-setup` (`723135c7bb11eb3ca3df1c41a9d7474f821f3af7`).
  Removed W8FP16, fallback, download/upload, capture and persistent tuning cache. Official OpenPI model,
  transforms and LIBERO evaluator are unchanged.
- [BitBLAS](https://github.com/microsoft/BitBLAS) is an external MIT-licensed
  dependency, not vendored here. A compatible BitBLAS/TVM build is not verified.
- Public checkpoint: [fatdove/pi05-libero-w8a8](https://huggingface.co/fatdove/pi05-libero-w8a8),
  revision `2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee`, taken from the development
  branch's public-checkpoint inspection. Its model card declares Apache-2.0;
  original OpenPI/Gemma terms remain relevant. Weights and reference metadata
  are not included. Full model loading, CUDA execution and LIBERO success rates
  remain unverified.

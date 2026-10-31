# W8A8 source and license record

- Official OpenPI baseline: `Physical-Intelligence/openpi`, commit
  `215abfb217dbac7d5f1273282331b9b1866c0479`.
- Adapted source: [JingxuanZhang77/openpi_duquant](https://github.com/JingxuanZhang77/openpi_duquant),
  commit `7d4d1582297ced3131a7b3f9403f7ce403fd59f9`.
- Source file: [src/openpi/models_pytorch/bitblas_w8a8_layers.py](https://github.com/JingxuanZhang77/openpi_duquant/blob/7d4d1582297ced3131a7b3f9403f7ce403fd59f9/src/openpi/models_pytorch/bitblas_w8a8_layers.py).
- Upstream root license: Apache License 2.0. An unmodified copy is retained in
  `LICENSE.openpi_duquant`. No separate W8A8 author/copyright notice was present
  in that source file. Repository ownership is not asserted to be authorship.
- Local adaptations: `src/openpi/models_pytorch/bitblas_w8a8_layers.py` and
  `src/openpi/policies/w8a8_policy.py`. Modified on 2026-10-31.
  Retained W8A8 math, BitBLAS configuration, explicit Linear replacement and
  complete checkpoint loading concepts. Removed W8FP16, fallback, download/upload,
  capture and persistent tuning cache. Added strict validation, separate raw/packed
  weights, integer kernel probes and diagnostics; retained official model and policy behavior.
- `scripts/inspect_w8a8.py` is a new diagnostic entry point, not the external LIBERO evaluator.
- Existing OpenPI `LICENSE` and `LICENSE_GEMMA.txt` remain applicable and unchanged.
- BitBLAS is an external runtime dependency, not vendored here. Its upstream project
  is [microsoft/BitBLAS](https://github.com/microsoft/BitBLAS) (MIT).
  A working BitBLAS/TVM build revision is **not verified or pinned** by this port.
- Candidate checkpoint: [fatdove/pi05-libero-w8a8](https://huggingface.co/fatdove/pi05-libero-w8a8).
  Earlier report recorded revision `2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee`.
  No model files were downloaded or inspected for this implementation; checkpoint
  format, provenance and applicable model terms still require verification.

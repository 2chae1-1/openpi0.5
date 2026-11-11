# π0.5 LIBERO lightweight model

The default lightweight policy uses 15% Gemma language FFN pruning, trailing language-padding removal, and two views (base and left wrist). Vision pooling defaults to `none` and the policy encodes 512 image tokens. The CPU encodes the prefix and the selected device runs the 10 denoising steps.

## Set up

Use the OpenPI PyTorch setup in the [repository README](../README.md) and the simulation environment in the [LIBERO README](../examples/libero/README.md). Use a local BF16 PyTorch π0.5 LIBERO checkpoint with its normalization assets. Place the OpenPI PaliGemma tokenizer at `big_vision/paligemma_tokenizer.model` inside the OpenPI cache. Set `OPENPI_DATA_HOME` to select the cache directory.

Build the compact checkpoint in the model environment, from the repository root. Set `DENSE_DIR` to your original checkpoint and `PRUNED_DIR` to a new output directory.

```bash
python -m pruning.run_structured_ffn --checkpoint-dir "$DENSE_DIR" \
  --output-dir "$PRUNED_DIR" --target language --prune-fraction 0.15 \
  --multiple-of 1 --device cpu --skip-forward
```

Use the same dense checkpoint to create the compact checkpoint and to supply the normalization assets. The server checks the pruning target, rate, and layer widths. Checkpoint directories can be moved across machines.

## Serve and evaluate

Run this from the repository root in the model environment.

```bash
PYTHONPATH="$PWD/src:$PWD:$PWD/packages/openpi-client/src" \
  python scripts/serve_libero_lightweight.py final \
    --dense-dir "$DENSE_DIR" --pruned-dir "$PRUNED_DIR" \
    --device cuda:0 --vision-pooling none --port 18765
```

Use `--device cpu` for CPU denoising, or choose a CUDA device visible to the process. To evaluate pooling, serve `final --vision-pooling max` with the same checkpoint. To serve the dense baseline, use `original` and omit `--pruned-dir`.

Run the client in your LIBERO environment from the repository root. Configure MuJoCo rendering for that environment. Match the client comparison model to the server's positional model argument.

```bash
python examples/libero/main.py --args.host 127.0.0.1 --args.port 18765 \
  --args.comparison-model final --args.task-suite-name libero_10 \
  --args.task-id 8 --args.initial-state-id 0 --args.seed 123
```

The server prints `COMPARE_SETTINGS` with its active model settings. The client prints one `ROLLOUT_RESULT` JSON record with the suite, task, initial state, seed, success, and action steps. Record the pooling setting with these results because both pruned variants report the comparison model as `final`.

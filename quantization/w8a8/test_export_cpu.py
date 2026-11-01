"""Small CPU export/load integration tests. No GPU or model download."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "quantization/w8a8"))

from quantize_cpu import quantize_linear_weights
from openpi.models_pytorch.bitblas_w8a8_layers import replace_w8a8_layers
from openpi.policies.w8a8_policy import read_manifest, validate_state_dict


class ExportLoadTest(unittest.TestCase):
    def test_conversion_file_and_existing_layers(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)

                # 0번과 2번만 양자화하고, 1번은 원래 자료형으로 유지합니다.
                original = nn.Sequential(
                    nn.Linear(16, 12),
                    nn.Linear(12, 8),
                    nn.Linear(8, 4, bias=False),
                ).to(dtype=dtype).eval()

                state = {
                    key: value.detach().clone()
                    for key, value in original.state_dict().items()
                }
                targets = ["0", "2"]
                converted = quantize_linear_weights(state, targets)

                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    save_file(
                        {
                            key: value.contiguous().clone()
                            for key, value in converted.items()
                        },
                        str(path / "model.safetensors"),
                    )
                    (path / "w8a8_config.json").write_text(
                        json.dumps({
                            "quantization": "w8a8_bitblas",
                            "complete_model": True,
                            "model_type": "CPU_Test_Only",
                            "w8a8_layers": targets,
                        }),
                        encoding="utf-8",
                    )

                    # 실제 파일과 기존 manifest reader를 사용합니다.
                    names = read_manifest(path)
                    loaded = load_file(
                        str(path / "model.safetensors"), device="cpu"
                    )
                    self.assertEqual(names, targets)

                    restored = nn.Sequential(
                        nn.Linear(16, 12),
                        nn.Linear(12, 8),
                        nn.Linear(8, 4, bias=False),
                    ).to(dtype=dtype).eval()

                    replace_w8a8_layers(restored, names)
                    validate_state_dict(restored, loaded, names)
                    restored.load_state_dict(loaded, strict=True)

                    restored_state = restored.state_dict()
                    self.assertEqual(set(restored_state), set(converted))

                    # 저장 전후 숫자와 자료형이 그대로인지 확인합니다.
                    for key, expected in converted.items():
                        self.assertEqual(
                            restored_state[key].dtype, expected.dtype, key
                        )
                        self.assertTrue(
                            torch.equal(restored_state[key], expected), key
                        )

                    for name in targets:
                        layer = restored.get_submodule(name)
                        self.assertEqual(layer.qweight.dtype, torch.int8)
                        self.assertEqual(
                            layer.weight_scale.dtype, torch.float32
                        )
                        self.assertEqual(
                            tuple(layer.qweight.shape),
                            tuple(state[name + ".weight"].shape),
                        )

                    self.assertIsNone(restored[2].bias)
                    self.assertTrue(torch.equal(
                        restored[0].bias, state["0.bias"]
                    ))
                    self.assertTrue(torch.equal(
                        restored[1].weight, state["1.weight"]
                    ))
                    self.assertTrue(torch.equal(
                        restored[1].bias, state["1.bias"]
                    ))
                    print(
                        f"통과: {dtype} → INT8 변환 → 실제 파일 저장"
                        " → 기존 계층 로딩 / 비대상·bias 보존",
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

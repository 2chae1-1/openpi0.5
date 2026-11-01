# pi05_libero Action Expert structured FFN pruning

이 실행기는 원본 dense PyTorch 체크포인트를 공식 OpenPI 로더로 읽고, 공식 BF16 선택 변환 후 Action Expert의 FFN 중간 채널만 물리적으로 제거합니다. 학습, gradient, calibration 데이터는 사용하지 않습니다. 제공된 `structured_ffn.py`의 가중치 크기 휴리스틱을 사용하며, LIBERO 정확도를 보장하거나 SPAR/Wanda를 재현한 구현은 아닙니다. 기존 모델 로더와 W8A8 구현은 수정하지 않습니다.

## 먼저 CPU 테스트

저장소 루트에서 실행합니다. 아래 테스트는 PyTorch만 필요하며 실제 체크포인트를 사용하지 않습니다.

```powershell
python -m unittest pruning.test_structured_ffn -v
python -m unittest pruning.test_run_structured_ffn -v
```

두 번째 테스트는 작은 BF16 모형과 모의 로더로 실행기, 메모리 계산, JSON 저장, 디스크 state_dict 복원을 검증합니다. 공식 체크포인트나 GPU 검증으로 간주하지 않습니다.

## 원본 PyTorch 체크포인트 준비

실제 실행에는 저장소의 Python 3.11 이상 환경과 README의 **PyTorch Support** 설치 절차가 필요합니다. 특히 저장소에서 지정한 `transformers==4.53.2` 및 `transformers_replace` 패치가 필요합니다. 실행기는 라이브러리나 환경을 자동 변경하지 않습니다.

입력은 `model.safetensors`를 포함한 원본 dense `pi05_libero` 디렉터리입니다. 공개 `gs://openpi-assets/checkpoints/pi05_libero`에 JAX `params`만 있다면, 먼저 저장소의 공식 변환기를 사용합니다. 아래 명령의 `--checkpoint-dir`는 **JAX 체크포인트의 params 디렉터리**입니다.

```powershell
uv run examples/convert_jax_model_to_pytorch.py --checkpoint-dir /path/to/pi05_libero/params --config-name pi05_libero --output-path checkpoints/pi05_libero_pytorch --precision bfloat16
```

`model.safetensors`가 없는 입력에 JAX 로더로 우회하지 않습니다. 로컬 디렉터리 외에 OpenPI `download.maybe_download`가 지원하는 URI도 받을 수 있습니다. URI는 실제 실행 시 다운로드하므로, 이미 변환된 로컬 파일 사용을 권장합니다. 파일이 실제로 공식 원본에서 변환되었는지는 사용자가 준비한 입력에 달려 있습니다.

## 10% 제거 실행

```powershell
uv run python -m pruning.run_structured_ffn --checkpoint-dir checkpoints/pi05_libero_pytorch --output-dir checkpoints/pi05_libero_ffn10 --device cuda
```

실행 순서는 `get_config("pi05_libero")` → `config.model.load_pytorch(config, model.safetensors)` → `to_bfloat16_for_selected_params("bfloat16")` → 원본 출력 검사 → `prune_pi05_ffns(target="expert")` → 저장 → 압축 출력 검사입니다. `src/openpi/policies/policy_config.py`와 같은 로딩 및 BF16 경로를 사용합니다. 모듈을 바꾸기 때문에 `pytorch_compile_mode`는 끕니다. BF16 안정성 처리를 위해 공식 코드가 유지하는 FP32 가중치는 그대로 유지합니다.

기본값은 `--prune-fraction 0.10 --multiple-of 1`입니다. `gemma_300m`의 FFN 4096개 중 410개를 제거하여 3686개를 유지합니다. 실제 제거율은 약 **10.009765625%**이며 전체 모델 파라미터의 10%를 제거한다는 뜻은 아닙니다. `--multiple-of 64`를 사용하면 3712개를 유지하여 **9.375%**를 제거합니다. 각 레이어의 실제 제거율과 유지 인덱스를 저장합니다.

각 MLP에서 `gate_proj`, `up_proj`의 동일한 출력 행과 `down_proj`의 동일한 입력 열을 함께 제거합니다. 비전, 언어 모델, attention, action 입출력 projection은 압축 대상이 아닙니다. W8A8로 교체된 모듈에는 적용하지 마세요.

출력 검사는 고정 noise와 synthetic 이미지·BOS 토큰을 사용한 `sample_actions` 호출입니다. 기본 배치 1에서 원시 모델 출력 `[1, 10, 32]`를 기대하지만, JSON에는 실제 실행으로 얻은 형태만 기록합니다. LIBERO 정책 후처리의 `[10, 7]` 출력이나 실제 환경 성공률을 측정하지 않습니다. 기본 denoising step은 10이며 `--num-steps`로 조절할 수 있습니다.

## 산출물과 상태

매 실행마다 새로운 `--output-dir`를 사용합니다. 기존 산출물이 있으면 실패 기록이 이전 성공 결과와 섞이지 않도록 덮어쓰지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `report.json` | 전후 전체/Expert FFN 파라미터 수, dtype별 가중치 바이트·MiB, FFN 차원, 실제 출력 형태, 단계별 검증 상태 및 오류 |
| `pruning_manifest.json` | 원본 모델 설정, 레이어별 새 차원과 유지 인덱스, 실제 제거율, state_dict 복원 정보 |
| `pruned_state_dict.pt` | 물리적으로 크기가 줄어든 전체 모델 state_dict. 기본으로 저장 |

가중치 메모리는 중복을 제거한 파라미터의 `numel() * element_size()` 합계입니다. BF16/FP32 혼합 dtype을 반영하며 activation, KV cache, CUDA allocator 메모리 및 최대 VRAM 사용량은 포함하지 않습니다. 직렬화 파일 크기는 별도로 기록합니다.

`completed`(종료 코드 0)는 압축 및 전후 synthetic 출력 검사가 완료된 경우입니다. `partial` 또는 `unverified`(코드 2)는 일부 검사가 수행되지 않았거나 실행 조건이 부족한 경우입니다. 실행 중 오류는 `failed`(코드 1)로 기록합니다. 의존성/체크포인트 부재 등으로 로딩이 안 되면 전후 수치는 `null`, 출력 형태는 `null`, 가중치 산출물은 `not_saved`입니다. 실제 측정값 대신 추정값을 채우지 않습니다. CLI 인자 오류와 기존 출력 디렉터리 충돌은 실행 시작 전에 중단합니다.

모델을 실행하지 않고 환경만 검사하려면 다음과 같이 실행합니다. 이 경우 다운로드하지 않으며 결과는 `unverified`입니다.

```powershell
python -m pruning.run_structured_ffn --checkpoint-dir checkpoints/pi05_libero_pytorch --output-dir checkpoints/pi05_libero_preflight --preflight-only
```

GPU 없이 충분한 RAM으로 압축·저장만 시도하려면 `--device cpu --skip-forward`를 사용합니다. BF16 가중치를 유지하고 출력 검증은 미검증으로 남깁니다. `--no-save-state-dict`를 사용하면 JSON만 저장하며 해당 manifest만으로는 모델을 복원할 수 없습니다. 전체 체크포인트 로딩·저장에는 상당한 RAM과 디스크 공간이 필요합니다.

## 압축 모델 복원

동일한 OpenPI/Transformers 환경에서 다음 함수를 사용합니다. 원본 체크포인트를 다시 다운로드하지 않습니다.

```python
from pruning.run_structured_ffn import load_pruned_model

model = load_pruned_model("checkpoints/pi05_libero_ffn10", device="cuda")
# 준비된 observation/noise로 model.sample_actions(...) 호출
```

새 `PI0Pytorch` 생성 → 공식 BF16 변환 → `prepare_pi05_pruned_shapes` → `torch.load(weights_only=True)` → `load_state_dict(strict=True)` 순서입니다. 기존 tied weight의 파라미터 공유 관계를 보존하며, 복원 중 새 모델과 state_dict를 위한 RAM이 필요합니다. manifest는 제거된 채널의 가중치를 복구하는 파일이 아니라 **압축 구조를 다시 만드는 파일**입니다. 일반 `create_trained_policy`는 압축 구조를 알지 못하므로 이 state_dict를 원본 `model.safetensors` 대신 넣을 수 없습니다. 복원 함수는 모델만 반환하며 정책의 tokenizer, norm assets, 전후처리를 별도로 구성해야 합니다.

실제 체크포인트의 복원 후 추론, GPU 성능, LIBERO 정확도는 이 작업 환경에서 미검증입니다. 실행 보고서에서도 전체 모델의 복원 roundtrip과 LIBERO 평가를 별도 미검증 상태로 남깁니다.

## 이 환경의 검증 기록

`verification/report.json`은 이 작업 환경에서 `--preflight-only`를 실제 실행한 결과입니다. Python 3.10, CPU 전용 PyTorch, OpenPI 의존성 및 로컬 체크포인트 부재를 기록합니다. 제공된 CPU 테스트 및 추가 CPU 통합 테스트 결과는 `verification/cpu_tests.json`에 저장합니다. 기존 `cpu_proof_result.json`은 제공된 25% toy/분석 예제이며, 본 실행기의 공식 모델 10% 압축 결과가 아닙니다.

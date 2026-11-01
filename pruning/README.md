# pi05_libero Action Expert structured FFN pruning

이 실행기는 원본 dense PyTorch 체크포인트를 공식 OpenPI 로더로 읽고, 공식 BF16 선택 변환 후 Action Expert의 FFN 중간 채널만 물리적으로 제거합니다. 기존 `structured_ffn.py`의 **Magnitude Pruning을 Baseline으로 유지**하며 새로운 pruning 알고리즘은 추가하지 않습니다. 학습, gradient, calibration 데이터는 사용하지 않습니다. LIBERO 정확도를 보장하거나 SPAR/Wanda를 재현한 구현은 아닙니다. 공식 OpenPI 코드와 기존 W8A8 구현은 수정하지 않습니다.

## 먼저 CPU 테스트

저장소 루트에서 실행합니다. 아래 테스트는 PyTorch만 필요하며 실제 체크포인트를 사용하지 않습니다.

```powershell
python -m unittest pruning.test_structured_ffn -v
python -m unittest pruning.test_run_structured_ffn -v
python -m unittest pruning.test_lm_head_audit -v
```

두 번째 테스트는 작은 BF16 모형과 모의 원본 로더로 실행기, 메모리 계산, JSON 저장을 검증하고, 공개 `load_pruned_model()`을 통해 디스크 state_dict를 복원해 Action을 비교합니다. 사용할 수 없는 공식 모델 클래스와 설정만 작은 fixture로 대체합니다. 복원 실패와 출력 불일치가 성공으로 기록되지 않는지도 검사합니다. 세 번째 테스트는 `lm_head`의 dtype별 메모리, 공유 파라미터 및 호출 관측을 검사합니다. 이 CPU 테스트는 공식 체크포인트나 GPU 검증으로 간주하지 않습니다.

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

실행 순서는 `get_config("pi05_libero")` → `config.model.load_pytorch(config, model.safetensors)` → `to_bfloat16_for_selected_params("bfloat16")` → 원본 출력 검사 → `prune_pi05_ffns(target="expert")` → 저장 → 압축 출력 검사 → `load_pruned_model()` 복원 → 복원 모델 출력 비교입니다. `src/openpi/policies/policy_config.py`와 같은 로딩 및 BF16 경로를 사용합니다. 모듈을 바꾸기 때문에 `pytorch_compile_mode`는 끕니다. BF16 안정성 처리를 위해 공식 코드가 유지하는 FP32 가중치는 그대로 유지합니다.

기본값은 `--prune-fraction 0.10 --multiple-of 1`입니다. `gemma_300m`의 FFN 4096개 중 410개를 제거하여 3686개를 유지합니다. 실제 제거율은 약 **10.009765625%**이며 전체 모델 파라미터의 10%를 제거한다는 뜻은 아닙니다. `--multiple-of 64`를 사용하면 3712개를 유지하여 **9.375%**를 제거합니다. 각 레이어의 실제 제거율과 유지 인덱스를 저장합니다.

각 MLP에서 `gate_proj`, `up_proj`의 동일한 출력 행과 `down_proj`의 동일한 입력 열을 함께 제거합니다. 비전, 언어 모델, attention, action 입출력 projection은 압축 대상이 아닙니다. W8A8로 교체된 모듈에는 적용하지 마세요.

출력 검사는 고정 noise와 synthetic 이미지·BOS 토큰을 사용한 `sample_actions` 호출입니다. 기본 배치 1에서 원시 모델 출력 `[1, 10, 32]`를 기대하지만, JSON에는 실제 실행으로 얻은 형태만 기록합니다. LIBERO 정책 후처리의 `[10, 7]` 출력이나 실제 환경 성공률을 측정하지 않습니다. 기본 denoising step은 10이며 `--num-steps`로 조절할 수 있습니다.

복원 검증에서는 저장 직전과 같은 압축 모델의 Action을 기준으로, **동일 observation·noise·denoising step·device**에서 복원 모델의 Action을 생성합니다. 두 출력의 형태, dtype, 유한값을 확인하고 `torch.testing.assert_close(rtol=1e-5, atol=1e-6)`로 비교합니다. `verification.restore_roundtrip`에 비교 상태, 허용오차와 최대 절대 차이를 기록하고 `outputs.restored`에 실제 출력 정보를 기록합니다. 원본과 압축 모델 사이에는 형태만 비교하므로, 복원 일치 검증이 pruning으로 인한 정확도 손실을 검증하는 것은 아닙니다.

## Action Expert의 `lm_head`

로컬 공식 코드의 `PI0Pytorch.sample_actions()`는 `denoise_step()`에서 `PaliGemmaWithExpertModel.forward()`를 호출합니다. suffix 경로는 `gemma_expert.model.forward()`로 hidden state를 구하고 `action_out_proj`로 Action을 생성합니다. `GemmaForCausalLM` wrapper의 `forward()`를 거치지 않으므로 이 경로에서는 `gemma_expert.lm_head`를 호출하지 않습니다. 근거는 `src/openpi/models_pytorch/pi0_pytorch.py`의 `sample_actions`/`denoise_step`, `src/openpi/models_pytorch/gemma_pytorch.py`의 suffix-only 분기 및 `transformers_replace/models/gemma/modeling_gemma.py`의 `GemmaForCausalLM.forward`입니다.

실행기는 원본·압축·복원 `sample_actions` 호출마다 임시 forward-pre-hook으로 head 호출 수를 관측하고, 각 `outputs.*.lm_head_usage`에 `called_on_observed_sample_actions` 또는 `not_called_on_observed_sample_actions`와 `forward_calls`를 기록합니다. 생략하거나 실패한 호출은 미검증으로 남깁니다. 이 관측은 실행한 synthetic 추론 경로에 한정됩니다. **`lm_head`는 제거하지 않고 저장·복원합니다.**

실제 로드한 모델의 `before.action_expert_lm_head`와 `after.action_expert_lm_head`에 파라미터 수, dtype별 바이트와 MiB를 보고합니다. `shared_with_rest`와 `exclusive`는 모델의 다른 부분과 공유하는 파라미터를 구분합니다. head 메모리는 전체 모델 메모리에 이미 포함되므로 두 값을 더하지 않습니다. 공유 파라미터가 있으면 head의 전체 바이트가 제거 시 회수 가능한 메모리와 같지 않을 수 있습니다.

실제 체크포인트를 로드하지 않은 이 환경에서 아래 값은 **공식 설정으로 계산한 추정값**입니다. `gemma_300m`의 hidden width 1024, vocabulary 257152, bias 없는 head를 기준으로 합니다. 보고서의 실측값을 대신 채우지 않습니다.

| 항목 | 설정 기반 추정값 |
| --- | --- |
| `lm_head` 파라미터 수 | 263,323,648 |
| BF16 가중치 메모리 | 526,647,296 bytes / 502.25 MiB |
| FP32 가중치 메모리 | 1,053,294,592 bytes / 1004.5 MiB |

## 산출물과 상태

매 실행마다 새로운 `--output-dir`를 사용합니다. 기존 산출물이 있으면 실패 기록이 이전 성공 결과와 섞이지 않도록 덮어쓰지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `report.json` | 전후 전체/Expert FFN/`lm_head` 파라미터 수와 가중치 바이트·MiB, FFN 차원, 원본·압축·복원 출력, head 호출 관측, 복원 비교 및 오류 |
| `pruning_manifest.json` | 원본 모델 설정, 레이어별 새 차원과 유지 인덱스, 실제 제거율, state_dict 복원 정보 |
| `pruned_state_dict.pt` | 물리적으로 크기가 줄어든 전체 모델 state_dict. 기본으로 저장 |

가중치 메모리는 동일한 파라미터 객체의 중복을 제거한 `numel() * element_size()` 합계입니다. 서로 다른 Parameter 객체가 같은 storage를 공유하는 경우까지 중복 제거하지는 않습니다. BF16/FP32 혼합 dtype을 반영하며 activation, KV cache, CUDA allocator 메모리 및 최대 VRAM 사용량은 포함하지 않습니다. 직렬화 파일 크기는 별도로 기록합니다.

`completed`(종료 코드 0)는 압축, 원본·압축 출력 형태 검사와 **저장한 모델의 복원 후 Action 일치 검증까지 통과**한 경우입니다. `partial` 또는 `unverified`(코드 2)는 일부 검사가 수행되지 않았거나 실행 조건이 부족한 경우입니다. 예를 들어 `--skip-forward`, 원본 forward 실패 또는 `--no-save-state-dict` 사용 시 복원 roundtrip은 미검증이며 `completed`가 될 수 없습니다. 복원 단계의 의존성 오류, 파일 부재, state_dict 로드 실패, 추론 실패, 출력 불일치는 `restore_roundtrip.status="failed"`, 전체 `failed`(코드 1)로 기록합니다. 저장에 성공했더라도 복원에 실패하면 성공으로 기록하지 않습니다.

최초 모델 로딩 전 의존성/체크포인트 부재로 실행할 수 없으면 전후 수치는 `null`, 출력 형태는 `null`, 가중치 산출물은 `not_saved`입니다. 실제 측정값 대신 추정값을 채우지 않습니다. CLI 인자 오류와 기존 출력 디렉터리 충돌은 실행 시작 전에 중단합니다.

모델을 실행하지 않고 환경만 검사하려면 다음과 같이 실행합니다. 이 경우 다운로드하지 않으며 결과는 `unverified`입니다.

```powershell
python -m pruning.run_structured_ffn --checkpoint-dir checkpoints/pi05_libero_pytorch --output-dir checkpoints/pi05_libero_preflight --preflight-only
```

GPU 없이 충분한 RAM으로 압축·저장만 시도하려면 `--device cpu --skip-forward`를 사용합니다. BF16 가중치를 유지하고 출력 검증과 복원 roundtrip은 미검증으로 남깁니다. `--no-save-state-dict`를 사용하면 JSON만 저장하며 해당 manifest만으로는 모델을 복원할 수 없습니다. 전체 체크포인트 로딩·저장·복원에는 상당한 RAM과 디스크 공간이 필요합니다.

## 압축 모델 복원

동일한 OpenPI/Transformers 환경에서 다음 함수를 사용합니다. 원본 체크포인트를 다시 다운로드하지 않습니다.

```python
from pruning.run_structured_ffn import load_pruned_model

model = load_pruned_model("checkpoints/pi05_libero_ffn10", device="cuda")
# 준비된 observation/noise로 model.sample_actions(...) 호출
```

새 `PI0Pytorch` 생성 → 공식 BF16 변환 → `prepare_pi05_pruned_shapes` → `torch.load(weights_only=True)` → `load_state_dict(strict=True)` 순서입니다. 기존 tied weight의 파라미터 공유 관계를 보존하며, 복원 중 새 모델과 state_dict를 위한 RAM이 필요합니다. manifest는 제거된 채널의 가중치를 복구하는 파일이 아니라 **압축 구조를 다시 만드는 파일**입니다. 일반 `create_trained_policy`는 압축 구조를 알지 못하므로 이 state_dict를 원본 `model.safetensors` 대신 넣을 수 없습니다. 복원 함수는 모델만 반환하며 정책의 tokenizer, norm assets, 전후처리를 별도로 구성해야 합니다.

실행기는 위 복원 함수를 직접 호출해 roundtrip을 검증합니다. 다만 이 작업 환경에서는 실제 체크포인트의 복원 후 추론, GPU 실행·성능 및 LIBERO 정확도를 검증하지 않았습니다. CPU fixture 테스트의 성공과 실제 모델 검증을 구분하며, 수행하지 않은 검증은 미검증으로 남깁니다.

## 이 환경의 검증 기록

`verification/report.json`은 이번 수정 후 `--preflight-only`를 다시 실행한 기록이며 Python 3.10, CPU 전용 PyTorch, OpenPI 의존성 및 로컬 체크포인트 부재를 기록합니다. 이 기록의 미검증 상태를 실제 모델 성공 결과로 해석하지 않습니다. `verification/cpu_tests.json`에는 위 세 CPU 테스트 suite의 실행 환경, 결과 및 검증 범위를 저장합니다. 기존 `cpu_proof_result.json`은 제공된 25% toy/분석 예제이며, 본 실행기의 공식 모델 10% 압축 결과가 아닙니다.

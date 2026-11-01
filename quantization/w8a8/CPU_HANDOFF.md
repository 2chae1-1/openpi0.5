# W8A8 CPU 변환 코드 전달

> 2026-11-01: 현재 우선 경로는 공개 180계층 checkpoint 재사용이다.
> [PUBLIC_CHECKPOINT.md](PUBLIC_CHECKPOINT.md)의 다운로드 → 로딩 → INT8 → infer 절차를 따른다.
> 아래 252개 FP exporter와 기록은 보존하며 공개 W8A8 checkpoint를 재양자화하지 않는다.

## 기준 코드와 환경
- 기준 커밋: 3c85dfde7f03cd15172745c4c8c71c70f503f3f1
- Python: 3.13.18
- PyTorch: 2.11.0+cpu
- NumPy: 2.1.3
- safetensors: 0.8.0

## 추가한 파일
- quantize_cpu.py: 명시한 Linear 가중치를 CPU에서 INT8로 변환하는 함수
- test_export_cpu.py: 실제 safetensors 저장과 기존 W8A8 계층 로딩 연결 테스트

## 사용 범위
quantize_linear_weights(state, layer_names)에 CPU state_dict와
명시적인 대상 레이어 이름 목록을 전달합니다.
대상의 .weight를 .qweight 및 FP32 .weight_scale로 바꾼 새 dict를 반환합니다.
bias와 대상 이외의 텐서는 유지하며 입력 dict는 변경하지 않습니다.

이 함수는 모델 구조에서 레이어를 자동 선택하지 않습니다.
이 함수 자체는 전체 체크포인트 내보내기, 정규화 통계 복사, 모델 구조 검증을 포함하지
않습니다. 별도 `export_cpu.py`가 이 절차를 연결합니다(아래 후속 절차 참고).

## 확인한 내용
- 작은 모델에서 INT8 변환·파일 저장·재로딩
- FP32, FP16, BF16 원본에서 기존 W8A8 계층으로 엄격한 로딩
- 양자화 대상 이외의 가중치 및 bias 보존
- 0인 가중치와 bias 없는 레이어 처리
- 빈 목록·중복·없는 레이어·잘못된 크기·정수 원본·NaN·무한대·중복 양자화 거부
  (이 항목들은 Colab에서 별도 소형 검사로 확인)
- 기존 test_validation.py와 새 test_export_cpu.py의 통합 실행 통과

## 재실행
저장소 루트에서:
python -B -m unittest discover -s quantization/w8a8 -p "test*.py" -v

## 완료하지 않은 내용
- 실제 pi0.5 가중치 변환 및 전체 체크포인트 생성
- 실제 양자화 대상 레이어 이름·개수 확정
- 전체 pi0.5 전용 로더 실행과 정규화 통계 확인
- 실제 BitBLAS 계층 실행, LIBERO 정확도, 속도, 메모리, Jetson 검증

## 팀원에게 전달할 주의점
이 결과는 작은 CPU 모델의 변환·저장·로딩 검증입니다.
실제 pi0.5의 용량 감소율·정확도·속도 결과로 해석하지 않습니다.
전체 모델 적용 전 대상 레이어와 원본 체크포인트를 정해야 합니다.

## 후속: GPU 서버에서 실제 pi05_libero 검증

아래 코드는 실행 준비 단계입니다. Windows에서는 실제 모델 생성, 체크포인트 다운로드/
로딩, exporter, 커널 초기화, 추론을 실행하지 않았습니다. 위 CPU 소형 테스트 결과와
아래 실제 체크포인트 검증 상태를 구분합니다. 현재 실제 체크포인트 검증은 모두 미실행입니다.

### 입력과 환경

- 저장소 루트에서 서버의 기존 OpenPI 가상환경을 활성화합니다. Python >=3.11,
  저장소에 지정된 PyTorch 2.7.1 / Transformers 4.53.2와 공식 `transformers_replace`
  적용, safetensors, CUDA 및 BitBLAS/TVM이 필요합니다.
- BitBLAS의 호환 버전은 아직 확정하지 않았습니다. README의 API 계약을 확인하고
  실제 사용한 버전/빌드 커밋, GPU/driver, 저장소 커밋(`git rev-parse HEAD`)을 기록합니다.
- PaliGemma tokenizer 캐시도 미리 준비합니다. checkpoint 로더는 로컬 파일만 받지만,
  공식 policy transform 생성은 tokenizer를 download helper로 준비할 수 있습니다.
- 충분한 호스트 RAM과 VRAM, 출력 디스크 공간이 필요합니다. export는 CPU 연산이지만
  전체 FP/INT8 checkpoint를 다룹니다. 실제 peak 메모리는 미측정입니다.
- 입력은 `pi05_libero`의 **FP PyTorch** `model.safetensors`와
  `assets/physical-intelligence/libero/norm_stats.json`이 들어 있는 디렉터리입니다.
  JAX `params/`만 있는 디렉터리나 이미 W8A8인 가중치는 exporter 입력이 아닙니다.
  JAX 원본은 저장소의 `examples/convert_jax_model_to_pytorch.py`로 별도 변환해야 합니다.
  해당 변환과 자산 배치도 여기서는 실행 검증하지 않았습니다.

### 서버 실행 명령

다음 경로를 서버의 실제 경로로 바꿉니다. `OUT`과 report 파일은 존재하지 않아야 하며
부모 디렉터리는 미리 준비합니다. shell pipeline 없이 실행하여 실패 exit code를 확인합니다.

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
FP=/srv/checkpoints/pi05_libero_fp
OUT=/srv/checkpoints/pi05_libero_w8a8_validation

# ① FP 전체 tensor 로딩/키·shape·유한값 검사 및 norm 통계 검사
# ② 공식 모델의 meta 구조와 후보 252개 nn.Linear 대조
# ③ 기존 CPU quantizer 재사용 → 저장 → 모든 tensor exact round-trip → 재로딩 검증
python quantization/w8a8/export_cpu.py \
  --source "$FP" \
  --targets quantization/w8a8/targets_pi05_libero_candidate.json \
  --output "$OUT"

# 선택적인 파일 헤더 진단. 이것만 통과한 것은 전체 모델 로딩 성공이 아님.
python scripts/inspect_w8a8.py "$OUT"

# ④ 저장한 checkpoint를 전용 로더로 strict load, 모든 GPU 커널 probe 후 실제 추론
# 초기 진단은 torch.compile 비활성화 고정. 기본 denoising 10 steps.
python scripts/validate_w8a8_forward.py "$OUT" \
  --device cuda:0 --report /srv/checkpoints/pi05_libero_w8a8_smoke.json
```

각 명령은 이전 명령 성공 후 실행합니다. exporter는 252개 레시피와 후보 JSON의 일치를
검사한 뒤 실제 모델에서 모든 이름이 `nn.Linear`인지 확인합니다. 실제 차원은 성공한
출력의 `w8a8_config.json`의 `validated_target_shapes`에 남습니다. 개수나 이름이 다르면
그 자리에서 중단하며 이름을 자동 추정하거나 무시하지 않습니다. 후보 원본 JSON의
`candidate_not_model_validated` 상태는 실행 전 가정을 나타내므로 임의로 성공으로 바꾸지 않습니다.

exporter의 구조 검사는 meta 모델을 사용합니다. FP checkpoint 전체 tensor를 읽어
구조와 대조하지만 FP 원본 모델에 materialize하여 추론한 검증은 아닙니다.
정해진 non-persistent buffer만 제외하고, 실제 모델에서 PaliGemma lm_head/embedding이
공유될 때만 생략된 한쪽을 다른 쪽에서 복원합니다. 둘 다 있는데 값이 다르거나 나머지
키가 누락/추가되면 실패합니다. 출력 manifest의 검증 플래그는 실제로
실행한 검사만 기록합니다. `full_model_load_executed=false`, `gpu_kernel_executed=false`는
exporter 범위이며, 이후 GPU 실행 결과는 별도 smoke report에서 확인합니다.

저장 파일은 raw INT8 weight와 FP32 scale입니다. packing은 GPU 로더에서 수행하며
packed weight를 재저장하지 않습니다. 저장 중 실패한 `.partial-*` 디렉터리는 진단용으로
남고, 기존 입력/출력을 덮어쓰지 않습니다. 출력이 완성되어도 GPU 검증 성공을 뜻하지 않습니다.

### 실제 관측 입력과 통과 기준

기본 입력은 고정 seed의 합성 이미지와 0 state입니다. **실제 checkpoint의 forward 경로**를
실행하는 smoke test이며 실제 LIBERO episode나 정확도 검증이 아닙니다.
실제 관측을 쓰려면 pickle 없는 NPZ에 아래 세 키를 저장하여 전달합니다.

- `observation/state`: 유한한 `(8,)` 벡터
- `observation/image`, `observation/wrist_image`: HWC RGB `uint8` 이미지

```bash
python scripts/validate_w8a8_forward.py "$OUT" \
  --device cuda:0 --observation /srv/observations/libero_obs.npz \
  --prompt "pick up the book" \
  --report /srv/checkpoints/pi05_libero_w8a8_observation.json
```

성공 기준은 exit code 0과 report의 아래 세 항목이 모두 `passed`인 것입니다.

- `strict_model_load_and_kernel_probes`: 저장 checkpoint의 전체 키/shape/value 검증,
  실제 모델 strict load 및 모든 계층의 BitBLAS packing/INT8 probe 완료
- `policy_infer`: 공식 입력/정규화/토큰화/샘플링/출력 변환을 통한 추론 완료,
  actions가 `(action_horizon, 7)`이고 모두 유한함(현재 config는 `(10, 7)`)
- `all_target_layers_called`: 모든 대상의 `forward_calls`가 추론 전보다 증가함

실패 시 report의 `error`, 단계 상태와 원래 traceback을 보존합니다. 실패 이후 단계는
`not_run`으로 남습니다. 초기 환경/입력 검사 실패라면 모델 로딩도 `not_run`입니다.
CUDA 커널 오류나 실제 token 길이의 shape 오류는 FP fallback으로 우회하지 않습니다.
`opt_m` 기본은 `1 16 32 64`이며 실제 펼친 행 수 지원은 서버에서 확인해야 합니다.
필요하면 `--opt-m`에 검증할 값을 명시하고 동일 값을 사용해 재실행합니다.

여전히 별도 검증이 필요한 항목: FP 원본과 수치/정확도 비교, LIBERO 성공률,
profiler의 INT8 가속 증거, latency/peak RAM·VRAM, torch.compile, Jetson Orin 호환성.
합성 입력 또는 실제 관측 한 건의 smoke 통과로 이 항목을 성공 처리하지 않습니다.

### 이번 Windows 점검 범위

수정 Python 소스의 AST/메모리 내 컴파일과 `git diff --check`, 후보 JSON과 명시적
252개 레시피의 일치만 정적으로 확인했습니다. 모델 라이브러리를 import하거나 실행하지
않았습니다. 새 tied alias 회귀 테스트는 추가했으나 이번에는 실행하지 않았습니다.
서버에서는 기존 CPU 회귀 테스트도 위 재실행 명령으로 먼저 실행할 수 있습니다.

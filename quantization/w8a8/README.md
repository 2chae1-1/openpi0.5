# π0.5 LIBERO W8A8 실행 준비

## 목적
공식 OpenPI를 기반으로 공개 W8A8 모델의 실행 경로를 준비한다.
필요한 외부 코드와 의존성만 검토하여 가져온다.

## 현재 상태
- **실제 LIBERO 성공률 평가 절차**: [LIBERO 실행 문서](../../examples/libero/README.md#공개-w8a8-모델로-libero-평가)를 따른다.
  기존 `scripts/serve_policy.py`의 `policy:w8a8`가 전용 로더를 재사용하며,
  `examples/libero/main.py`의 관측 전송·행동 실행·성공률 계산은 그대로 사용한다.
  Windows 정적·CPU 검증만 수행했으며 실제 CUDA·LIBERO 평가는 실행하지 않았다.
- **공개 180계층 checkpoint 재사용을 우선한다.** 확인한 형식, 252개 후보와의 차이,
  다운로드 → 전체 로딩 → INT8 probe → policy.infer 명령 및 통과 기준은
  [PUBLIC_CHECKPOINT.md](PUBLIC_CHECKPOINT.md)를 따른다. 실제 GPU 실행은 미검증이다.
- 외부 W8A8 구현은 재현 검증 후보
- 작업 브랜치: quantization/w8a8-setup
- 양자화 코드 이식: 최소 구현 추가, 정적 검사 완료 (2026-10-31)
- 모델 로딩·추론·LIBERO 평가: 실행 검증 전
- Jetson Orin 호환성·속도·정확도: 실행 검증 전

## 작업 환경
- Windows PC: 코드 편집 및 GitHub 업로드
- GPU 서버: 모델 다운로드, 실행 및 LIBERO 평가
- Orin 보드: 파일·환경·모델·캐시는 ~/ssd 아래에 저장

## 참고 자료
- 공식 코드: https://github.com/Physical-Intelligence/openpi
- 양자화 구현: https://github.com/JingxuanZhang77/openpi_duquant
- 모델: https://huggingface.co/fatdove/pi05-libero-w8a8
- 실제 평가: [기존 LIBERO 평가기](../../examples/libero/main.py)

## 검증 원칙
- 큰 모델 가중치는 GitHub에 올리지 않는다.
- 외부 코드의 출처와 라이선스 고지를 보존한다.
- BitBLAS는 양자화 연산에 사용하는 기존 라이브러리다.
- CUDA 연산 코드를 직접 개발하거나 팀원의 최적화 코드를 임의로 수정하지 않는다.
- 출력이 나오는 것과 실제 INT8 가속이 작동하는 것을 구분한다.
- 정확도와 속도는 같은 평가 조건에서 원본 모델과 비교한다.

## 이번에 추가한 구현

공식 코드 기준 커밋은 `215abfb217dbac7d5f1273282331b9b1866c0479`다.
기존 공식 모델·추론·설정·평가 파일과 의존성 파일은 변경하지 않았다.

- `src/openpi/models_pytorch/bitblas_w8a8_layers.py`: 외부 구현에서 W8A8 연산과
  명시적 Linear 교체만 축소 이식했다. 동적 per-tensor A8, 출력 채널별 W8 scale,
  INT8×INT8 → INT32 BitBLAS 연산 후 실수 출력 복원을 사용한다.
- `src/openpi/policies/w8a8_policy.py`: 공식 `pi05_libero` 전용 로더다.
  이미 준비된 로컬 체크포인트만 받는다. FP 체크포인트 자동 변환, Hub 다운로드,
  재저장·업로드, W8FP16, DuQuant/ATM/SmoothQuant 기능은 포함하지 않는다.
- `scripts/inspect_w8a8.py`: 계층 목록·텐서 헤더 확인과 선택적 GPU 초기화 진단.
- `NOTICE.md`, `LICENSE.openpi_duquant`: 출처·정확한 커밋·수정 사항과 원본 Apache-2.0 라이선스.

공식 `create_trained_policy`에는 연결하지 않았다. W8A8 체크포인트는 서버의
`policy:w8a8 --policy.dir /path/to/checkpoint`로 선택하면 기존 `load_w8a8_policy()`를 사용한다.
기존 FP용 `--env LIBERO`와 `policy:checkpoint` 실행도 유지한다.

## 지원하는 입력 계약과 중단 조건

이 계약은 외부 exporter/loader 소스와 공개 revision의 manifest·safetensors 헤더·통계
JSON에서 확인했다. tensor payload 및 실제 모델/GPU 실행 검증은 남아 있다.

```text
CHECKPOINT_DIR/
  w8a8_config.json
  model.safetensors
  assets/physical-intelligence/libero/norm_stats.json
```

`w8a8_config.json`에는 `quantization="w8a8_bitblas"`, `complete_model=true`,
중복 없는 비어 있지 않은 `w8a8_layers` 목록이 있어야 한다. 계층명을 추정하거나
정규식 기본값으로 대신하지 않는다. 목록의 모든 항목은 공식 모델의 실제 `nn.Linear`여야 한다.
진단에 실제 이름·개수가 출력된다. 로컬 exporter의 후보는 두 transformer stack의
252개 Linear이며 실제 모델 대조 전에는 확정된 대상이 아니다.

각 대상은 raw INT8 `qweight` `[out_features, in_features]`, 양수·유한한 실수
`weight_scale` `[out_features]`, 원본 계층에 bias가 있다면 `bias`가 필요하다.
대상 이외의 **모든** 공식 persistent weight/buffer도 필요하다. 키 누락·추가,
형상 불일치, 잘못된 dtype, 비유한 값은 GPU 초기화 전에 중단한다.
실제 모델이 PaliGemma `lm_head.weight`와 embedding을 공유할 때만 생략된 한쪽 키를
다른 쪽에서 복원한다. 둘 다 존재하면서 값이 다르면 거부한다. 그 외 alias 누락은 중단한다.
공개 180계층 checkpoint의 헤더/manifest는 이 계약과 일치한다. Expert attention
72개는 BF16으로 남기며 후보 252개를 강제로 적용하지 않는다.

정규화 통계는 공식 config의 asset ID에서 읽는다. `state`, `actions` 및
mean/std와 π0.5 quantile 정규화용 q01/q99가 필수다. 각 통계는 유한한 같은 길이의
벡터여야 하며 std ≥ 0, q99 ≥ q01이어야 한다. 공식 LIBERO 입력 8차원·출력 7차원을
포함하고 모델의 32차원 이내여야 한다. 32차원 padding 여부는 추정하지 않는다.
누락 시 빈 dict나 config 외부 통계로 대체하지 않는다.

체크포인트 raw weight와 BitBLAS packing 결과를 분리 보관한다.
`.weight`는 공식 Gemma의 dtype/device 참조용 빈 텐서이며 실수 가중치 행렬이 아니다.
임의로 `.weight`를 행렬 연산이나 재저장에 사용하는 외부 코드는 지원하지 않는다.
실수 scale이 BF16으로 변환되지 않도록 공식 선택적 BF16 변환은 계층 교체 전에 수행한다.

BitBLAS import·커널 생성·packing·INT8 시험 호출 실패는 원인을 연결한 오류로 중단한다.
raw weight나 FP32 matmul로 대체하지 않는다. 각 계층의 초기화는 signed INT8 입력으로
실제 커널을 호출하고 INT64 CPU 기준 연산과 모든 출력 채널을 비교·동기화한다.
이 검사는 초기화 증거이며 모든 실제 token 길이의 실행, profiler의 INT8 명령 확인,
정확도·속도 검증을 대신하지 않는다.

## GPU 서버에서 사용할 진입점 (이번에는 실행하지 않음)

서버의 기존 OpenPI 환경에서 `openpi` 소스가 import 가능해야 한다.
경로는 서버에 이미 준비한 체크포인트로 바꾼다.

```bash
# CPU에서 safetensors 헤더와 manifest의 대상 목록만 확인. 모델/GPU 초기화 없음.
python scripts/inspect_w8a8.py /path/to/checkpoint

# GPU 커널 생성·packing·INT8 시험 연산. 초기 검증에만 compile을 비활성화.
python scripts/inspect_w8a8.py /path/to/checkpoint \
  --initialize --device cuda:0 --disable-compile-for-validation
```

두 번째 명령은 GPU 연산·커널 컴파일을 수행한다. `--enable-tuning`은 별도 선택이며
기본 off여도 BitBLAS 코드 생성·컴파일은 발생할 수 있다. 실제 펼친 행 수에 맞춰
`--opt-m`을 검증해야 한다. 기본값은 외부 구현과 같은 `1 16 32 64`다.

```python
from openpi.policies.w8a8_policy import load_w8a8_policy
from openpi.models_pytorch.bitblas_w8a8_layers import w8a8_diagnostics

policy, model = load_w8a8_policy(
    "/path/to/checkpoint", device="cuda:0", disable_compile_for_validation=True,
)
print(w8a8_diagnostics(model))
# result = policy.infer(observation)  # 공식 LIBERO 형식의 관측 입력을 별도 준비
# print(w8a8_diagnostics(model))      # 실행 후 계층별 forward_calls 확인
```

`disable_compile_for_validation` 기본값은 False다. True일 때만 **로컬 config 복사본**의
`pytorch_compile_mode=None`을 지정한다. 공식 기본 `max-autotune`과 원본 설정 객체는
유지한다. 환경변수 `OPENPI_DISABLE_TORCH_COMPILE`은 도입하지 않았다.
compile 활성 경로와 BitBLAS의 조합은 미검증이다.

진단은 계층 이름·차원·INT8 weight dtype·장치·operator 생성·packing·INT8 probe 성공과
추론 호출 수를 구분한다. 헤더 검사만으로 필수 비양자화 가중치/통계/커널 검증이
완료됐다고 표시하지 않는다. profiler 검증은 항상 별도 미완료로 표시한다.

## 의존성과 서버 실행 전 준비

- 패키지는 이번 작업에서 설치하지 않았다. 프로젝트의 Python >= 3.11,
  PyTorch 2.7.1, Transformers 4.53.2 및 공식 Transformers 교체 파일 적용이 필요하다.
- 직접 사용하는 `safetensors`와 BitBLAS/TVM이 서버 환경에 있어야 한다.
  BitBLAS는 `bitblas.ops.Matmul/MatmulConfig`, `transform_weight`, `backend="tir"`,
  INT8/INT32 조합을 제공해야 한다. **실제 호환 버전·빌드 커밋은 미확정**이므로
  임의 버전을 의존성 파일/lock에 추가하지 않았다. 재현 환경에서 먼저 고정해야 한다.
- 공식 transform factory는 PaliGemma tokenizer를 기존 download helper로 준비한다.
  서버 실행 전에 tokenizer 캐시를 준비한다. 이 전용 로더는 체크포인트를 자동 다운로드하지 않는다.
- 체크포인트 revision·hash·정확한 config와 tensor key 목록, raw/packed 여부,
  scale 형식, 공유 가중치 alias, 실제 대상 계층 개수, 통계 길이·출처를 확인한다.
- 충분한 CPU/GPU 메모리가 필요하다. 현재는 공식 모델 초기 생성과 전체 state dict를
  CPU에 만들며 raw/packed INT8 weight를 둘 다 보관한다. peak RAM/VRAM은 미측정이다.
- 장치별 CUDA/driver/PyTorch/BitBLAS/TVM 환경과 target 감지를 기록한다.
  A40 결과로 Orin의 ARM64/SM87 호환성·속도를 판단하지 않는다.

## 초기 이식 검증 기록 (공개 metadata 확인 전)

새 Python 파일의 AST/바이트코드 컴파일 및 정적 소스 검사를 수행했다.
Python 3.12에서 Python 3.11 문법 기준 AST 검사와 메모리 내 컴파일을 수행했다.
Ruff가 로컬에 없어 Ruff lint/format은 수행하지 않았고 패키지를 설치하지 않았다.
모델 다운로드, 모델 생성, GPU 커널 초기화·추론·LIBERO 평가, 패키지 설치,
Git commit/push는 수행하지 않았다. 기존 공식 추적 파일도 변경하지 않았다.

2026-11-01 후속 점검에서는 공개 manifest·모델 문서·norm JSON·tensor 헤더만 읽었고,
CPU 회귀 13개와 정적 검사가 통과했다. 전체 suite의 파일 저장 테스트는 로컬
safetensors 부재로 실행하지 못했다. 상세 결과는 [PUBLIC_CHECKPOINT.md](PUBLIC_CHECKPOINT.md)에 있다.

공개 체크포인트의 전체 tensor 값과 실제 모델 로딩 호환성, BitBLAS API/packing,
모든 실제 입력 shape의 커널 실행, compile 호환성, 추론 정확도·속도·메모리,
Orin 호환성은 모두 **실행 검증 전**이다. GPU 서버에서 실패 메시지를 확인하고
형식을 확정한 다음 동일 조건의 FP 원본과 LIBERO 정확도·동기화된 latency를 비교한다.

외부 원본과 로컬 수정의 상세 출처는 [NOTICE.md](NOTICE.md)를 참고한다.

## 실제 체크포인트 검증 인계

공개 checkpoint 우선 실행은 [PUBLIC_CHECKPOINT.md](PUBLIC_CHECKPOINT.md)를 따른다.
아래 FP export 절차는 보존한 별도 경로이며 공개 W8A8 checkpoint에는 적용하지 않는다.

로컬 FP 체크포인트 → 구조 대조 → CPU 양자화/저장/파일 재로딩 → GPU 전용 로더와
실제 `policy.infer()` 실행 명령은 [CPU_HANDOFF.md](CPU_HANDOFF.md)의 후속 절차를 따른다.
`scripts/validate_w8a8_forward.py`는 compile을 끈 스모크 테스트이며 정확도·속도 평가가 아니다.

## CPU 검증 회귀 테스트

`weight_scale`은 로딩 대상 dtype으로 변환한 후에도 모든 값이 유한하고 0보다 커야 한다.
변환에 의한 overflow/underflow도 거부한다. 헤더 진단에서 지원하는 scale dtype은
`F16`, `BF16`, `F32`, `F64`이며, 실제 값 검증은 전체 로더에서 수행한다.
검증된 PaliGemma lm_head/embedding 쌍 이외의 공유 가중치 alias 누락은 중단한다.

저장소 루트에서 다음과 같이 실행한다. `unittest`와 기존 CPU PyTorch만 사용한다.

```bash
python -B -m unittest discover -s quantization/w8a8 -p test_validation.py -v
```

헤더 테스트는 `safetensors` reader를 대체하여 실제 CLI의 헤더 검증 분기를 실행한다.
실제 safetensors 파일 읽기, 전체 모델 로딩 또는 GPU 실행 검증을 대신하지 않는다.

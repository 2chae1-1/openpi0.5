# π0.5 W8A8 GPU 벤치마크 인계

목적은 기존 `scripts/benchmark_pi05_fp_w8a8.py`로 W8A8의 GPU 추론시간, VRAM,
체크포인트 크기를 측정하는 것이다. 재학습·재양자화·LIBERO 에피소드/전체 성공률 평가는 실행하지 않는다.
**실제 전체 모델 로딩, CUDA/BitBLAS 실행, GPU 성능 및 LIBERO 검증은 아직 완료하지 않았다.**
Windows CPU 회귀 검사 통과를 GPU 성공으로 해석하면 안 된다.

## 1. 브랜치와 서버 환경 준비

이 변경을 포함한 GitHub 브랜치를 Linux GPU 서버에 checkout하고 저장소 루트로 이동한다.
작성 시 확인한 브랜치는 `handoff/libero-w8a8`다. 작성자가 GitHub Desktop에서 commit/push한 뒤 받는다.

```bash
git fetch origin
git switch handoff/libero-w8a8
git pull --ff-only
git submodule update --init --recursive
```

- Linux, Bash, `tee`, `mktemp`, NVIDIA GPU와 CUDA 사용 가능한 드라이버가 필요하다.
  저장소의 기존 기준 환경은 Python 3.11, PyTorch 2.7.1, Transformers 4.53.2다.
  W8A8의 실제 필요 VRAM은 미측정이므로 특정 GPU에서 수용 가능하다고 가정하지 않는다.
- 기존 OpenPI 환경이 있으면 재사용한다. 새 환경은 저장소의
  [설치 안내](README.md#installation)와 [PyTorch 설정](README.md#pytorch-support)을 따른다.
  저장소 루트의 기본 가상환경을 만들 때 사용할 명령은 다음과 같다.

  ```bash
  GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --python 3.11
  GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv pip install --python .venv/bin/python -e .
  cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
  ```

  `transformers_replace`는 모델이 요구하는 기존 교체 파일이다. 의존성 재설치로 덮어써졌다면 다시 적용한다.
  위 경로는 Python 3.11 `.venv` 기준이다. 다른 환경을 쓰면 실제 설치 위치에 적용하고
  실행 시 `PYTHON_BIN`도 같은 환경으로 지정한다. `UV_LINK_MODE=copy`는 새 설치 시
  Transformers 교체가 uv의 hardlink 캐시에 전파되는 것을 피하기 위한 설정이다.
- **BitBLAS와 호환 TVM/CUDA 빌드는 GPU 담당자가 준비해야 한다.** `pyproject.toml`과 lock에는
  검증된 BitBLAS 조합이 고정되어 있지 않다. 임의 버전을 설치하면 성공한다고 가정하지 않는다.
  기존 구현은 `bitblas.ops.Matmul`, `MatmulConfig`, `backend="tir"`, `transform_weight`,
  INT8×INT8→INT32 연산을 사용한다. 실제 호환성은 커널 생성·packing·정수 probe와 추론으로 확인한다.
- `safetensors`, NumPy, OpenPI 및 `openpi-client` 등 기존 프로젝트 의존성이 필요하다.
  프로세스 전체 VRAM 관측용 `pynvml` 모듈도 확인한다. 없다면 다음처럼 설치할 수 있다.

  ```bash
  uv pip install --python .venv/bin/python nvidia-ml-py
  ```

  NVML이 없거나 드라이버/권한 문제로 관측하지 못하면 기존 벤치마크는 해당 값을 `null`과 이유로 남긴다.
  모델 검증 성공과 NVML 측정 성공은 별도로 확인한다.
- 기존 OpenPI transform은 PaliGemma tokenizer를 사용한다. 기본 캐시는
  `~/.cache/openpi/big_vision/paligemma_tokenizer.model`이며 `OPENPI_DATA_HOME`으로 캐시 루트를 바꿀 수 있다.
  캐시가 없으면 기존 코드가 최초 다운로드를 시도하므로 네트워크가 가능한 준비 단계에서 미리 받는다.

  ```bash
  JAX_PLATFORMS=cpu .venv/bin/python -c 'from openpi.shared.download import maybe_download; print(maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))'
  ```

다음은 기본 의존성·CUDA 가용성 확인이며 실제 모델/INT8 커널 검증을 대신하지 않는다.

```bash
nvidia-smi
JAX_PLATFORMS=cpu .venv/bin/python -c 'import torch, transformers, safetensors, pynvml; from bitblas.ops import Matmul, MatmulConfig; from bitblas.utils import auto_detect_nvidia_target; print(torch.__version__, torch.version.cuda, transformers.__version__); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
```

LIBERO 시뮬레이터, EGL, ffmpeg, 평가용 데이터셋은 이번 고정 관측 추론 측정에 필요하지 않다.

## 2. 로컬 체크포인트 준비

Git 브랜치에는 모델 가중치가 포함되지 않는다. 기본 위치는 다음과 같다.

```text
checkpoints/pi05-libero-w8a8-public/
  model.safetensors
  w8a8_config.json
  assets/physical-intelligence/libero/norm_stats.json
```

이것은 임의 INT8 state dict가 아니라 `pi05_libero` **전체 W8A8 체크포인트**여야 한다.
manifest는 `quantization="w8a8_bitblas"`, `complete_model=true`, 중복 없는 명시적 `w8a8_layers`를 요구한다.
기존 로더가 전체 key·shape·dtype, INT8 qweight, 양수/유한 scale, 정규화 통계와 strict load를 검증한다.
검증을 통과시키려고 manifest를 임의로 만들거나 누락된 가중치를 FP32로 대체하지 않는다.

공개 체크포인트 다운로드 명령과 기존에 기록된 revision은
[기존 체크포인트 안내](examples/libero/README.md#서버-준비)와
[NOTICE](quantization/w8a8/NOTICE.md)에 있다. 이 revision은 기존 조사 기록이며 실제 GPU 호환성 증거는 아니다.
이미 받은 체크포인트를 재사용할 수 있고, 다른 위치면 `W8A8_CHECKPOINT`만 지정한다.
실제 출처를 확인한 경우에만 `W8A8_REVISION`에 해당 revision을 기록한다.
벤치마크와 셸 스크립트는 체크포인트를 자동 다운로드하거나 변환하지 않는다.

## 3. 실행 명령

환경과 체크포인트를 위 기본 위치에 준비했다면 저장소 루트에서 **한 명령**으로 실행한다.
실행 권한을 추가할 필요 없이 Bash가 읽는다.

```bash
bash scripts/run_w8a8_gpu.sh
```

체크포인트 위치만 다르면 다음 한 줄을 사용한다.

```bash
W8A8_CHECKPOINT=/path/to/pi05-libero-w8a8 bash scripts/run_w8a8_gpu.sh
```

다른 Python 환경이나 GPU를 쓰는 예:

```bash
PYTHON_BIN=/path/to/venv/bin/python W8A8_CHECKPOINT=/path/to/pi05-libero-w8a8 DEVICE=cuda:1 bash scripts/run_w8a8_gpu.sh
```

필요한 설정은 환경변수로 지정하거나 스크립트 상단의 기본값을 편집한다. 경로에 공백이 있으면 따옴표로 감싼다.
**상대 경로는 모두 저장소 루트 기준**이며 `PYTHON_BIN`은 인자 없는 실행 파일 하나다.

| 설정 | 기본값 / 의미 |
| --- | --- |
| `W8A8_CHECKPOINT` | `checkpoints/pi05-libero-w8a8-public` |
| `PYTHON_BIN` | `.venv/bin/python` |
| `RUNS` | `3`; 새 프로세스로 수행하는 독립 실험 횟수 |
| `OUTPUT_ROOT` | `data/w8a8-gpu`; 호출마다 고유 batch 폴더 생성 |
| `DEVICE` | `cuda:0`; CPU 실행은 허용하지 않음 |
| `WARMUP`, `ITERATIONS` | `10`, `50`; 실험당 추가 warmup / 측정 추론 호출 수 |
| `NUM_STEPS`, `SEED` | `10`, `42`; denoising steps / 고정 입력·noise seed |
| `PROMPT` | `pick up the book` |
| `OBSERVATION` | 비어 있음; 생략하면 고정 합성 입력, 지정하면 기록 NPZ 재사용 |
| `FP_CHECKPOINT` | 비어 있음; 지정한 경우에만 FP 비교 요청 |
| `W8A8_REVISION`, `FP_REVISION` | 비어 있음; 확인한 출처 revision만 선택적으로 기록 |
| `ENABLE_TUNING` | `0`; `1`이면 기존 `--enable-tuning` 활성화 |

기본 실험 한 번은 최초 추론 1회 + 추가 warmup 10회 + 측정 50회다.
기본 3회는 측정 호출 150회를 포함한다. seed는 각 실험에서 동일하게 유지한다.
batch size는 기존 코드의 1, `--opt-m`은 기존 기본 `[1, 16, 32, 64]`를 그대로 쓴다.
`torch.compile`은 기존 벤치마크가 끄지만 BitBLAS 커널 생성/컴파일은 수행한다.
커널 준비는 오래 걸릴 수 있으며 실제 진행은 각 `reports/benchmark.log`에서 확인한다.
`JAX_PLATFORMS=cpu`는 기존 worker가 OpenPI/JAX import 전에 설정한다. 모델 추론은 CUDA다.

`OBSERVATION` NPZ에는 `observation/state`(유한 `(8,)` 벡터),
`observation/image`, `observation/wrist_image`(비어 있지 않은 HWC `uint8` RGB)가 필요하다.
기존 코드가 `allow_pickle=False`로 읽는다. 고정 관측 반복은 LIBERO 에피소드 평가가 아니다.

### 선택: 로컬 FP와 비교

이미 준비된 **PyTorch FP 전체 체크포인트**에 `model.safetensors`와 같은 정규화 assets가 있을 때만 지정한다.
JAX 전용 체크포인트 경로는 사용할 수 없다. 이 작업을 위해 새로 학습하거나 변환할 필요는 없다.

```bash
W8A8_CHECKPOINT=/path/to/pi05-libero-w8a8 FP_CHECKPOINT=/path/to/pi05-libero-fp bash scripts/run_w8a8_gpu.sh
```

각 실험 안에서 FP → W8A8을 별도 새 프로세스로 순차 실행한다.
`FP_CHECKPOINT`를 생략하면 FP worker를 실행하지 않는다. 명시한 FP 경로에 비어 있지 않은
`model.safetensors`가 없으면 셸이 설정 오류로 중단한다. 이는 기존 Python CLI가 없는 FP를
`not_measured`로 건너뛰는 동작으로 비교 요청이 사라지는 것을 방지한다.
실제 FP·W8A8 로딩 코드와 비교 조건은 변경하지 않는다.

## 4. 결과와 오류 확인

기본 출력 구조는 다음과 같다. UTC timestamp와 `mktemp` 임의 접미사로 재실행/동시 실행 간 충돌을 피한다.

```text
data/w8a8-gpu/batch-<UTC timestamp>-<unique suffix>/
  run_01/
    command.txt       # 실제 Python 명령; 저장소 루트에서 해석
    console.log       # coordinator의 stdout/stderr 및 셸 실패 메시지
    exit_code.txt     # Python 종료 코드
    reports/
      summary.json   # 환경, 모델/INT8 검증, 지연시간, 메모리, 크기, 실패 단계
      summary.csv    # FP/W8A8 모델별 요약 행
      latency.csv    # first/warmup/measurement의 완료된 개별 호출
      benchmark.log  # 실제 worker의 로딩·커널·추론 진행과 오류 traceback
  run_02/            # 같은 구조
  run_03/            # 같은 구조
```

`reports/`는 기존 Python CLI의 `--output-dir`이며 새 폴더/빈 폴더만 허용하는 검사를 유지한다.
실험별 보고서를 합쳐 덮어쓰지 않는다. batch 폴더 전체를 전달하면 된다.
`data/`는 기존 `.gitignore` 대상이므로 결과가 자동으로 commit되지는 않는다.

- 성공 확인: 최상위 `status=passed`, `models.w8a8.status=passed` 및
  `checks.policy_infer`, `int8_weights`, `int8_kernels`, `all_target_layers_called`가 모두 `passed`여야 한다.
- 시간: `models.w8a8.latency_statistics`의 mean/p50/p95(ms). CUDA 동기화한 `policy.infer()` 전체 시간으로
  전처리·action CPU 반환·후처리를 포함한다. `measurement` 50회만 집계하며 로딩/최초 추론/warmup은 제외한다.
- 크기: `checkpoint_weights.bytes`는 **`model.safetensors` 파일 크기**다.
  assets·metadata·다운로드 캐시나 GPU packed weight 크기를 합한 값이 아니다.
- VRAM: `memory.loaded`는 로딩 후, `memory.loading`/`inference`는 구간별 PyTorch allocator peak다.
  BitBLAS/TVM 외부 할당은 PyTorch 값에 포함되지 않으므로 `memory.nvml`도 확인한다.
  NVML의 프로세스 peak는 50ms 간격 관측 최대값이며 순간 peak를 놓칠 수 있다.
  장치 전체 값은 다른 프로세스도 포함한다. `status`와 `errors`를 함께 보고 누락 값을 0으로 해석하지 않는다.
- FP 생략 시 `models.fp.status`와 `comparison.status`는 `not_measured`다.
  개선율은 JSON `null` / CSV 빈칸이며, W8A8 절대값만으로 속도·VRAM·크기 감소율을 주장하지 않는다.
  FP 비교도 양쪽 검증·입력·noise·정규화·환경 signature 등 기존 조건을 만족해야 계산한다.
- W8A8은 manifest에 지정된 Linear에 적용된다. 나머지 FP 연산은 기존 모델 설계다.
  오류 시 FP32 fallback을 실행하지 않는다. `profiler_verified=false`이므로 하드웨어 명령 profiler 검증을 뜻하지 않는다.

Python/로그 기록이 실패하면 셸은 첫 실패에서 중단하고 0이 아닌 코드를 반환한다.
실행 직후 `echo $?`로 확인할 수 있다. 기존 결과는 보존되며 이후 실험은 실행하지 않는다.
모델/BitBLAS/체크포인트 검증 실패는 기존 Python이 부분 JSON/CSV와 traceback을 남긴다.
잘못된 CLI 값, Python 부재, 디스크 오류, 강제 종료 등 보고서 생성 전 실패에서는 파일이 일부 없을 수 있다.
사전 설정 오류는 stderr를, 실행 시작 후 오류는 `console.log`와 `reports/benchmark.log`를 확인한다.
원인을 해결한 뒤 같은 명령으로 새 batch를 실행한다. 커널 캐시·GPU 온도·다른 프로세스는
새 프로세스로 초기화되지 않으므로 실험 당시 GPU 상태도 함께 기록한다.

## 5. 기존 W8A8 파일 6개 검토

| 기존 파일 | 구분 | 역할 / 유지 판단 |
| --- | --- | --- |
| `scripts/benchmark_pi05_fp_w8a8.py` | 실행 필수 | CLI, 새 CUDA worker, 측정 및 JSON/CSV/로그 생성 |
| `src/openpi/policies/w8a8_policy.py` | 실행 필수 | manifest·정규화·체크포인트 strict 검증과 policy 로딩 |
| `src/openpi/models_pytorch/bitblas_w8a8_layers.py` | 실행 필수 | INT8 가중치/활성화, BitBLAS INT32 연산, 정수 probe 및 진단 |
| `quantization/w8a8/test_benchmark.py` | 테스트 | CPU 회귀 검사; GPU 실행 시 직접 호출하지 않지만 유지 |
| `quantization/w8a8/NOTICE.md` | 출처 문서 | 외부 출처·변경·체크포인트·미검증 사항 기록; 유지 |
| `quantization/w8a8/LICENSE.openpi_duquant` | 외부 라이선스 | Apache-2.0 원문; 유지 필수 |

역할이 다른 파일이므로 삭제할 중복 구현은 없다. 6개 모두 그대로 유지한다.
기존 [측정 상세 문서](examples/libero/README.md)는 직접 Python CLI 사용법과 결과 해석의 참고 자료로 유지한다.
위 실행 필수 3개만 복사하면 되는 것은 아니다. 전체 OpenPI 소스·의존성·공식 Transformers 교체 파일과
기존 `LICENSE`, `LICENSE_GEMMA.txt`도 필요하다. 새 셸은 기존 CLI의 실행만 조정한다.

## 6. 로컬 검증 범위

Windows CPU 회귀 검사는 다음 명령으로 실행한다.

```powershell
python -B -m unittest discover -s quantization/w8a8 -p test_benchmark.py -v
```

CPU 테스트는 보고서·FP 생략·실패 보존·INT8 호출 계측·NVML 대체 객체 등을 검사한다.
이번 변경에서 기존 CPU 테스트 21개가 모두 통과했다. Git Bash 문법 검사와 대체 실행기로
기본 3회 반복, 설정 전달, 결과 격리/재실행 보존, 첫 실패 중단 및 종료 코드 보존도 확인했다.
실제 벤치마크의 체크포인트 누락 경로에서 부분 JSON/CSV/로그와 종료 코드 1을 확인했다.
로컬 CPU 검사는 Python 3.10.11 / PyTorch 2.12.0+cpu 환경이며 위 운영 기준 환경과는 다르다.
실제 모델 체크포인트 로딩, CUDA/BitBLAS 연산, GPU 수치 및 LIBERO 품질을 검증하는 테스트가 아니다.
GPU 담당자는 서버에서 생성한 결과와 실패 로그를 근거로 검증 상태를 갱신한다.

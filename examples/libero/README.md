# π0.5 LIBERO W8A8 경량화 확인

`handoff/libero-w8a8`에서 로컬 W8A8 체크포인트를 한 명령으로 측정한다.
개발 브랜치 `quantization/w8a8-setup` (`723135c7`)의
`scripts/benchmark_pi05_fp_w8a8.py`를 재사용했다. 실행 파일은 이 스크립트 하나다.
실제 INT8 가중치·BitBLAS 호출, 체크포인트 크기, GPU 메모리와 추론시간을 저장한다.
시뮬레이터·성공률·Replay 영상·학습·재양자화는 실행하지 않는다.

**Windows CPU 회귀 검사만 수행했다. 전체 모델 로딩과 CUDA/BitBLAS 실행 및 실제 수치는 미검증이다.**

## 서버 준비

- Linux NVIDIA GPU 서버와 기존 OpenPI Python 3.11+ 환경(아래 기본 경로 `.venv/bin/python`).
  프로젝트의 PyTorch 2.7.1, Transformers 4.53.2와
  [공식 Transformers 교체 파일 적용](../../README.md#pytorch-support), `safetensors`가 필요하다.
- BitBLAS/TVM은 `bitblas.ops.Matmul/MatmulConfig`, `backend="tir"`, `transform_weight`,
  INT8×INT8→INT32를 지원해야 한다. 호환 빌드/GPU 조합은 아직 검증하지 않았다.
  초기 커널 생성·packing·정수 probe는 실행하며 튜닝은 기본 off다. FP fallback은 없다.
- NVML Python 모듈 `pynvml`이 있으면 프로세스 전체 VRAM도 50 ms 간격으로 관측한다.
  없을 때만 `uv pip install --python .venv/bin/python nvidia-ml-py`로 설치한다.
  모듈/드라이버/권한상 측정 불가인 값은 `null`과 이유를 남긴다.
- PaliGemma tokenizer 캐시를 준비한다. 캐시가 없으면 기존 OpenPI transform이 최초 다운로드를 시도한다.
  체크포인트는 자동 다운로드하지 않는다. LIBERO 시뮬레이터·평가용 assets·EGL·ffmpeg는 필요 없다.
  단, 아래 checkpoint의 `assets/physical-intelligence/libero/norm_stats.json`은 필수다.

체크포인트 디렉터리는 다음 구조여야 한다. 기존 로더의 strict tensor 검증과 정규화 통계 검증을 그대로 사용한다.

```text
checkpoints/pi05-libero-w8a8-public/
  model.safetensors
  w8a8_config.json
  assets/physical-intelligence/libero/norm_stats.json
```

체크포인트가 아직 로컬에 없으면 저장소 루트에서 한 번 내려받는다. 아래 고정 revision의 파일 목록을 확인했다.
실제 모델 로딩과 GPU 호환성 검증은 별도로 필요하다. `README.md`도 받아 모델 카드와 라이선스 선언을 보존한다.

```bash
uvx --from huggingface_hub hf download fatdove/pi05-libero-w8a8 \
  model.safetensors w8a8_config.json \
  assets/physical-intelligence/libero/norm_stats.json README.md \
  --revision 2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee \
  --local-dir checkpoints/pi05-libero-w8a8-public
```

## 복사해서 실행할 명령

변경 파일이 포함된 저장소 루트에서 실행한다. Python 또는 체크포인트 위치가 다르면 해당 경로만 바꾼다.

```bash
.venv/bin/python -u scripts/benchmark_pi05_fp_w8a8.py --w8a8-checkpoint checkpoints/pi05-libero-w8a8-public --output-dir "data/w8a8-smoke-$(date +%Y%m%d-%H%M%S)"
```

기본값은 `cuda:0`, batch 1, denoising 10 steps, seed 42, 고정 합성 RGB 이미지 2장과 8차원 state,
최초 추론 1회 + 추가 warmup 10회 + 측정 50회다. 필요하면 같은 명령 뒤에
`--device cuda:1 --warmup 5 --iterations 100`처럼 붙인다.
`torch.compile`은 이 벤치마크에서 끄고 `JAX_PLATFORMS=cpu`를 worker import 전에 설정한다.
첫 커널 준비는 오래 걸릴 수 있다. 진행 상황은 출력 디렉터리의 `benchmark.log`에 기록한다.

FP가 없으면 `models.fp.status`와 `comparison.status`는 `not_measured`이며
속도·크기·메모리 개선율을 추정하지 않는다. JSON 숫자는 `null`, CSV 숫자 칸은 빈칸이다.
FP 경로를 지정했어도 `model.safetensors`가 없으면 같은 방식으로 건너뛴다.
이미 로컬 FP 체크포인트가 있을 때만 위 명령에 `--fp-checkpoint /path/to/pi05-libero-fp`를 붙인다.
FP/W8A8은 각각 새 프로세스에서 순차 측정하며 입력·noise·정규화·모델 설정·실행 환경이 일치하고
양쪽 검증이 성공한 경우에만 비교한다. FP를 새로 받거나 변환할 필요는 없다.

## JSON/CSV 결과

`--output-dir` 아래에 자동 저장한다. 기존 결과가 있는 디렉터리는 덮어쓰지 않는다.

| 파일 | 내용 |
| --- | --- |
| `summary.json` | 전체 결과, 환경, 레이어별 INT8 증거, 메모리, 평균/p50/p95, 실패 단계·이유 |
| `summary.csv` | 모델별 1행: 체크포인트 bytes, mean/p50/p95 ms, INT8 검증, VRAM, 선택적 FP 비교 |
| `latency.csv` | 모든 완료된 호출의 `model,phase,iteration,latency_ms`; `phase=measurement`만 집계 대상 |
| `benchmark.log` | 로딩/커널 준비/추론 진행과 오류 traceback |

`summary.json`의 `models.w8a8`에서 다음 값을 확인한다.

| 항목 | JSON 필드와 의미 |
| --- | --- |
| INT8 가중치 | `checks.int8_weights=passed`; `diagnostics_after.layers`의 실제 `qweight_dtype`, shape, bytes |
| BitBLAS INT8 사용 | `checks.int8_kernels`와 `checks.all_target_layers_called`가 `passed`; 모든 manifest 레이어의 정수 reference probe 통과, 실제 추론 `forward_calls` 증가, 호출 입력 `torch.int8`·출력 `torch.int32` 확인 |
| 체크포인트 크기 | `checkpoint_weights.bytes` / `gib`: `model.safetensors` 파일 전체. 통계/metadata/캐시 중복은 제외 |
| 로딩 직후 GPU 메모리 | `memory.loaded.allocated_bytes` / `reserved_bytes` |
| PyTorch 최대 VRAM | `memory.loading`과 `memory.inference`의 `peak_allocated_bytes` / `peak_reserved_bytes`; CSV는 두 구간의 최대값 |
| 프로세스 전체 VRAM | `memory.nvml.process_used_bytes` / `process_peak_used_bytes`; BitBLAS/TVM의 외부 CUDA 할당 포함, 로딩/추론 구간은 `phases`에 별도 저장 |
| 장치 전체 VRAM | `memory.nvml.device_used_bytes` / `device_peak_used_bytes`; 다른 프로세스도 포함 |
| 추론시간 | `latency_statistics.mean_ms`, `p50_ms`, `p95_ms`, `samples` |

시간은 CUDA를 호출 전후 동기화한 `policy.infer()` 전체 wall time이다.
전처리·GPU 연산·CPU로 action 반환·후처리를 포함하고, 로딩/커널 준비/최초 추론/warmup은 평균과 percentile에서 제외한다.
1회는 `pi05_libero` action chunk 전체(10 actions)를 만드는 호출이다. percentile은 선형 보간이다.
합성 입력의 고정 관측 성능 확인이며 LIBERO 성공률이나 품질을 나타내지 않는다.

W8A8은 manifest의 Linear 레이어만 의미하며 나머지 모델에는 FP 가중치/연산이 남는다.
INT8 확인은 실행 경로와 정수 probe 기반이며 하드웨어 명령 profiler 검증은 `profiler_verified=false`로 남긴다.
PyTorch peak는 allocator 내부 최대값이고 외부 할당을 포함하지 않는다.
NVML peak는 50 ms 간격 **관측 최대값**이므로 더 짧은 순간의 peak를 놓칠 수 있다.
NVML 값의 정의는 [NVIDIA NVML API](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)를 따른다.

정상 W8A8 단독 측정은 FP 미측정이어도 exit 0 / 최상위 `status=passed`다.
요청된 모델 검증 실패·CUDA 부재·중단은 exit 1 / `incomplete`로 부분 JSON/CSV와 로그를 남긴다.
NVML만 측정 불가여도 모델 검증은 계속하며 `memory.nvml.status`와 `errors`를 반드시 함께 확인한다.
출력 폴더 전체를 팀에 전달하면 된다. 실제 GPU 측정값은 서버에서 실행한 뒤에만 보고한다.

## 기존 Windows CPU 검사 기록

최소 GPU 전달본에서는 실행 중 호출하지 않는 CPU 회귀 테스트 파일을 제외했다.
[기존 테스트 원본](https://github.com/2chae1-1/openpi0.5/blob/b9965ddea250f1191b9411a2a5d12c6896e911de/quantization/w8a8/test_benchmark.py)은
보고서, FP 생략, 실패 보존, INT8 호출 계측, NVML 대체 객체를 검사한다.
기존 21개 통과 기록은 [GPU 인계 문서](../../GPU_HANDOFF.md#6-기존-로컬-검증-기록과-gpu-검증-범위)에 설명한다.
CPU 검사 기록은 전체 모델·CUDA·실제 BitBLAS 검증 결과가 아니다.

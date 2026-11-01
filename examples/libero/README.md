# 공개 W8A8 π0.5 LIBERO 일회 평가

`handoff/libero-w8a8`은 `main` (`215abfb217dbac7d5f1273282331b9b1866c0479`) 기반이다.
[fatdove/pi05-libero-w8a8](https://huggingface.co/fatdove/pi05-libero-w8a8)을 내려받아
기존 WebSocket 서버와 LIBERO 평가기로 4개 suite의 성공률을 측정한다. 재양자화·학습·FP 체크포인트는 필요 없다.
**Windows 정적/소형 CPU 검사만 수행했다. 전체 모델 로딩, CUDA·BitBLAS·LIBERO 실행과 성공률은 미검증이다.**

## 사전 환경

- Linux NVIDIA GPU 서버, driver, CUDA Toolkit(`nvcc`), C++ 빌드 도구, Bash, Git, `uv`.
- 모델 서버: 기존 Python 3.11 OpenPI 환경(기본 `.venv/bin/python`), 저장소의 전체 의존성,
  PyTorch **2.7.1**, Transformers **4.53.2**와 [공식 저장소 패치](../../README.md#setup), `safetensors`.
- BitBLAS/TVM: `bitblas.ops.Matmul/MatmulConfig`, `backend="tir"`, `transform_weight`,
  INT8×INT8→INT32를 지원하는 빌드. 이 포트와 호환되는 빌드/GPU 조합은 아직 검증하지 않았다.
  미설치 환경의 설치 출발점은 `uv pip install --python .venv/bin/python bitblas safetensors`와
  [BitBLAS 공식 안내](https://github.com/microsoft/BitBLAS#installing-with-pip)다.
  커널 생성·packing·정수 probe 실패 시 서버가 중단한다. 실패 로그를 전달한다.
- 평가기: **이미 정상 동작하는 LIBERO 환경**(기본 `examples/libero/.venv/bin/python`),
  [평가 의존성](requirements.in), `third_party/libero`의 assets/BDDL/init states, EGL/OpenGL 및 ffmpeg.
  저장소의 기존 구성은 서버 Python 3.11 / 평가 Python 3.8 분리다. 기존 환경을 재생성하거나 sync하지 않는다.
- Hugging Face 접근 및 최초 실행 때 PaliGemma tokenizer 다운로드를 위한 Google Cloud Storage 접근.
  공개 가중치는 약 5.27 GB이며, 모델 생성 중 CPU RAM과 GPU VRAM의 실제 최대 사용량은 미측정이다.

모든 명령은 전달받은 저장소 루트에서 실행한다. Python 경로가 다르면 아래 두 변수만 기존 환경으로 바꾼다.
기존 Docker 서버 이미지에는 BitBLAS가 없으므로 아래 호스트 실행을 사용한다.

## 1. 코드 전달 및 모델 다운로드

이 작업은 **commit/push하지 않은 로컬 변경**이므로 원격 `clone --branch handoff/libero-w8a8`로 받을 수 없다.
함께 전달한 `libero-w8a8.patch`를 기존 저장소의 깨끗한 작업 트리에 적용한다.
이미 변경 파일이 포함된 작업 디렉터리를 받았다면 첫 세 명령은 생략한다.

```bash
git switch -c handoff/libero-w8a8 215abfb217dbac7d5f1273282331b9b1866c0479
git apply --check /path/to/libero-w8a8.patch
git apply /path/to/libero-w8a8.patch
```

```bash
set -euo pipefail
git submodule update --init --recursive third_party/libero
export CKPT="$PWD/checkpoints/pi05-libero-w8a8-public"
uvx --from huggingface_hub hf download fatdove/pi05-libero-w8a8 \
  --revision 2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee \
  --local-dir "$CKPT" \
  --include model.safetensors w8a8_config.json \
    assets/physical-intelligence/libero/norm_stats.json
```

이 3개 파일의 상대 경로를 유지한다. 가중치는 Git에 넣지 않는다(`checkpoints/`는 이미 ignore됨).
revision은 개발 브랜치의 공개 체크포인트 조사에서 가져온 값이다.
파일별 다운로드 옵션은 [Hugging Face 공식 문서](https://huggingface.co/docs/huggingface_hub/en/guides/download)를 따른다.

## 2. 모델 서버 — 터미널 1

```bash
set -euo pipefail
export SERVER_PYTHON="$PWD/.venv/bin/python"
export CKPT="$PWD/checkpoints/pi05-libero-w8a8-public"
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export RUN="$PWD/data/libero/w8a8-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
printf '%s\n' "$RUN" > data/libero/latest-w8a8-run.txt
git rev-parse HEAD > "$RUN/repository-base.txt"
sha256sum scripts/serve_policy.py src/openpi/policies/w8a8_policy.py \
  src/openpi/models_pytorch/bitblas_w8a8_layers.py examples/libero/README.md \
  > "$RUN/source-sha256.txt"
printf '%s\n' 'fatdove/pi05-libero-w8a8@2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee' > "$RUN/checkpoint.txt"
uv pip freeze --python "$SERVER_PYTHON" > "$RUN/server-packages.txt"
nvidia-smi > "$RUN/nvidia-smi.txt"
"$SERVER_PYTHON" -u scripts/serve_policy.py --env LIBERO --port 8000 \
  policy:w8a8 --policy.dir "$CKPT" --policy.device cuda:0 \
  --policy.disable-compile-for-validation 2>&1 | tee "$RUN/server.log"
```

8000 포트에서 서버가 대기할 때 다음 단계로 진행한다. 최초 tokenizer 다운로드와 BitBLAS 커널 생성이 오래 걸릴 수 있다.
위 옵션은 `torch.compile`만 끄며 BitBLAS 커널 생성과 각 레이어의 INT8 정수 probe는 수행한다.
체크포인트는 엄격하게 로딩하며 FP matmul로의 fallback은 없다.

## 3. LIBERO 평가 — 터미널 2, 같은 저장소 루트

```bash
set -euo pipefail
export LIBERO_PYTHON="$PWD/examples/libero/.venv/bin/python"
export RUN="$(cat data/libero/latest-w8a8-run.txt)"
export PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
uv pip freeze --python "$LIBERO_PYTHON" > "$RUN/libero-packages.txt"
for suite in libero_spatial libero_object libero_goal libero_10; do
  "$LIBERO_PYTHON" -u examples/libero/main.py \
    --args.host 127.0.0.1 --args.port 8000 \
    --args.task-suite-name "$suite" --args.num-trials-per-task 50 \
    --args.seed 7 --args.replan-steps 5 \
    --args.video-out-path "$RUN/videos/$suite" \
    2>&1 | tee "$RUN/$suite.log"
  if grep -q 'Caught exception:' "$RUN/$suite.log"; then
    echo "평가 중 예외 발생: $suite — 로그를 전달하세요." >&2
    exit 1
  fi
  grep -q 'Total episodes: 500$' "$RUN/$suite.log"
  grep -E 'Total success rate:|Total episodes:' "$RUN/$suite.log" \
    | sed "s/^/$suite /" | tee -a "$RUN/summary.txt"
done
```

각 suite는 10 tasks × 50회, 총 2,000 episodes다. 네 suite 모두 정상 완료했을 때만 전체 평균을 보고한다.
평가기 내부에서 잡은 예외도 로그로 검사한다. 도중 실패하면 남은 suite를 진행하지 말고 현재 로그를 전달한다.
LIBERO 최초 경로 질문에는 이 저장소의 `third_party/libero` 경로를 사용한다.
EGL 대신 기존 환경에 X 디스플레이가 준비되어 있으면 `MUJOCO_GL=glx`로 바꾼다.

## 4. 결과 전달

평가가 끝나면 터미널 1의 서버를 `Ctrl-C`로 종료한다.
`data/libero/latest-w8a8-run.txt`에 기록된 디렉터리 전체를 전달한다:
`summary.txt`(suite별 최종 성공률 0~1/episode 수), `server.log`, suite별 로그,
Python 패키지 목록, GPU 정보, 코드 SHA256, 저장소 base commit, HF revision, `videos/`.
실패한 경우 같은 디렉터리의 실패 로그를 전달한다. 기존 평가기의 영상 이름은 task 성공/실패별로 덮어써서 일부 replay만 남는다.

## 기존 FP 실행

같은 모델 서버 환경에서 기존 기본 FP 및 사용자 FP 체크포인트 선택도 유지된다.

```bash
"$SERVER_PYTHON" scripts/serve_policy.py --env LIBERO
"$SERVER_PYTHON" scripts/serve_policy.py --env LIBERO \
  policy:checkpoint --policy.config pi05_libero --policy.dir /path/to/fp-checkpoint
```

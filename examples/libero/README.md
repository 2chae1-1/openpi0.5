# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## 공개 W8A8 모델로 LIBERO 평가

**브랜치 다운로드 → W8A8 체크포인트 다운로드 → 환경 설정 → 모델 서버 실행 → LIBERO 평가 → 결과 공유** 순서다.
사용 브랜치는 `quantization/w8a8-setup`이다.
아래 명령은 `uv`, Git/Git LFS, NVIDIA driver/CUDA Toolkit(`nvcc`), EGL/OpenGL 및 빌드 도구가 있는 Linux GPU 서버의 Bash 기준이다.
`scripts/serve_policy.py`가 기존 `load_w8a8_policy()`로 모델을 읽고, 기존 WebSocket 서버에 연결한다.
`examples/libero/main.py`의 관측 전처리·action 실행·성공률 계산은 그대로 사용한다.
FP 모델이나 학습 데이터 다운로드, 재양자화, 학습, export는 필요 없다. PaliGemma tokenizer는 첫 실행 때 별도 캐시 다운로드가 필요하다.

**실제 CUDA·LIBERO 검증은 실행하지 않았다.** 이 연결은 Windows의 정적·CPU 검사 범위에서만 확인했다.
Python 3.11 문법 AST·메모리 내 컴파일, 서버 Ruff 검사, 기존 CPU 회귀 21개,
의존성을 대체한 서버 연결 검사 10개 및 실제 Tyro 0.9.22 CLI 파싱 검사 11개를 통과했다.
CLI 검사는 무거운 import와 평가 함수 본문을 제외했으며 모델 로딩·네트워크·시뮬레이터 실행 검증이 아니다.
BitBLAS/TVM의 호환 버전·빌드는 아직 미확정이며, 아래 설치 명령만으로 GPU 실행이 검증되지는 않는다.
기존 Docker 서버 이미지에는 BitBLAS가 없으므로 W8A8은 아래 호스트 서버 경로를 사용한다.

### 1. 브랜치와 체크포인트 다운로드

```bash
set -euo pipefail
git clone --branch quantization/w8a8-setup --single-branch \
  https://github.com/2chae1-1/openpi0.5.git
cd openpi0.5
git submodule update --init --recursive third_party/libero
export CKPT="$PWD/checkpoints/pi05-libero-w8a8-public"
uvx --from huggingface_hub hf download fatdove/pi05-libero-w8a8 \
  --revision 2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee \
  --local-dir "$CKPT" \
  --include model.safetensors w8a8_config.json 'assets/**'
```

필수 파일은 `model.safetensors`, `w8a8_config.json`,
`assets/physical-intelligence/libero/norm_stats.json`이다. 폴더 구조를 유지한다.
공개 모델의 형식과 revision 근거는 [PUBLIC_CHECKPOINT.md](../../quantization/w8a8/PUBLIC_CHECKPOINT.md)에 있다.
다운로드 옵션은 [Hugging Face 문서](https://huggingface.co/docs/huggingface_hub/en/guides/download)를 따른다.

### 2. 환경 설정 (최초 한 번)

서버는 Python 3.11, 시뮬레이터는 기존 Python 3.8 환경을 각각 사용한다. 모든 명령은 저장소 루트에서 실행한다.

```bash
set -euo pipefail
# 모델 서버 환경: 저장소에 고정된 PyTorch 2.7.1 / Transformers 4.53.2 사용
uv venv --python 3.11 .venv
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --frozen
uv pip install --python .venv/bin/python bitblas safetensors \
  'torch==2.7.1' 'transformers==4.53.2'
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/

# LIBERO 환경: 기존 평가 의존성을 별도 설치
uv venv --python 3.8 examples/libero/.venv
uv pip sync --python examples/libero/.venv/bin/python \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python \
  -e packages/openpi-client -e third_party/libero
```

BitBLAS 설치의 출발점은 [공식 설치 안내](https://github.com/microsoft/BitBLAS/blob/main/README.md#installing-with-pip)다.
로더에는 `bitblas.ops.Matmul/MatmulConfig`, `transform_weight`, `backend="tir"` 및 INT8×INT8→INT32가 필요하다.
지원되는 CUDA/TVM 빌드를 준비해야 하며 커널 생성·packing·INT8 probe가 실패하면 서버는 중단한다.
추가 설치 후 환경을 유지하도록 실행 명령에는 `uv run --no-sync`를 사용한다.

### 3. 모델 서버 실행 (터미널 1)

```bash
set -euo pipefail
export JAX_PLATFORMS=cpu
export CKPT="$PWD/checkpoints/pi05-libero-w8a8-public"
export RUN="$PWD/data/libero/w8a8-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
printf '%s\n' "$RUN" > data/libero/latest-w8a8-run.txt
git rev-parse HEAD > "$RUN/repository.txt"
uv pip freeze --python .venv/bin/python > "$RUN/server-packages.txt"
nvidia-smi > "$RUN/nvidia-smi.txt"
uv run --no-sync scripts/serve_policy.py --env LIBERO --port 8000 \
  policy:w8a8 --policy.dir "$CKPT" --policy.device cuda:0 \
  --policy.disable-compile-for-validation 2>&1 | tee "$RUN/server.log"
```

서버가 8000 포트에서 대기할 때까지 기다린다. 최초 실행은 tokenizer 다운로드와 BitBLAS 커널 컴파일·probe로 오래 걸릴 수 있다.
위 명령은 기존 로더의 검증용 옵션으로 `torch.compile`을 끈다. BitBLAS 커널 생성은 수행하며, compile 활성 조합은 미검증이다.
`--policy.dir`에는 다운로드한 로컬 폴더를 지정한다. 기본 FP 서버는 기존처럼 `uv run scripts/serve_policy.py --env LIBERO`로 실행한다.
커스텀 FP 체크포인트의 `policy:checkpoint --policy.config ... --policy.dir ...`도 유지된다.

### 4. LIBERO 평가 (터미널 2, 같은 저장소 루트)

```bash
set -euo pipefail
export RUN="$(cat data/libero/latest-w8a8-run.txt)"
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
uv pip freeze --python examples/libero/.venv/bin/python > "$RUN/libero-packages.txt"
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8000 \
  --args.task-suite-name libero_spatial --args.num-trials-per-task 50 \
  --args.seed 7 --args.replan-steps 5 \
  --args.video-out-path "$RUN/videos/libero_spatial" \
  2>&1 | tee "$RUN/libero_spatial.log"
```

LIBERO가 처음 실행되며 경로 설정을 묻는 경우 기본 로컬 경로를 사용한다.
EGL 문제가 있으면 기존 평가와 같이 `MUJOCO_GL=glx`를 사용하고 X 디스플레이를 준비한다.
서버가 다른 머신이면 `--args.host`를 서버 주소로 바꾼다. 클라이언트 옵션은 `tyro.cli(eval_libero)` 구조에 따라 `--args.*`다.
전체 4개 suite 비교는 위 평가 명령의 `libero_spatial`을 `libero_object`, `libero_goal`, `libero_10`으로 각각 바꾸고
로그 파일과 영상 폴더도 suite 이름별로 바꿔 실행한다. 각 suite는 task당 50회, seed 7, replan 5를 유지한다.

### 5. 결과 공유

`$RUN`의 서버·평가 로그와 환경 정보, Git commit, HF revision
`2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee`, 실제 실행 명령을 함께 공유한다.
각 suite의 마지막 `Total success rate`(0~1), `Total episodes`, 성공 수를 보고하고, 4개 suite 평균은 같은 조건으로 모두 완료했을 때만 계산한다.
`Caught exception`이나 서버 오류가 있으면 실패 로그도 공유하고 정상 완료된 성공률로 해석하지 않는다.
기존 평가기는 task·성공/실패별 같은 영상 이름을 덮어쓰므로 영상은 일부 replay만 남으며 모든 episode 기록이 아니다.
아래 Results 표는 기존 FP 체크포인트의 참고값이며 이번 W8A8 측정값이 아니다.

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

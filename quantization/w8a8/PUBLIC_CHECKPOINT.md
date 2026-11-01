# 공개 π0.5 LIBERO W8A8 재사용 인계 (2026-11-01)

작업 저장소: `C:\Users\laure\projects\openpi0.5`, 브랜치 `quantization/w8a8-setup`.
공개 checkpoint를 우선 사용한다. 기존 252개 후보 JSON, CPU quantizer/exporter는 보존하며
공개 W8A8 checkpoint를 exporter에 넣거나 다시 양자화하지 않는다.

## 확인한 출처와 범위

- [모델 문서](https://huggingface.co/fatdove/pi05-libero-w8a8/blob/2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee/README.md),
  [manifest](https://huggingface.co/fatdove/pi05-libero-w8a8/blob/2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee/w8a8_config.json),
  [정규화 통계](https://huggingface.co/fatdove/pi05-libero-w8a8/blob/2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee/assets/physical-intelligence/libero/norm_stats.json).
- Hub API에서 확인한 revision: `2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee`.
- [openpi_duquant 원본](https://github.com/JingxuanZhang77/openpi_duquant/blob/7d4d1582297ced3131a7b3f9403f7ce403fd59f9/src/openpi/models_pytorch/bitblas_w8a8_layers.py)
  의 `save_w8a8_to_hf()`와 `load_w8a8_policy()`를 확인했다.
  원본 파일 SHA256: `81d02a7ac477a2fbbb7a8ef933b8acb992a1c8f5c23ad940da2614c538593788`.
  원본 exporter는 packing 이전 INT8를 저장하고 scale을 FP16으로 저장한다.
  원본 loader는 비양자화 state 로딩 → 명시한 Linear 교체 → GPU packing → policy 생성 순서다.
- Windows에서는 작은 JSON/모델 문서와 safetensors **헤더만** 읽었다.
  HTTP 206 및 Content-Range를 검사하고 `0-7`, `8-163895` 범위만 읽었다.
  전체 파일 크기는 5,266,572,696 bytes이며 tensor payload는 다운로드하지 않았다.
- `reference/`는 해당 revision의 작은 원문과 헤더 snapshot이다. 헤더는 가중치 파일이
  아니므로 `model.safetensors`로 이름을 바꾸거나 로더에 전달하지 않는다.
  헤더의 shape/dtype은 실제 값의 유한성·양수성·정확도 증거가 아니다.

| 원문 | SHA256 |
| --- | --- |
| w8a8_config.json | `edc030d56afbb066b88d108c681bf1d46b62816e515e2084b100390773bcce78` |
| norm_stats.json | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| MODEL_CARD.md | `5ebf7a39d68d191ab60e6fe75628f515160b65330c948f84b416e858a18498de` |
| model.safetensors.header.json | `3806609773244cb99ec8bda07d84c056ce96d1381f3cad82e84eb0bdedbb1417` |

## 180개와 252개의 차이 및 호환성 판단

| 대상 | 공개 checkpoint | 기존 exporter 후보 |
| --- | ---: | ---: |
| PaliGemma language model attention 18 × 4 | 72 | 72 |
| PaliGemma language model MLP 18 × 3 | 54 | 54 |
| Gemma Expert MLP 18 × 3 | 54 | 54 |
| Gemma Expert attention 18 × 4 | BF16 유지 | 72 |
| 양자화 합계 | 180 | 252 |

공개 목록은 후보의 부분집합이며 정확히 Expert `self_attn.{q,k,v,o}_proj` 72개가 빠진다.
헤더에서도 이 72개는 `.weight` BF16으로 존재한다. 이것은 누락이나 모델 depth 차이가
아니라 양자화 선택 차이다. 선택 이유나 252개보다 정확도가 좋은지는 이 자료만으로 판단하지 않는다.

| 계약 | 공개 파일에서 확인한 내용 | 현재 로더 처리 |
| --- | --- | --- |
| manifest | `quantization=w8a8_bitblas`, `complete_model=true`, 명시한 180개, `model_type=PI0Pytorch` | 이미 지원. model_type을 로컬 exporter의 `pi05_libero`와 비교하지 않음 |
| tensor key | 995개. `.qweight`와 `.weight_scale` 각각 180개, manifest와 정확히 일치 | 명시한 180개만 교체, 나머지 가중치는 strict load |
| weight | I8, 2차원 `[out,in]`; 다른 INT8 key 없음 | raw weight를 유지하고 GPU에서 별도 packing |
| scale | F16 `[out]` 180개 | 기존 FP32 buffer로 로딩; destination dtype에서 유한·양수 검사 |
| 비양자화 dtype | BF16 512개, F32 122개, I64 1개 | 공식 선택적 BF16 변환과 기존 dtype/shape 검사 재사용 |
| 공유 가중치 | PaliGemma embedding 존재, PaliGemma lm_head 없음. Expert lm_head 존재 | 실제 모델이 tied인 경우에만 기존 helper가 PaliGemma head 복원; Expert head 보존 |
| buffer | rotary inv_freq 2개, vision position_ids 1개 존재 | 공식 모델 state에 없는 경우에만 기존 helper가 이 세 key 제거 |
| norm_stats | `assets/physical-intelligence/libero/norm_stats.json`, 최상위 `norm_stats`; state 8, actions 7, mean/std/q01/q99 | 공식 normalize.load 형식과 일치. JSON 값의 길이·유한성·std≥0·q99≥q01 확인 |

**확인한 직렬화 계약은 현재 로더와 일치한다.** manifest 변환, key rename, scale 재양자화,
252개 강제 교체는 필요 없다. 전체 실제 모델과의 키/shape 대조 및 tensor 값 검사는
아래 서버 단계에서 확정한다. 헤더만으로 GPU 실행 호환성을 확정하지 않는다.

기존 BitBLAS W8A8 연산, 명시적 계층 교체, raw/packed 분리, FP32 scale, 정수 reference
probe, strict state validation, alias/buffer helper 및 공식 Policy/transform을 재사용한다.
이번 runtime 수정은 단계별 진단 로그뿐이다. 공식 모델·설정·추론·의존성 파일은 수정하지 않았다.
`scripts/validate_w8a8_forward.py`는 실제 저장소에 존재하며 기존 검증 경로를 유지한다.

## GPU 서버 실행 순서 (아래 전체 미실행)

Linux 서버에서 해당 브랜치와 수정 파일을 준비한다. 서버의 기존 OpenPI 환경을 활성화하고
Python ≥3.11, 프로젝트 PyTorch 2.7.1 / Transformers 4.53.2, 공식 transformers_replace,
safetensors, huggingface_hub, CUDA, BitBLAS/TVM을 준비한다. BitBLAS 호환 빌드는 미확정이다.
모델 문서의 AutoModel 예시 대신 아래 전용 로더를 사용한다. 원본 loader와 반환 형식도
다르므로 이 저장소에서는 `(policy, model)`을 받는다.

아래 Bash 명령은 저장소 루트에서 순서대로 실행한다. 실패 시 즉시 중단한다.
로그와 report는 새 경로를 사용하고, 충분한 CPU RAM/VRAM을 준비한다(peak 미측정).
공식 transform factory가 사용할 PaliGemma tokenizer 캐시도 필요하다.

```bash
set -euo pipefail
test "$(git branch --show-current)" = quantization/w8a8-setup
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export CKPT=/srv/checkpoints/pi05-libero-w8a8-public
export LOG=/srv/checkpoints/w8a8-public-run-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
git rev-parse HEAD > "$LOG/repository.txt"
git diff > "$LOG/working-tree.patch"
git status --short > "$LOG/status.txt"
python -m pip freeze > "$LOG/packages.txt"
nvidia-smi > "$LOG/nvidia-smi.txt"
```

1. **다운로드**: GPU 서버에서만 전체 checkpoint를 받는다. 지정 revision과 파일을 유지한다.

```bash
python - <<'PY' > "$LOG/download.log" 2>&1
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="fatdove/pi05-libero-w8a8",
    revision="2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee",
    local_dir=os.environ["CKPT"],
    allow_patterns=["model.safetensors", "w8a8_config.json", "README.md", "assets/**"],
)
PY
python scripts/inspect_w8a8.py "$CKPT" > "$LOG/headers.json" 2> "$LOG/headers.log"
```

통과: 다운로드 exit 0, headers JSON의 `layer_count=180`, 모든 qweight I8/scale F16 형상 일치.
이 단계는 full model load/scale 값/GPU 검증이 아니다. 실패 시 download.log의 Hub 오류,
디스크 공간, headers.log의 manifest/key/shape 오류와 revision을 확인한다.

2. **전체 CPU 로딩 검사**: 서버에서 실제 모델을 만들고 기존 helper로 검사 및 strict load한다.
   이 단계도 전체 가중치 메모리를 사용하며 Windows에서는 실행하지 않는다.

```bash
python - <<'PY' > "$LOG/load.log" 2>&1
import dataclasses
import os
from pathlib import Path
from safetensors.torch import load_file
from openpi.training import config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.bitblas_w8a8_layers import replace_w8a8_layers
from openpi.policies.w8a8_policy import read_manifest, load_norm_stats, prepare_checkpoint_state, validate_state_dict
p = Path(os.environ["CKPT"])
names = read_manifest(p)
assert len(names) == 180
c = config.get_config("pi05_libero")
data = c.data.create(c.assets_dirs, c.model)
load_norm_stats(p, data)
model = PI0Pytorch(dataclasses.replace(c.model, pytorch_compile_mode=None))
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
replace_w8a8_layers(model, names)
state = load_file(str(p / "model.safetensors"), device="cpu")
prepare_checkpoint_state(model, state)
validate_state_dict(model, state, names)
model.load_state_dict(state, strict=True)
print("PASS: full strict CPU load, normalization, 180 targets and positive finite scales; no kernels run")
PY
```

통과: exit 0과 마지막 PASS. 실패 시 load.log의 missing/unexpected key, shape, scale,
norm_stats, transformers_replace 오류를 확인한다. key를 무시하거나 빈 통계로 우회하지 않는다.
tokenizer 다운로드 오류는 체크포인트 형식 오류와 구분한다.

3. **INT8 커널 검사**: 전용 로더가 다시 strict load한 뒤 180개 GPU packing/probe를 수행한다.

```bash
python scripts/inspect_w8a8.py "$CKPT" --initialize --device cuda:0 \
  --disable-compile-for-validation \
  > "$LOG/kernels.json" 2> "$LOG/kernels.log"
```

통과: exit 0, `layer_count=180`, `all_int8_probes_passed=true`, 각 계층의
`operator_created`, `weight_packed`, `int8_probe_passed=true`, `pytorch_compile_mode=null`.
kernels.log의 `strict checkpoint load passed`로 로딩과 커널 단계를 구분한다.
실패 시 마지막 `INT8 probe passed [i/180]` 다음의 계층 이름과 연결된 원인 traceback을
확인한다. BitBLAS/TVM import/API, CUDA target, compiler, packing, 정수 reference 불일치를
구분한다. 기본 probe M=1이며 모든 실제 token 길이를 검증한 결과는 아니다.
`--opt-m 1 16 32 64`가 기본이고 `--enable-tuning`은 기본 off다. FP fallback은 없다.

4. **실제 policy.infer 경로 검사**: checkpoint의 공식 전처리·정규화·토큰화·샘플링·출력 변환을 실행한다.

```bash
python scripts/validate_w8a8_forward.py "$CKPT" --device cuda:0 \
  --report "$LOG/synthetic-smoke.json" > "$LOG/synthetic-smoke.log" 2>&1

# 실제 LIBERO 관측 NPZ를 준비한 경우에만 실행
python scripts/validate_w8a8_forward.py "$CKPT" --device cuda:0 \
  --observation /srv/observations/libero_obs.npz --prompt "pick up the book" \
  --report "$LOG/recorded-smoke.json" > "$LOG/recorded-smoke.log" 2>&1
```

NPZ는 `allow_pickle=False`로 읽으며 `observation/state` 유한한 `(8,)`,
`observation/image`, `observation/wrist_image` HWC RGB uint8을 포함해야 한다.
report 파일은 기존 파일이면 거부한다. report의 `input_kind`로 합성/기록 관측을 구분한다.
통과: exit 0, `strict_model_load_and_kernel_probes`, `policy_infer`,
`all_target_layers_called`가 모두 `passed`; diagnostics의 layer_count=180 및 모든 probe 통과;
actions_shape=`[10,7]`, 모든 action 유한, 각 계층 forward_calls 증가.
실패 시 JSON `error`, 단계 상태, log의 전체 traceback을 보존한다. 실패 이후는 `not_run`이다.
실제 flattened M 오류는 probe 통과와 별도로 해결해야 한다.

## 로컬 검증과 남은 검증

가능한 CPU 회귀 명령:

```bash
python -B -m unittest discover -s quantization/w8a8 -p test_validation.py -v
python -B -m unittest discover -s quantization/w8a8 -p test_public_checkpoint.py -v
# safetensors 설치된 서버에서 기존 파일 저장/재로딩 검사까지
python -B -m unittest discover -s quantization/w8a8 -p 'test*.py' -v
```

로컬 Python 3.10.11 / torch 2.12.0+cpu / NumPy 2.0.2는 서버 지원 환경과 다르다.
기존 test_validation 9개와 public metadata/소형 CPU 로딩 회귀 4개, 합계 13개가 통과했다.
관련 Python 9개 파일의 AST/메모리 내 컴파일, JSON 파싱 및 `git diff --check`가 통과했다.
Ruff가 설치되어 있지 않아 Ruff lint/format 검사는 실행하지 않았다.
전체 suite는 로컬 safetensors 부재로 test_export_cpu import가 실패했다.
새 public 테스트의 header reader는 실제 저장한 헤더를 사용하는 대체 reader이며 실제
safetensors payload를 읽지 않는다. 소형 모델은 실제 module 경로와 180/252 선택을
보존하지만 차원은 2×2이므로 실제 PI0 모델 로딩을 검증한 결과가 아니다.

**미검증:** 실제 전체 checkpoint 로딩·값 검사, 실제 GPU BitBLAS/TVM 빌드·packing·INT8
커널 및 실제 입력 M, 실제 policy.infer, 실제 LIBERO 관측/episode/성공률,
FP 원본 수치·정확도 비교, profiler INT8 증거, latency/peak RAM·VRAM, torch.compile,
Jetson Orin ARM64/SM87 호환성과 성능. 공개 모델 문서의 성능 수치는 우리 검증 결과가 아니다.
커밋·Push 및 원본 모델 코드 변경은 하지 않았다.

이번 수정 파일은 `w8a8_policy.py`, `inspect_w8a8.py`, `validate_w8a8_forward.py`의
진단 로그와 README/CPU_HANDOFF/NOTICE 문서다. 새 파일은 이 문서,
`test_public_checkpoint.py`, `reference/`의 네 metadata snapshot이다.
작업 시작 전부터 존재한 `export_cpu.py`, `test_validation.py`, alias helper 및
forward 검증 스크립트의 미커밋 작업은 보존했다. 기존 BitBLAS 계층 구현과
`targets_pi05_libero_candidate.json`은 이번에 수정하지 않았다.

## FP 대 공개 W8A8 추론 벤치마크 (GPU 서버 전용, 실제 측정 미실행)

`scripts/benchmark_pi05_fp_w8a8.py`는 위의 헤더 → strict load → INT8 커널 →
`policy.infer()` 검사가 끝난 뒤 실행한다. 준비된 모델을 비교하며 재양자화·재학습·Pruning을
수행하지 않는다. 아래의 다운로드·공식 변환·모델 로딩·벤치마크는 **Linux GPU 서버에서만**
수행한다. Windows에서 전체 가중치를 내려받거나 CUDA/BitBLAS를 실행한 결과가 아니다.

### 공식 FP 로딩 경로와 형식 조건

공식 FP 경로는 `config.get_config("pi05_libero")`와
`policy_config.create_trained_policy(config, checkpoint_dir)`다. 공식 설정은 π0.5,
action horizon 10, padded action dimension 32이며 LIBERO 정책 출력은 `(10, 7)`이다.
`scripts/serve_policy.py`의 LIBERO 기본 checkpoint는
`gs://openpi-assets/checkpoints/pi05_libero`다.

- [공식 LIBERO checkpoint 안내](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
  와 [공식 PyTorch 안내 및 변환 절차](https://github.com/Physical-Intelligence/openpi#pytorch-support)를 확인했다.
- [공식 policy_config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/policy_config.py)는
  루트에 `model.safetensors`가 있으면 PyTorch 로더를, 없으면 `params/`의 JAX 로더를 선택한다.
  `pytorch_weight_path` 설정의 예시 문자열은 배포된 PyTorch 파일의 존재 증거가 아니다.
- [공식 변환기](https://github.com/Physical-Intelligence/openpi/blob/main/examples/convert_jax_model_to_pytorch.py)는
  이 저장소에도 `examples/convert_jax_model_to_pytorch.py`로 존재하며 JAX/Orbax를
  `model.safetensors`로 저장한다. 새 변환기나 비공식 key 변환은 추가하지 않는다.
- 이번 환경에서는 GCS 객체 목록에 접근하지 못했으므로 공식 원격 checkpoint에 현재
  PyTorch 파일이 있는지는 **미확인**이다. 다운로드한 로컬 디렉터리의 실제 파일로 판정한다.
  `params/`만 존재하는 FP를 PyTorch W8A8과 직접 속도 비교하지 않는다.

벤치마크는 준비된 **로컬 디렉터리만** 받으며 자동 다운로드/변환을 하지 않는다.
FP의 `model.safetensors`가 없거나 `--fp-checkpoint`를 생략하면 FP는 `not_run`으로 남기고
W8A8 측정은 계속한다. 전체 비교는 미검증이며 speedup은 `null`, 종료 코드는 0이 아니다.
준비되지 않은 FP를 JAX 경로로 대신 실행하지 않는다.

### 서버 환경과 FP 준비

앞 절의 OpenPI/Python 3.11 환경, PyTorch 2.7.1, Transformers 4.53.2 및 이 저장소의
`transformers_replace`, safetensors, huggingface_hub, CUDA, BitBLAS/TVM을 사용한다.
새 대형 의존성은 필요 없다. GPU의 BF16 및 BitBLAS INT8 지원, CUDA/toolkit/driver와
BitBLAS 빌드 조합을 먼저 확인한다. 공개 모델 문서는 A40에서의 실행을 설명하지만
이 저장소와 Jetson Orin ARM64/SM87에서의 실행 보증은 아니다. Orin 호환성은 후속 검증 사항이다.

앞 절의 `CKPT`, `PYTHONPATH`, `JAX_PLATFORMS=cpu` 설정과 W8A8 다운로드/검사를 완료한
같은 셸에서 다음을 실행한다. FP를 이미 공식 경로로 준비했다면 다운로드·변환 단계는
건너뛰고 `FP`를 그 디렉터리로 설정한다.

```bash
# 저장소 루트, 활성화된 GPU 서버 Python 환경에서 실행
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export FP=/srv/checkpoints/pi05_libero_pytorch
export CKPT=/srv/checkpoints/pi05-libero-w8a8-public

# 서버에서만 공식 checkpoint 다운로드. 전체 가중치와 assets가 포함된다.
export FP_JAX="$(python - <<'PY'
from openpi.shared import download
print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))
PY
)"

if test -f "$FP_JAX/model.safetensors"; then
  # 이미 공식 PyTorch 파일이 있다면 해당 디렉터리를 그대로 사용
  export FP="$FP_JAX"
elif test -d "$FP_JAX/params"; then
  # 공식 변환기만 사용. 입력 경로에 pi05 이름을 유지한다.
  test ! -e "$FP"
  python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir "$FP_JAX" --config-name pi05_libero \
    --output-path "$FP" --precision bfloat16

  # 현재 변환기의 assets 탐색은 checkpoint의 부모를 보므로 직접 확인/복사한다.
  # 가중치 변환이 아니라 원본 공식 정규화 자산의 보존이다.
  test -f "$FP_JAX/assets/physical-intelligence/libero/norm_stats.json"
  mkdir -p "$FP/assets"
  cp -a "$FP_JAX/assets/." "$FP/assets/"
else
  echo "공식 FP checkpoint의 지원 형식을 확인할 수 없습니다" >&2
  exit 1
fi
test -s "$FP/model.safetensors"
test -f "$FP/assets/physical-intelligence/libero/norm_stats.json"
```

현재 공식 변환기에는 경로의 `pi05` 문자열을 검사하는 코드가 있어 입력 디렉터리 이름을
임의로 바꾸지 않는다. 변환 자체의 수치 보존은 이번 Windows 작업에서 검증하지 않았다.
변환 로그, 원본 checkpoint의 출처/객체 버전, 변환 시 Git SHA와 precision을 팀에서 보존한다.
`--fp-revision`/`--w8a8-revision`은 사용자가 선언하는 provenance이며 로컬 가중치의 revision을
자동 인증하는 기능이 아니다. 알 수 없는 revision을 임의로 채우지 않는다.

### 복사해서 실행할 벤치마크 명령

다른 GPU 작업을 중단하고, 앞 절의 `inspect_w8a8.py` 및 `validate_w8a8_forward.py`가
통과한 뒤 실행한다. 출력 디렉터리는 새 경로 또는 비어 있는 경로여야 한다.

```bash
# 저장소 루트. FP/CKPT는 앞 절에서 준비한 로컬 디렉터리다.
export BENCH_OUT="$PWD/results/pi05-fp-w8a8-$(date +%Y%m%d-%H%M%S)"
python scripts/benchmark_pi05_fp_w8a8.py \
  --fp-checkpoint "$FP" --w8a8-checkpoint "$CKPT" \
  --output-dir "$BENCH_OUT" --device cuda:0 \
  --w8a8-revision 2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee \
  --prompt "pick up the book" --num-steps 10 --seed 42 \
  --warmup 10 --iterations 50 --opt-m 1 16 32 64
```

이 명령은 synthetic smoke input을 사용한다. 실측 latency는 기록하지만 실제 LIBERO 작업의
성공률이나 episode 처리 성능을 나타내지 않는다. 기록한 LIBERO 관측을 사용하려면 위 명령에
`--observation /srv/observations/libero_obs.npz`를 추가한다. NPZ는 기존 검증 스크립트와 같은
`observation/state` `(8,)` 유한 벡터, `observation/image`, `observation/wrist_image` HWC RGB
uint8을 포함하며 `allow_pickle=False`로 읽는다. prompt는 `--prompt` 값으로 두 모델에 동일하게 준다.
동일한 관측 하나를 반복하므로 데이터셋 전체의 latency 분포를 측정한 결과도 아니다.

FP 준비가 불가능한 서버에서 W8A8의 부분 결과만 남기는 명령은 다음과 같다.
완전한 비교가 아니므로 의도적으로 종료 코드가 0이 아니며 speedup을 계산하지 않는다.

```bash
python scripts/benchmark_pi05_fp_w8a8.py \
  --w8a8-checkpoint "$CKPT" \
  --output-dir "$PWD/results/pi05-w8a8-only-$(date +%Y%m%d-%H%M%S)" \
  --w8a8-revision 2c9896f69be4a7dc7d7604688de3cd9e6a8ebeee
```

| CLI 인자 | 기본값 / 의미 |
| --- | --- |
| `--fp-checkpoint` | 생략 가능. 공식 PyTorch FP 로컬 디렉터리. 없으면 FP `not_run` |
| `--w8a8-checkpoint` | 필수. 공개 180계층 manifest와 `model.safetensors`, assets가 있는 로컬 디렉터리 |
| `--output-dir` | 필수. 새 경로 또는 비어 있는 결과 디렉터리 |
| `--fp-id` | `gs://openpi-assets/checkpoints/pi05_libero`; 출처 식별자 |
| `--w8a8-id` | `fatdove/pi05-libero-w8a8`; 출처 식별자 |
| `--fp-revision`, `--w8a8-revision` | 기본 `null`; 실제 준비한 가중치의 알려진 revision을 선언 |
| `--device` | `cuda:0`; 두 모델에 같은 장치 사용 |
| `--observation` | 생략 시 synthetic smoke; 지정 시 기존 검증 형식의 LIBERO NPZ |
| `--prompt` | `pick up the book` |
| `--num-steps` | `10`; denoising 반복 수 |
| `--seed` | `42`; observation/noise 생성 및 PyTorch seed |
| `--warmup` | `10`; 별도 첫 추론 이후 추가 warmup 횟수 |
| `--iterations` | `50`; 통계에 포함하는 측정 추론 횟수 |
| `--opt-m` | `1 16 32 64`; 기존 BitBLAS 커널 준비 설정 |
| `--enable-tuning` | 기본 off; 지정하면 기존 BitBLAS tuning 활성화 및 설정 기록 |

batch size는 공식 `Policy.infer()`의 단일 관측 계약에 따라 1로 고정한다.
dtype은 공식 선택적 BF16 + 나머지 FP32 방식을 유지하며 W8A8 scale은 기존 FP32다.
`torch.inference_mode()`를 사용하고 두 모델의 `pytorch_compile_mode=None`을 적용한다.
이 CLI는 compile 활성화 비교를 지원하지 않는다. compile 실험 결과와 이 기준 결과를 섞지 않는다.
두 모델은 동일 seed의 같은 초기 noise 배열을 모든 호출에 전달하며 입력/noise/정규화 통계의
해시와 모델 설정·dtype·환경의 일치 여부를 비교 조건으로 검사한다.

### 시간·메모리·크기와 결과 판독

단일 CLI가 FP → W8A8을 각각 새로운 자식 프로세스에서 실행한다. 앞 모델의 CUDA allocator와
모델 참조가 다음 모델에 남지 않도록 프로세스 종료를 격리 경계로 사용한다. 다른 프로세스의
GPU 사용과 전력/온도/clock 변화는 통제하지 않으므로 전용 GPU와 동일 환경에서 반복한다.
JAX는 CPU로 고정하여 전처리 import가 별도의 GPU 메모리를 예약하지 않게 한다.

- 모델 로딩/초기화 전체 시간과, W8A8 `initialize_kernel()` 호출을 CUDA 동기화하여 합한
  커널 준비 시간을 분리한다. 후자는 packing·BitBLAS build/compile·INT8 probe가 포함된 합계이며
  세 작업을 독립 측정한 값이 아니다. 커널 시간을 제외한 로딩 시간도 함께 남긴다.
- 첫 `policy.infer()`와 추가 warmup 총시간을 별도로 남기고, warmup 이후 매 호출 전후 CUDA를
  동기화한 wall-clock latency를 기록한다. 공식 전처리·정규화·토큰화·전송·sampling·출력 변환을
  포함한 전체 policy 기준이다. action 검사는 실행 결과에 적용하며 내부 denoising step별 profiler는 추가하지 않는다.
- 측정 호출만으로 mean/p50/p95/min/max를 계산한다. 전체 비교 통과 시 동일 통계량의
  `FP / W8A8` speedup을 계산한다. 초기 build/cache 비용을 steady-state latency에 섞지 않는다.
  디스크의 BitBLAS/TVM cache는 삭제하거나 강제로 통제하지 않으므로 로딩·첫 추론 시간의
  cold/warm cache 조건은 팀에서 별도로 기록해야 한다.
- CUDA 동기화와 peak reset으로 로딩 완료 후 allocated/reserved, 로딩 및 커널 준비 구간의
  peak allocated/reserved, 추론 구간의 peak allocated/reserved를 구분한다.
  PyTorch allocator 통계는 CUDA context나 외부 BitBLAS/TVM 직접 할당까지 포함한 전체 VRAM이
  아니므로 `nvidia-smi`와 같은 값으로 해석하지 않는다.
- 크기는 각 로컬 디렉터리의 실제 `model.safetensors` 1개에 대해 bytes와 GiB를 기록한다.
  assets/manifest/tokenizer/Hub cache 중복 사본은 합산하지 않는다. 두 가중치 크기가 확인되면
  감소 비율과 압축 배율을 남기며 디스크 직렬화 dtype과 런타임 dtype은 구분한다.

최종 결과 파일은 `--output-dir` 아래 세 개다.

| 파일 | 내용 |
| --- | --- |
| `summary.json` | 측정 시각/Git SHA, Python/PyTorch/CUDA/BitBLAS/GPU 환경, checkpoint 출처/revision, 입력·noise·dtype·compile·추론 설정, 모델별 검사/실패 단계, 시간·VRAM·크기와 비교 |
| `latency.csv` | `model,phase,iteration,latency_ms`; 첫 추론, warmup, 측정 호출을 모두 기록. 통계는 측정 phase만 사용 |
| `benchmark.log` | 실행 과정, 경고, 오류 및 traceback |

성공 판단은 exit 0, 두 모델의 실제 추론 완료, 출력 shape `(10, 7)`과 전체 action 유한성,
두 모델의 비교 조건 일치, W8A8 공개 180계층의 INT8 probe 및 실제 첫 추론 중 각 계층의
forward 호출 증가를 함께 확인한다. FP/W8A8 action 값의 일치는 요구하지 않는다.
INT8 검증은 기존 BitBLAS 진단과 정수 reference probe를 사용하며 profiler에서 Tensor Core
instruction을 직접 검증했다는 뜻은 아니다. 커널 사용을 확인하지 못하면 성공으로 기록하지 않는다.
실패 단계와 이미 얻은 모델 결과/개별 latency는 가능한 범위에서 보존하고, 미측정 값은
`null` 또는 `not_run`으로 남긴다. 실제 LIBERO success rate는 이 스크립트에서 측정하지 않는다.

### 실패 시 확인 사항과 로컬 검증 범위

- **FP 로딩 실패:** 로컬 `model.safetensors` 유무, 공식 converter/config, tied weights,
  assets의 `physical-intelligence/libero/norm_stats.json`을 확인한다. `params/`만 있으면
  공식 변환을 별도 준비하거나 FP `not_run`을 유지한다. JAX 결과로 speedup을 채우지 않는다.
- **W8A8 로딩 실패:** 고정 revision, 원본 180계층 manifest, 전체 가중치 다운로드 완료 여부,
  strict key/shape/scale 검사와 정규화 자산을 확인한다. Expert attention을 추가로 양자화하거나
  252계층 후보로 대체하지 않는다. 서로 다른 정규화 통계는 공정 비교 검사를 통과하지 못한다.
- **OOM:** `benchmark.log`의 실패 단계로 CPU RAM 부족과 CUDA OOM을 구분한다.
  다른 GPU 프로세스, GPU 선택, 실제 여유 VRAM을 확인한다. 반복 수를 줄여도 모델 로딩
  peak 자체는 줄지 않는다. 한 모델만 CPU/offload로 바꾼 결과는 같은 비교로 취급하지 않는다.
- **BitBLAS build/packing/probe 오류:** 앞 절의 커널 검사부터 다시 확인한다. PyTorch/CUDA/driver,
  BitBLAS/TVM 버전, GPU target, toolkit/compiler와 마지막 계층 traceback을 보존한다.
  probe가 통과해도 실제 입력 token M에서 추가 compile/오류가 발생할 수 있다.
- **부분 결과/종료 코드:** FP 부재, 모델 실패, 비교 조건 불일치에서 nonzero는 예상 동작이다.
  `summary.json`과 로그를 확인하고 원인을 해결한 뒤 새 출력 디렉터리로 다시 실행한다.

Windows 검증은 Python 문법, `--help`, 통계/speedup/JSON/CSV 및 가능한 CPU 회귀에 한정한다.
GPU 로딩, BitBLAS 실제 packing/probe/forward, latency/VRAM, FP 변환 수치 보존, 실제 LIBERO
관측 및 성공률, Orin 성능은 서버 팀원이 확인해야 한다. 실측 전에는 이 문서의 설명이나 공개
모델 카드 수치를 우리 benchmark 결과로 사용하지 않는다.

이번 벤치마크 구현의 Windows CPU 검증: `test_validation.py` 21개(기존 9개 + 신규 12개),
`test_public_checkpoint.py` 4개, 총 **25개 PASS**. 통계/동일 통계량 speedup 및 비교 차단,
JSON/CSV, 부분 결과/덮어쓰기 방지, 의존성을 import하지 않는 `--help`, 기존 INT8 진단,
synthetic/NPZ 입력과 dtype/shape signature를 검사했다. Python AST/메모리 내 compile와
`git diff --check`도 PASS다. 기존 `test_export_cpu.py`는 `safetensors` 미설치로 import 단계
FAIL이며 GPU 실패를 의미하지 않는다. Ruff는 미설치로 미실행이다. 환경은 위의 로컬
Python 3.10.11이며 서버의 지원 Python ≥3.11과 구분한다.

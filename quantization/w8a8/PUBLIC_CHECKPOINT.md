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

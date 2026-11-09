# π0.5 LIBERO SigLIP FFN 프루닝

기준은 풀링 커밋 `e10472a`이며, 그 부모 `98f124f`의 체크포인트 도구를 재사용합니다. 입력은 원본 `pi05_libero`를 PyTorch로 변환한 로컬 `model.safetensors`입니다. [기존 변환·복원 절차](../pruning/README.md)를 따릅니다.

`Pi0Config.load_pytorch`가 `PI0Pytorch`를 구성하고 safetensors 가중치를 로드합니다. 이미지 경로는 패치된 Transformers 4.53.2의 PaliGemma → SigLIP → 멀티모달 프로젝터입니다. 대상은 `paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[*].mlp`의 `SiglipMLP`입니다. JAX 변환기에도 같은 `fc1`과 `fc2` 체크포인트 키가 매핑되어 있습니다.

- 채널별 점수는 `L2([fc1.weight[j, :], fc1.bias[j]]) × L2(fc2.weight[:, j])`입니다. 레이어별로 상위 채널을 유지합니다.
- 동일한 인덱스로 `fc1.weight` 행, `fc1.bias` 원소, `fc2.weight` 열을 물리적으로 잘라냅니다. `fc2.bias`는 출력 차원에 속합니다. 패치당 SigLIP 폭, 16×16 패치 격자, 프로젝터 출력 2,048차원은 그대로입니다.
- 첫 비교의 목표는 약 10%입니다. 폭을 64의 배수로 맞추면 기본 4,304에서 3,904로 줄어 실제 제거 비율은 약 9.29%입니다. 실제 폭과 파라미터·바이트 절감량은 보고서에 기록합니다.
- 복원은 `pruning.run_structured_ffn.load_pruned_model(artifact_dir, device="cpu")`를 사용합니다. manifest로 먼저 FFN 폭을 재구성한 뒤 `load_state_dict(strict=True)`로 로드합니다. LIBERO 정책의 토크나이저와 정규화 자산은 원본 체크포인트 자산을 사용합니다.

## 로컬 CPU

PyTorch 단위 테스트는 축소한 두 선형층에서 슬라이스 방향, 출력 폭과 패치 수, 다른 파라미터의 동일성, 저장·재로드 후 출력 일치를 검사합니다. 전체 모델 환경에서는 풀링 캐시 테스트의 `sample-actions-skip-empty` 경로도 실행합니다.

```sh
python -m unittest pruning.test_vision_ffn pruning.test_structured_ffn pruning.test_run_structured_ffn pruning.test_lm_head_audit
```

원본 체크포인트와 패치된 Transformers 환경을 준비한 뒤 먼저 `--preflight-only`를 실행합니다. 이 검사는 추론을 수행하지 않으며 별도 디렉터리에 보고서를 남깁니다. 실제 실행용 출력 경로는 새 디렉터리여야 합니다.

```sh
python -m pruning.run_structured_ffn --target vision --device cpu --checkpoint-dir "$dense_ckpt" --output-dir "$preflight_dir" --preflight-only
```

실행 전 예상 시간은 **원본 읽기 + 압축본 쓰기와 다시 읽기 + 모델 초기화·채널 선정 + CPU 추론 3회**로 잡습니다. 체크포인트당 약 7 GiB, 유효 전송률 100–300 MiB/s를 가정하면 파일 전송에만 약 1–4분이 필요합니다. 여기에 초기화 및 추론 시간이 추가되며, RAM이 부족해 스와핑하면 추정이 크게 달라집니다. 여유 RAM과 체크포인트 크기를 확인하고 첫 실행은 denoising 1 step으로 잡습니다.

```sh
OPENPI_VISION_TOKEN_POOLING=none python -m pruning.run_structured_ffn --target vision --device cpu --checkpoint-dir "$dense_ckpt" --output-dir "$vision_artifact" --prune-fraction 0.10 --multiple-of 64 --num-steps 1 --seed 123
```

`report.json`에서 원본·프루닝 직후·재로드 후 각 단계의 유한한 액션 `[1, 10, 32]`와 프로젝트된 이미지 `[1, 256, 2048]` 두 개를 확인합니다. 세 번 모두 동일한 합성 관측과 노이즈를 사용하고, 배치 전체에서 마스킹된 오른쪽 손목 슬롯은 SigLIP 전에 제외합니다. 풀링 `none`에서 물리적 prefix는 712토큰입니다. 복원 후 액션을 프루닝 직후 액션과 비교합니다.

## 이후 공유 서버 평가

1. 비교는 원본 dense 대 vision-only입니다. 두 케이스에 같은 전처리, 카메라 필터, 풀링 `none`, 샘플링 스텝 수, 관측과 고정 노이즈를 사용합니다. 워밍업 후 GPU 동기화를 포함해 **두 뷰의 이미지 인코딩**과 **전체 모델 추론**을 각각 측정합니다. 반복 횟수, 중앙값, 관측·노이즈 식별자를 남깁니다.
2. 서버의 별도 평가 실행 파일에 vision 체크포인트를 연결하고, 실행 전에 모델 Git SHA와 평가 스크립트 Git SHA를 함께 기록합니다. 평가 스크립트가 Git 밖에 있으면 경로와 SHA-256을 기록합니다. 원본과 프루닝에 같은 파일 버전과 같은 장치·정밀도 배치를 적용합니다.
3. 물리 GPU 1에서 모델을 띄우기 전 아래 조회를 실행합니다. GPU 이름이 T400이 아니거나 다른 연산 프로세스가 표시되거나 여유 VRAM이 2,560 MiB 미만이면 중단합니다. 확인한 물리 1번 UUID로 장치를 노출하면 프로세스 안에서는 해당 GPU가 `cuda:0`입니다. 새 실행 전마다 상태를 다시 확인합니다.

```sh
nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.free --format=csv
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv
```

LIBERO에서 동일한 task·초기 상태 인덱스와 시드 123을 두 케이스에 적용하고, rollout별 성공 여부를 짝지어 기록합니다. 6개 초기 상태로 먼저 비교한다면 총 12개 rollout입니다. 풀링 파일럿의 18개가 실제 60분으로 확인될 경우 초기 추정은 `12 × 60/18 ≈ 40분`이며, 실측 완료 시간과 두 모델의 추론 시간을 얻은 뒤 다시 산정합니다.

## 연구 참고

[NViT](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Global_Vision_Transformer_Pruning_With_Hessian-Aware_Saliency_CVPR_2023_paper.html)는 Hessian 기반 중요도로 레이어마다 다른 폭을 배정합니다. [CORP](https://arxiv.org/abs/2602.05243)는 라벨 없는 캘리브레이션으로 제거된 채널의 기여를 근사해 남은 가중치에 반영합니다. 이번 비교는 학습·캘리브레이션 없는 가중치 점수로 기준을 만들고, 실제 LIBERO 관측으로 보정하는 방법은 후속 실험으로 분리합니다.

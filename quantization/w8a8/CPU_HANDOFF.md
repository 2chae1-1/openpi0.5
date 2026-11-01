# W8A8 CPU 변환 코드 전달

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
전체 체크포인트 내보내기 명령, 정규화 통계 복사, 모델 구조 검증도 포함하지 않습니다.

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

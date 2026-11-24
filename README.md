# kau-on-device-final

휘파람, 혀 찰싹, 박수 같은 짧은 소리를 16 kHz Mel 이미지로 바꿔 ResNet-18 / MobileNetV2로 분류하고, DX-M1 NPU용 `.dxnn`으로 실행하는 저장소입니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `features.py` | 1초 오디오를 uint8 `[3, 64, 64]`로 만드는 전처리 |
| `cpp_recognize/` | 같은 전처리의 C++ 구현과 마이크/파일 분류 |
| `notebooks/` | Colab 학습·컴파일, 보드 DXNN 테스트, FPS 측정 |
| `models/` | 학습 점수, 혼동 행렬, DXNN 테스트 결과 |
| `data/` | wav와 `calib_100.npy`. Git에는 없습니다 |
| `발표/` | 발표 자료. Git에는 없습니다 |

## Git에 없는 파일

가중치와 데이터는 용량 때문에 올리지 않습니다. 클론한 뒤 아래 위치에 직접 두면 노트북과 `recognize` 기본 경로가 맞습니다.

- `data/jsh/`, `data/psj/` — 클래스별 wav
- `data/calib_100.npy` — `(100, 3, 64, 64)` uint8 calibration
- `models/resnet18_v1/resnet18_v1.dxnn` — 보드 기본 모델 (`cpp_recognize`에서 `../models/resnet18_v1/resnet18_v1.dxnn`)
- `models/*/*.pt`, `models/*/*.onnx` — 학습 체크포인트와 ONNX

`cpp_recognize/recognize`는 빌드 결과입니다. `make -C cpp_recognize`로 다시 만듭니다.

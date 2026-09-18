# 고무패킹 ConvNeXt 멀티라벨 검사

Hikrobot MV-CS050-10GC 네 대가 촬영한 이미지를 각각 판별한 뒤, 한 제품에 해당하는 네 결과의 불량 클래스를 합쳐 최종 OK/NG 기록을 JSON으로 누적하는 프로토타입입니다. 모델 추론은 사진별로 독립 수행하고 최종 운영 판정만 제품 단위로 통합합니다.

> 저장소 용량 관리를 위해 원본 사진, 실시간 검사 이미지, 학습 체크포인트와 실행 산출물은 Git에 포함하지 않습니다. 학습 전 `data/images/`에 이미지 데이터를 별도로 준비하고, 모델 사용 시 체크포인트를 `runs/`에 배치해야 합니다.

## 모델과 판정 구조

- ImageNet 사전학습 ConvNeXt-Tiny backbone
- 결함 5종: `shrinkage`, `thread_defect`, `incomplete_molding`, `burr`, `contamination`
- 클래스별 독립 logit 5개를 출력하는 단일 `defect_score_head`
- `sigmoid(logit)`으로 클래스별 `defect_score` 계산
- `BCEWithLogitsLoss`로 클래스별 0/1 multi-label 학습
- 학습과 일반 추론은 threshold를 사용하지 않음
- Watch Folder만 클래스별 threshold로 최종 OK/NG를 판정

```text
이미지
  └─ ConvNeXt
      └─ defect_logits [5]
          └─ sigmoid
              └─ defect_scores [5]
                  └─ Watch Folder
                      └─ 클래스별 threshold 비교
                          └─ 양품 / 수축 / 미성형+수축 / ...
```

`defect_score`는 0.0~1.0이며 1에 가까울수록 모델이 해당 불량의 시각적 증거를 강하게 본다는 뜻입니다. 별도 severity 출력과 single-label class 출력은 없습니다. 점수들은 서로 독립적이므로 합이 1일 필요가 없고 여러 점수가 동시에 높을 수 있습니다.

Threshold는 Watch Folder 전용 운영 정책입니다. 학습 loss, `best.pt` 선택, 조기 종료 및 일반 점수 추론에는 사용되지 않습니다. 값을 낮추면 작은 의심도 NG로 잡아 엄격해지고, 높이면 느슨해집니다.

중요하게도 0/1 존재 라벨만으로 학습한 점수는 물리적인 결함 크기나 실제 심각도의 보정된 측정값은 아닙니다. 같은 불량의 경미·심각 사례를 모두 `1`로 라벨링하면 모델은 둘 다 `1`로 보내도록 학습합니다. 따라서 이 구현에서 `defect_score`는 결함 증거/판정 점수이며, threshold는 검출 민감도를 조절합니다. 경미한 결함은 통과시키고 심각한 결함만 폐기해야 한다면 라벨 `1`의 정의를 “존재”가 아니라 “폐기 대상”으로 바꾸거나 별도 정도 라벨이 필요합니다.

## 설치

Python 3.10 이상 환경에서 프로젝트 루트로 이동한 뒤 설치합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NVIDIA GPU를 사용할 경우 시스템 CUDA 환경에 맞는 PyTorch를 먼저 설치하는 편이 안전합니다. GPU가 없으면 `device: auto`가 CPU를 사용합니다. 사전학습 weight는 첫 학습 때 내려받으므로 폐쇄망이면 캐시에 준비하거나 `model.pretrained: false`로 바꾸십시오.

## 입력 이미지

한 번의 학습 또는 모델 추론에는 사진 한 장만 사용합니다. 실시간 watcher는 파일명의 같은 `object_####` 번호를 한 제품으로 보고 `camera_1`부터 `camera_4`까지 묶습니다.

```text
incomplete_molding_ng_object_0001_camera_1.png
incomplete_molding_ng_object_0001_camera_2.png
incomplete_molding_ng_object_0001_camera_3.png
incomplete_molding_ng_object_0001_camera_4.png
```

실시간 watcher는 ID를 대문자로 정규화하므로 대소문자만 다른 ID를 사용하면 안 됩니다.

## 라벨 manifest

`data/labels.csv`의 각 결함 열에는 반드시 정수 `0` 또는 `1`만 입력합니다.

```csv
sample_id,image_path,shrinkage,thread_defect,incomplete_molding,burr,contamination,split
P000001_TOP,images/P000001_TOP.jpg,0,0,0,0,0,train
P000002_SIDE,images/P000002_SIDE.jpg,1,0,1,0,0,val
```

- `0`: 해당 불량 없음
- `1`: 해당 불량 있음
- 양품: 모든 결함 열이 `0`
- 복합 불량: 해당하는 결함 열을 모두 `1`

두 번째 예시는 수축과 미성형이 동시에 있는 사진입니다. 사람은 `0.3`, `0.7` 같은 점수를 라벨링하지 않습니다. 연속적인 `defect_score`는 학습된 모델이 logits에 sigmoid를 적용해 생성합니다.

train과 val에는 결함마다 positive와 negative가 모두 있어야 합니다.

```powershell
python scripts/validate_manifest.py --config configs/default.yaml
```

## 학습

```powershell
python scripts/train.py --config configs/default.yaml
```

기본 loss는 결함별 독립 binary classification을 위한 `BCEWithLogitsLoss`입니다. `train.positive_class_weighting: auto`이면 train split에서 클래스별 `negative / positive` 비율을 `pos_weight`로 계산해 희소한 결함을 보정합니다.

```text
runs/best.pt       validation loss가 가장 낮은 checkpoint
runs/last.pt       마지막 epoch checkpoint
runs/history.json  train/validation loss 기록
```

학습 중 모델 선택과 조기 종료는 threshold가 필요 없는 validation `BCEWithLogitsLoss`만 사용합니다. 따라서 `decision.defect_score_thresholds`를 바꿔도 학습 결과와 `best.pt` 선택은 달라지지 않습니다.

기존 severity/class-head 체크포인트는 구조가 다르므로 사용할 수 없습니다. 새 manifest로 다시 학습해야 합니다.

## Threshold 조정

`configs/default.yaml`에서 Watch Folder의 결함별 최종 판정 민감도를 독립적으로 조절합니다.

```yaml
decision:
  criteria_version: TODO-v0
  defect_score_thresholds:
    shrinkage: 0.50
    thread_defect: 0.50
    incomplete_molding: 0.20  # 치명적 결함: 조금만 의심돼도 NG
    burr: 0.45
    contamination: 0.40
```

```text
score >= threshold → 해당 불량 검출
score <  threshold → 해당 불량 미검출
검출된 불량이 하나도 없음 → 양품
```

초기 `0.5`는 실행용 baseline일 뿐 생산 기준이 아닙니다. threshold 변경은 Watch Folder 결과만 바꾸며 모델을 다시 학습할 필요가 없습니다. 최종 threshold는 별도 평가에서 목표 불량 누락률을 만족하도록 정하고, 기준을 확정하면 `criteria_version`도 변경해야 결과를 추적할 수 있습니다.

## 이미지 한 건 추론

```powershell
python scripts/predict.py --config configs/default.yaml --checkpoint runs/best.pt --sample-id TEST001 --image path\to\TEST001.jpg
```

이 명령은 threshold를 적용하지 않고 클래스별 `defect_scores`만 출력합니다. 최종 OK/NG 판정은 Watch Folder에서 수행합니다.

## Watch Folder 개별 결과 JSON 예시

아래 파일은 네 장이 모두 판별될 때까지만 `runtime/results/`에 임시로 존재합니다. 제품 통합 기록이 완료되면 원본 이미지 네 장과 함께 자동 삭제됩니다.

```json
{
  "schema_version": 2,
  "sample_id": "P000123",
  "model": {
    "architecture": "convnext_tiny",
    "checkpoint_sha256_prefix": "a1b2c3d4e5f6"
  },
  "defect_scores": {
    "shrinkage": 0.72,
    "thread_defect": 0.03,
    "incomplete_molding": 0.91,
    "burr": 0.08,
    "contamination": 0.04
  },
  "decision": {
    "status": "NG",
    "result_label": "수축불량 + 미성형 불량",
    "defect_types": ["shrinkage", "incomplete_molding"],
    "defect_names_ko": ["수축불량", "미성형 불량"],
    "detected_defects": [
      {
        "defect_type": "shrinkage",
        "defect_name_ko": "수축불량",
        "defect_score": 0.72,
        "threshold": 0.5
      },
      {
        "defect_type": "incomplete_molding",
        "defect_name_ko": "미성형 불량",
        "defect_score": 0.91,
        "threshold": 0.2
      }
    ],
    "max_defect_score": 0.91,
    "applied_rule": "per_class_defect_score_threshold",
    "evaluated_thresholds": {
      "shrinkage": 0.5,
      "thread_defect": 0.5,
      "incomplete_molding": 0.2,
      "burr": 0.45,
      "contamination": 0.4
    },
    "criteria_version": "TODO-v0",
    "provisional": true
  }
}
```

## 실시간 폴더 감시

```powershell
python scripts/watch_folder.py --config configs/default.yaml --checkpoint runs/best.pt
```

카메라 프로그램은 `runtime/inbox/`에 이미지를 한 장씩 저장합니다. 파일 쓰기가 끝난 뒤 원자적으로 rename하는 방식을 권장합니다. watcher는 파일 크기와 수정 시간이 연속으로 동일하고 Pillow decode가 성공한 파일만 처리합니다.

- 사진별 임시 결과: `runtime/results/<sample_id>.json`
- 제품별 영구 누적 기록: `runtime/state/inspections.json`
- 한 제품의 네 판정이 모두 준비되면 각 카메라가 검출한 불량 클래스의 합집합으로 최종 판정
- 클래스별 최대 점수, 검출 카메라, 카메라별 판정도 `inspections.json`에 함께 기록
- 카메라별 추론 시간과 첫/마지막 이미지 도착부터 제품 통합 판정까지 걸린 시간도 기록
- 통합 기록을 안전하게 저장한 후 해당 원본 이미지 네 장과 임시 결과 JSON 네 개를 삭제

`inspections.json`은 삭제하지 않으며 다음 형식으로 제품 기록이 계속 추가됩니다.

```json
{
  "schema_version": 1,
  "inspections": [
    {
      "inspection_number": 1,
      "cycle_id": "object_0001",
      "decision": {
        "status": "NG",
        "defect_types": ["shrinkage", "burr"],
        "defect_names_ko": ["수축불량", "burr 불량"],
        "applied_rule": "union_of_camera_defects"
      },
      "camera_results": [
        {"camera": 1, "status": "NG", "defect_types": ["shrinkage"]},
        {"camera": 2, "status": "OK", "defect_types": []},
        {"camera": 3, "status": "NG", "defect_types": ["burr"]},
        {"camera": 4, "status": "OK", "defect_types": []}
      ]
    }
  ]
}
```

현재 폴더만 처리하고 종료하려면 `--once`를 사용합니다.

```powershell
python scripts/watch_folder.py --config configs/default.yaml --checkpoint runs/best.pt --once
```

별도 GUI 창에서 최신 제품 판정과 최근 이력을 실시간으로 보려면 다른 터미널에서 실행합니다.

```bash
python scripts/show_inspections.py --config configs/default.yaml
```

전체화면으로 시작하려면 `--fullscreen`을 추가합니다. 실행 중 `F11`로 전체화면을 전환하고 `Esc`로 해제할 수 있습니다.

## 촬영 및 데이터 주의사항

- 작은 burr가 crop되지 않도록 전체 형상을 포함
- 노출, gain, white balance, 조명 위치와 색온도를 고정
- Bayer 변환과 RGB/BGR 순서를 고정
- 결함별 촬영 순서를 섞어 시간·배경 shortcut 방지
- 같은 실물의 연속 frame이나 다른 시점 이미지를 서로 다른 split에 넣지 않기
- 이미지로 판별할 수 없는 기준은 모델도 안정적으로 학습할 수 없으므로 라벨 기준을 작업자 간 통일

현재 전처리는 원본 전체를 정사각 padding한 후 resize합니다. 작은 결함이 224에서 사라지면 광학 배율과 ROI를 먼저 개선하고 `image_size` 320/384도 검증하십시오.

## 테스트

```powershell
python -m pytest
```

데이터 확보 후에는 알려진 이미지→예상 multi-label JSON, checkpoint 저장/복원 동일성, 클래스별 threshold sweep, 깨진 이미지, 중복 이벤트, 실제 검사 PC 지연시간 테스트를 추가하십시오.

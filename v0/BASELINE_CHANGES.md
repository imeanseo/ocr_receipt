# Baseline → v0 변경사항

이 문서는 `baseline_code`와 `v0` 사이의 모든 변경사항을 정리합니다.

---

## 📋 변경사항 요약

| 항목 | Baseline | v0 | 변경 이유 |
|------|----------|-----|-----------|
| **Detection 방식** | QUAD (use_polygon=False) | POLY (use_polygon=True) | CLEval 평가 방식이 POLY이므로 필수 |
| **Box Threshold** | 0.4 | 0.3 | Recall 향상 (더 많은 텍스트 검출) |
| **Max Candidates** | 300 | 500 | CLEval 최대 500개 제한에 맞춤 |
| **Backbone** | ResNet-18 | ResNet-34 | 더 강력한 feature extraction |
| **Optimizer** | Adam | AdamW | 더 나은 weight decay |
| **Scheduler** | StepLR | CosineAnnealingLR | 더 부드러운 학습 곡선 |
| **Max Epochs** | 10 | 20 | 더 긴 학습으로 성능 향상 |
| **Data Augmentation** | HorizontalFlip만 | 3가지 추가 | 작은 텍스트 검출 개선 |

---

## 📁 파일별 상세 변경사항

### 1. `configs/preset/models/head/db_head.yaml`

#### 변경 전 (Baseline)
```yaml
postprocess:
  thresh: 0.3                    # Binarization threshold
  box_thresh: 0.4                # Detection Box threshold
  max_candidates: 300            # Limit the number of detection boxes
  use_polygon: False             # Detection Box Type (QUAD or POLY)
```

#### 변경 후 (v0)
```yaml
postprocess:
  thresh: 0.3                    # Binarization threshold
  box_thresh: 0.3                # Detection Box threshold (낮춰서 Recall 향상)
  max_candidates: 500            # Limit the number of detection boxes (CLEval 최대 500개)
  use_polygon: True              # Detection Box Type (POLY - CLEval 평가 방식)
```

#### 변경 내용
- ✅ `box_thresh: 0.4 → 0.3` - 더 낮은 threshold로 더 많은 텍스트 영역 검출 (Recall 향상)
- ✅ `max_candidates: 300 → 500` - CLEval 평가에서 최대 500개까지 허용하므로 증가
- ✅ `use_polygon: False → True` - **중요**: CLEval 평가 방식이 POLY이므로 필수 변경

#### 영향
- **Recall 향상**: box_thresh를 낮춰서 더 많은 텍스트 영역을 검출
- **평가 방식 일치**: POLY 방식으로 변경하여 CLEval 평가와 일치
- **검출 개수 증가**: max_candidates 증가로 더 많은 후보 검출 가능

---

### 2. `configs/preset/models/encoder/timm_backbone.yaml`

#### 변경 전 (Baseline)
```yaml
models:
  encoder:
    _target_: ${encoder_path}.TimmBackbone
    model_name: 'resnet18'
    select_features: [1, 2, 3, 4]            # Output layer
    pretrained: true
```

#### 변경 후 (v0)
```yaml
models:
  encoder:
    _target_: ${encoder_path}.TimmBackbone
    model_name: 'resnet34'           # ResNet-18 → ResNet-34로 개선 (더 강력한 feature extraction)
    select_features: [1, 2, 3, 4]            # Output layer
    pretrained: true
```

#### 변경 내용
- ✅ `model_name: 'resnet18' → 'resnet34'` - 더 깊은 네트워크로 feature extraction 능력 향상

#### 영향
- **Feature Extraction 향상**: ResNet-34는 ResNet-18보다 더 많은 레이어(34 vs 18)로 구성되어 더 강력한 특징 추출
- **작은 텍스트 검출 개선**: 더 강력한 backbone으로 작은 텍스트 영역도 더 잘 검출
- **메모리 사용량 증가**: ResNet-34는 ResNet-18보다 약 2배의 파라미터를 가짐 (약 21M vs 11M)
- **학습 시간 증가**: 더 깊은 네트워크로 인해 학습 시간이 약간 증가할 수 있음

---

### 3. `configs/preset/models/model_example.yaml`

#### 변경 전 (Baseline)
```yaml
models:
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 100
    gamma: 0.1
```

#### 변경 후 (v0)
```yaml
models:
  optimizer:
    _target_: torch.optim.AdamW
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    T_max: 20                    # CosineAnnealingLR 사용 (더 부드러운 학습)
    eta_min: 1e-6
```

#### 변경 내용
- ✅ `optimizer: Adam → AdamW` - Weight decay를 더 효과적으로 적용
- ✅ `scheduler: StepLR → CosineAnnealingLR` - 더 부드러운 학습률 감소
- ✅ `T_max: 20` - CosineAnnealingLR의 주기 설정 (max_epochs와 동일)
- ✅ `eta_min: 1e-6` - 최소 learning rate 설정

#### 영향
- **AdamW**: Weight decay를 더 효과적으로 적용하여 일반화 성능 향상
- **CosineAnnealingLR**: StepLR보다 더 부드러운 학습률 감소로 안정적인 학습
- **학습 안정성**: Cosine 스케줄링은 학습 후반부에도 적절한 학습률을 유지하여 성능 향상

---

### 4. `configs/preset/datasets/db.yaml`

#### 변경 전 (Baseline)
```yaml
train_transform:
  _target_: ${dataset_path}.DBTransforms
  transforms:
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      p: 1.0
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
      border_mode: 0
      p: 1.0
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

#### 변경 후 (v0)
```yaml
train_transform:
  _target_: ${dataset_path}.DBTransforms
  transforms:
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      p: 1.0
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
      border_mode: 0
      p: 1.0
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast    # 추가
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
    - _target_: albumentations.CLAHE                       # 추가
      clip_limit: 2.0
      tile_grid_size: [8, 8]
      p: 0.3
    - _target_: albumentations.RandomRotate90              # 추가
      p: 0.2
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

#### 변경 내용
- ✅ `RandomBrightnessContrast` 추가 - 밝기와 대비를 랜덤하게 변경 (p=0.5)
- ✅ `CLAHE` 추가 - Contrast Limited Adaptive Histogram Equalization (p=0.3)
- ✅ `RandomRotate90` 추가 - 90도 회전 (p=0.2)

#### 영향
- **데이터 다양성 증가**: 다양한 조명 조건과 노이즈에 대응하는 모델 학습
- **일반화 성능 향상**: 다양한 환경에서 촬영된 영수증 이미지에 대응
- **작은 텍스트 검출 개선**: 다양한 밝기/대비 조건에서도 텍스트를 잘 검출하도록 학습
- **학습 시간**: 약간의 증강 처리 시간이 추가되지만 미미함

---

### 5. `configs/train.yaml`

#### 변경 전 (Baseline)
```yaml
trainer:
  max_epochs: 10
  num_sanity_val_steps: 1
  log_every_n_steps: 50
  check_val_every_n_epoch: 1
  deterministic: True
```

#### 변경 후 (v0)
```yaml
trainer:
  max_epochs: 20                  # Epoch 증가 (더 긴 학습)
  num_sanity_val_steps: 1
  log_every_n_steps: 50
  check_val_every_n_epoch: 1
  deterministic: True
```

#### 변경 내용
- ✅ `max_epochs: 10 → 20` - 학습 epoch를 2배로 증가

#### 영향
- **더 긴 학습**: 모델이 더 많은 데이터를 학습하여 성능 향상 기대
- **학습 시간**: 약 2배 증가 (약 1시간 → 2시간, V100 기준)
- **성능 향상**: 충분한 학습으로 더 나은 성능 달성 가능

---

### 6. `ocr/models/head/db_postprocess.py` (버그 수정)

#### 변경 내용
- ✅ `unclip` 메서드에서 빈 결과 체크 추가 - `IndexError: list index out of range` 방지

#### 변경 전
```python
expanded = np.array(offset.Execute(distance)[0])
```

#### 변경 후
```python
expanded_polygons = offset.Execute(distance)

# 빈 결과 체크
if not expanded_polygons or len(expanded_polygons) == 0:
    return None

expanded = np.array(expanded_polygons[0])
```

---

## 🎯 변경사항의 목적

### 1. **CLEval 평가 방식에 맞춤** (최우선)
- `use_polygon: True` - POLY 방식으로 변경
- `max_candidates: 500` - CLEval 최대 제한에 맞춤

### 2. **Recall 향상** (핵심 목표)
- `box_thresh: 0.3` - 더 낮은 threshold로 더 많은 텍스트 검출
- `ResNet-34` - 더 강력한 backbone으로 작은 텍스트 검출 개선
- **데이터 증강 강화** - 다양한 조건에서도 텍스트 검출 가능

### 3. **학습 안정성 및 성능 향상**
- `AdamW` - 더 나은 weight decay
- `CosineAnnealingLR` - 부드러운 학습률 스케줄링
- `max_epochs: 20` - 충분한 학습 시간

---

## 📊 예상 성능 개선

| Metric | Baseline | v0 (예상) | 개선 |
|--------|----------|-----------|------|
| **Precision** | 0.9651 | 0.95-0.96 | 약간 감소 가능 (box_thresh 낮춤) |
| **Recall** | 0.8194 | **0.85+** | **+0.03 이상** |
| **H-Mean** | 0.8818 | **0.90+** | **+0.02 이상** |

### 개선 근거
1. **Recall 향상**: box_thresh 낮춤 + ResNet-34 + 데이터 증강
2. **H-Mean 향상**: Recall 향상으로 인한 전체 성능 개선
3. **POLY 방식**: 평가 방식과 일치하여 정확한 성능 측정

---

## 📝 변경되지 않은 항목

다음 항목들은 baseline과 동일하게 유지되었습니다:

- ✅ 이미지 크기: 640x640
- ✅ Batch size: 16
- ✅ Learning rate: 0.001
- ✅ Weight decay: 0.0001
- ✅ Loss 설정: DBLoss (negative_ratio, loss weights 등)
- ✅ Decoder: UNet 구조
- ✅ Collate function: shrink_ratio, thresh_min/max

---

## 🚀 실행 방법

### 학습
```bash
cd /root/imeanseo_ocr/v0
python runners/train.py preset=example
```

### 평가 및 예측
```bash
# 평가
python runners/test.py preset=example "checkpoint_path='outputs/ocr_training/checkpoints/best.ckpt'"

# 예측
python runners/predict.py preset=example "checkpoint_path='outputs/ocr_training/checkpoints/best.ckpt'"
```

---

## ⚠️ 주의사항

### 1. **GPU 메모리**
- ResNet-34는 ResNet-18보다 메모리를 더 사용합니다
- OOM 발생 시 `batch_size`를 줄이세요 (16 → 12 또는 8)

### 2. **학습 시간**
- 20 epoch는 약 2시간 정도 소요될 수 있습니다 (RTX 3090 기준)
- 시간이 부족하면 `max_epochs`를 조정하세요

### 3. **POLY 형식 제출**
- 예측 결과를 POLY 형식으로 제출해야 합니다
- `use_polygon: True`로 설정되어 있으므로 자동으로 POLY 형식으로 출력됩니다

### 4. **데이터 증강**
- 증강이 너무 강하면 오히려 성능이 떨어질 수 있습니다
- 현재 설정은 보수적으로 설정되어 있으나, 필요시 조정 가능합니다

---

## 🔄 되돌리기 방법

특정 변경사항을 되돌리고 싶다면:

1. **POLY → QUAD로 되돌리기**: `use_polygon: False`로 변경 (권장하지 않음 - 평가 방식과 불일치)
2. **ResNet-34 → ResNet-18**: `model_name: 'resnet18'`로 변경
3. **box_thresh 높이기**: `box_thresh: 0.4`로 변경 (Recall 감소)
4. **데이터 증강 제거**: 추가된 증강 제거
5. **Epoch 줄이기**: `max_epochs: 10`으로 변경

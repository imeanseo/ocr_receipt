# v0 → v1 변경사항

현재 성능 (v0): **H-Mean 0.9450** (Precision 0.9766, Recall 0.9203)  
목표: **H-Mean 0.95+**

---

## 📋 변경사항 요약

| 항목 | v0 | v1 | 변경 이유 |
|------|-----|-----|-----------|
| **이미지 크기** | 640x640 | **1024x1024** | 작은 텍스트 검출 개선 |
| **Batch Size** | 16 | **8** | 메모리 절약 (이미지 크기 증가 대응) |
| **Gradient Accumulation** | 없음 | **2** | Effective batch size 유지 (8 * 2 = 16) |
| **Box Threshold** | 0.3 | **0.25** | Recall 향상 |
| **Loss Weight** | prob_map: 5.0 | **prob_map: 7.0** | 텍스트 영역 검출 강화 |

---

## 📁 파일별 상세 변경사항

### 1. `configs/preset/datasets/db.yaml`

#### 이미지 크기 변경
```yaml
# 변경 전 (v0)
max_size: 640
min_width: 640
min_height: 640

# 변경 후 (v1)
max_size: 1024
min_width: 1024
min_height: 1024
```

**적용 위치**: train_transform, val_transform, test_transform, predict_transform 모두

#### Batch Size 조정
```yaml
# 변경 전 (v0)
train_dataloader:
  batch_size: 16

# 변경 후 (v1)
train_dataloader:
  batch_size: 8  # 메모리 절약
```

**이유**: 1024x1024 이미지는 640x640보다 약 2.56배 크므로 메모리 사용량이 크게 증가

---

### 2. `configs/train.yaml`

#### Gradient Accumulation 추가
```yaml
trainer:
  accumulate_grad_batches: 2  # 추가
```

**효과**: 
- Effective batch size = 8 * 2 = 16 (v0와 동일)
- 메모리 사용량은 batch_size 8 수준으로 유지
- 학습 안정성 유지

---

### 3. `configs/preset/models/head/db_head.yaml`

#### Box Threshold 조정
```yaml
# 변경 전 (v0)
box_thresh: 0.3

# 변경 후 (v1)
box_thresh: 0.25
```

**이유**: Precision이 높으므로(0.9766) Recall을 더 올릴 여지가 있음

**예상 효과**: Recall +0.005~0.01

---

### 4. `configs/preset/models/loss/db_loss.yaml`

#### Loss Weight 조정
```yaml
# 변경 전 (v0)
prob_map_loss_weight: 5.0

# 변경 후 (v1)
prob_map_loss_weight: 7.0
```

**이유**: 텍스트 영역 검출을 더 강화하여 Recall 향상

**예상 효과**: Recall +0.005~0.01

---

## 💾 메모리 관리 전략

### 메모리 사용량 비교

| 설정 | 이미지 크기 | Batch Size | 메모리 (예상) |
|------|------------|------------|---------------|
| v0 | 640x640 | 16 | ~8GB |
| v1 | 1024x1024 | 8 | ~10-12GB |
| v1 (accumulation) | 1024x1024 | 8 (effective 16) | ~10-12GB |

### 메모리 절약 방법

1. **Batch Size 감소**: 16 → 8
2. **Gradient Accumulation**: Effective batch size 유지
3. **Mixed Precision**: 필요시 추가 가능 (현재 미적용)

### OOM 발생 시 대응

1. Batch size를 더 줄이기: 8 → 4
2. Gradient accumulation 증가: 2 → 4
3. 이미지 크기 조정: 1024 → 896 또는 800

---

## 📊 예상 성능 개선

| Metric | v0 (현재) | v1 (예상) | 개선 |
|--------|-----------|-----------|------|
| **H-Mean** | 0.9450 | **0.95+** | +0.005 |
| **Precision** | 0.9766 | 0.97+ | 유지 |
| **Recall** | 0.9203 | **0.93+** | +0.01 |

### 개선 근거

1. **이미지 크기 증가**: 작은 텍스트 검출 개선 → Recall 향상
2. **box_thresh 낮춤**: 더 많은 텍스트 영역 검출 → Recall 향상
3. **Loss weight 조정**: 텍스트 영역 검출 강화 → Recall 향상

---

## ⚠️ 주의사항

### 1. **메모리 사용량**
- 1024x1024 이미지는 메모리를 많이 사용합니다
- RTX 3090 (24GB) 기준으로는 문제없지만, 더 작은 GPU는 OOM 발생 가능
- OOM 발생 시 batch_size를 4로 줄이거나 gradient accumulation을 4로 증가

### 2. **학습 시간**
- 이미지 크기 증가로 학습 시간이 약 2-3배 증가
- 20 epochs: 약 3-4시간 (RTX 3090 기준)

### 3. **효과적인 Batch Size**
- Gradient accumulation으로 effective batch size는 16으로 유지
- 학습 안정성은 v0와 동일하게 유지

---

## 🚀 실행 방법

```bash
cd /root/imeanseo_ocr/v1
python runners/train.py preset=example
```

학습 중 메모리 모니터링:
```bash
watch -n 1 nvidia-smi
```

---

## 📝 추가 개선 가능 사항

v1에서 추가로 시도할 수 있는 것들:

1. **EfficientNet-B3 backbone**: ResNet-34보다 더 강력한 feature extraction
2. **30 epochs 학습**: 더 긴 학습으로 성능 향상
3. **TTA (Test Time Augmentation)**: 예측 시 앙상블
4. **Multi-scale training**: 다양한 이미지 크기로 학습

---

## 🔄 v0와의 호환성

- ✅ 데이터 경로 동일
- ✅ 모델 구조 동일 (이미지 크기만 변경)
- ✅ 평가 방식 동일 (CLEval, POLY)
- ⚠️ 메모리 요구사항 증가

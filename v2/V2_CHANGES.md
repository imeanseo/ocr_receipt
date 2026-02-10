# v2 변경사항: EfficientNet-B3 실험

## 목표
- **백본 변경**: ResNet34 → EfficientNet-B3
- **목적**: 더 효율적인 모델로 비슷하거나 더 나은 성능 달성
- **예상**: H-Mean 0.97-0.98

## 변경사항

### 1. 백본 아키텍처 변경
**파일**: `configs/preset/models/encoder/timm_backbone.yaml`

```yaml
# 변경 전 (v1)
model_name: 'resnet34'

# 변경 후 (v2)
model_name: 'tf_efficientnet_b3'
```

### EfficientNet-B3 특징

#### 장점
1. **효율성**: ResNet34 대비 적은 파라미터로 높은 성능
   - ResNet34: ~21M 파라미터
   - EfficientNet-B3: ~12M 파라미터 (43% 감소)
2. **복합 스케일링**: 깊이, 너비, 해상도를 균형있게 조정
3. **MBConv 블록**: Mobile Inverted Bottleneck Convolution 사용
4. **Squeeze-and-Excitation**: 채널 간 관계 학습
5. **빠른 학습**: 파라미터가 적어 학습 속도 빠름

#### 단점
1. **복잡한 구조**: ResNet보다 디버깅 어려움
2. **메모리 패턴**: 다른 메모리 사용 패턴 (중간 activation 크기)

### 2. Decoder 채널 조정
**파일**: `configs/preset/models/decoder/unet.yaml`

```yaml
# 변경 전 (v1 - ResNet34용)
in_channels: [64, 128, 256, 512]

# 변경 후 (v2 - EfficientNet-B3용)
in_channels: [32, 48, 136, 384]
```

**이유**: EfficientNet-B3의 feature 채널 수가 ResNet34와 다름
- ResNet34: [64, 128, 256, 512]
- EfficientNet-B3: [32, 48, 136, 384]

### 3. 메모리 관리 (EfficientNet-B3용)
**파일**: `configs/preset/datasets/db.yaml`, `configs/train.yaml`

```yaml
# Batch size 추가 감소 (v1: 8 → v2: 4)
batch_size: 4

# Gradient accumulation 증가 (v1: 2 → v2: 4)
accumulate_grad_batches: 4  # effective batch = 4 * 4 = 16
```

**이유**: EfficientNet-B3가 ResNet34보다 중간 activation 메모리 사용량이 높음
- ResNet34 (batch=8): ~20GB
- EfficientNet-B3 (batch=8): ~24GB (OOM!)
- EfficientNet-B3 (batch=4): ~12GB ✅

### 4. 기타 설정 (v1과 동일 유지)

- **이미지 크기**: 1024x1024
- **Epochs**: 20
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **데이터 증강**: RandomBrightnessContrast, CLAHE, RandomRotate90
- **후처리**: box_thresh=0.25, use_polygon=True, max_candidates=500

## 예상 결과

### 시나리오 1: 비슷한 성능 (가장 가능성 높음)
- H-Mean: 0.975-0.980
- 학습 시간: 40-45분 (v1보다 빠름)
- **결론**: 더 효율적인 모델로 비슷한 성능

### 시나리오 2: 성능 향상
- H-Mean: 0.980-0.985
- **결론**: EfficientNet이 영수증 텍스트 검출에 더 적합

### 시나리오 3: 성능 하락
- H-Mean: 0.960-0.975
- **원인**: EfficientNet이 작은 텍스트 검출에는 ResNet보다 약할 수 있음
- **대응**: v3에서 EfficientNet-B4로 시도

## 학습 포인트

이번 실험을 통해 배울 수 있는 것:
1. **백본 선택의 중요성**: CNN 아키텍처에 따른 성능 차이
2. **효율성 vs 성능**: 파라미터 수와 성능의 관계
3. **영수증 OCR 특성**: 어떤 feature extraction이 효과적인가?

## 실행 명령어

```bash
cd /root/imeanseo_ocr/v2
python runners/train.py preset=example
```

## 다음 단계

- v2 결과에 따라 v3 방향 결정:
  - 성능 좋음 → v3: EfficientNet-B4 (더 강력)
  - 성능 나쁨 → v3: 다른 백본 시도 (ViT, Swin 등)

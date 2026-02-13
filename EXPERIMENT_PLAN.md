# OCR Text Detection 실험 계획

## 현재 상태
- **Baseline**: H-Mean 0.8818 (ResNet18)
- **v0**: H-Mean 0.9450 (ResNet34 + 최적화)
- **v1**: H-Mean 0.9786 (고해상도 + 미세조정) ✅ **현재 최고**

## 실험 로드맵

### Phase 1: 백본 아키텍처 실험 (1주차)

#### v2: EfficientNet-B3
**목표**: 경량화된 모델로 비슷한 성능 달성
- **백본**: `tf_efficientnet_b3`
- **특징**: 
  - ResNet34 대비 파라미터 수 적음
  - 효율적인 feature extraction
  - 빠른 학습 속도
- **예상**: H-Mean 0.97-0.98
- **학습 시간**: ~40분

#### v3: EfficientNet-B4
**목표**: 더 강력한 EfficientNet으로 성능 향상
- **백본**: `tf_efficientnet_b4`
- **특징**:
  - B3보다 깊은 네트워크
  - 더 많은 파라미터
- **예상**: H-Mean 0.98+
- **학습 시간**: ~50분

#### v4: Vision Transformer (ViT)
**목표**: Transformer 기반 모델 실험
- **백본**: `vit_base_patch16_224` 또는 `vit_small_patch16_224`
- **특징**:
  - Self-attention 메커니즘
  - Global context 파악 우수
  - 작은 텍스트 검출에 유리할 수 있음
- **주의사항**:
  - 메모리 사용량 높음 (batch_size 조정 필요)
  - 학습 시간 길 수 있음
- **예상**: H-Mean 0.97-0.99
- **학습 시간**: ~60분

#### v5: Swin Transformer
**목표**: 계층적 ViT로 효율성과 성능 동시 달성
- **백본**: `swin_small_patch4_window7_224`
- **특징**:
  - Shifted window attention
  - CNN과 ViT의 장점 결합
  - 다양한 스케일 feature 추출 우수
- **예상**: H-Mean 0.98+
- **학습 시간**: ~55분

---

### Phase 2: 데이터 증강 실험 (2주차)

#### v6: 강력한 증강 (Heavy Augmentation)
- **기반**: 최고 성능 백본 선택
- **증강 추가**:
  - `ShiftScaleRotate` (p=0.3)
  - `GridDistortion` (p=0.2)
  - `ElasticTransform` (p=0.2)
  - `CoarseDropout` (p=0.2)
- **목표**: 일반화 성능 향상

#### v7: 해상도 실험
- **이미지 크기**: 1024 → 1280 또는 1536
- **목표**: 더 작은 텍스트 검출 개선
- **주의**: 메모리 관리 필수

---

### Phase 3: 후처리 및 앙상블 (3주차)

#### v8: 후처리 최적화
- **box_thresh**: 0.20, 0.22, 0.25, 0.28 실험
- **unclip_ratio**: 1.5, 2.0, 2.5 실험
- **max_candidates**: 500, 700, 1000 실험

#### v9: 앙상블
- **방법 1**: 최고 성능 2-3개 모델 결합
- **방법 2**: 다양한 백본 결과 Voting
- **방법 3**: NMS (Non-Maximum Suppression) 적용

#### v10: TTA (Test Time Augmentation)
- 원본 + HorizontalFlip
- 원본 + 다양한 스케일 (0.9, 1.0, 1.1)
- 원본 + 약간의 회전 (-5°, 0°, 5°)

---

## 실험 우선순위

### 높은 우선순위 (반드시 시도)
1. ✅ **v2: EfficientNet-B3** - 효율성 확인
2. ✅ **v3: EfficientNet-B4** - 성능 향상 기대
3. ✅ **v4: ViT** - Transformer 효과 검증

### 중간 우선순위 (시간 있으면)
4. **v5: Swin Transformer** - 최신 아키텍처
5. **v6: Heavy Augmentation** - 최고 백본 + 강력한 증강
6. **v8: 후처리 최적화** - 빠른 실험 가능

### 낮은 우선순위 (여유 있으면)
7. **v7: 고해상도** - 메모리 이슈 가능성
8. **v9: 앙상블** - 마지막 단계
9. **v10: TTA** - 추론 시간 증가

---

## 실험 기록 템플릿

각 버전마다 다음 정보 기록:

```markdown
### vX: [모델명]

**설정**:
- 백본: [백본명]
- 이미지 크기: [크기]
- Batch size: [크기]
- Epochs: [에폭]
- 주요 변경사항: [변경사항]

**결과**:
- H-Mean: [점수]
- Precision: [점수]
- Recall: [점수]
- 학습 시간: [시간]

**분석**:
- 장점: [장점]
- 단점: [단점]
- 다음 실험 아이디어: [아이디어]
```

---

## 학습 포인트

### 백본 비교를 통해 배울 점
1. **CNN vs Transformer**: 어떤 구조가 영수증 텍스트 검출에 더 효과적인가?
2. **모델 크기 vs 성능**: 파라미터 수와 성능의 trade-off
3. **효율성**: 학습 시간, 추론 속도, 메모리 사용량

### 데이터 증강을 통해 배울 점
1. **일반화**: 어떤 증강이 실제 성능 향상에 도움이 되는가?
2. **과적합 방지**: 증강 강도와 성능의 관계

### 후처리를 통해 배울 점
1. **Threshold 민감도**: 후처리 파라미터가 성능에 미치는 영향
2. **Precision-Recall Trade-off**: 균형 찾기

---

## 실험 진행 방법

1. **버전 생성**: `cp -r v1 v2` (기존 설정 복사)
2. **설정 변경**: 백본 또는 증강 수정
3. **학습 실행**: `python runners/train.py preset=example`
4. **예측 실행**: `python runners/predict.py preset=example checkpoint_path=...`
5. **CSV 변환**: `python ocr/utils/convert_submission.py ...`
6. **제출 및 기록**: 대시보드 점수 기록
7. **분석**: 결과 분석 및 다음 실험 계획

---

## 참고: timm 백본 목록

```python
# EfficientNet 계열
- tf_efficientnet_b0 ~ b7
- efficientnet_b0 ~ b7
- efficientnetv2_s, m, l

# ResNet 계열
- resnet18, 34, 50, 101, 152
- resnet50d, resnet101d

# Vision Transformer 계열
- vit_tiny_patch16_224
- vit_small_patch16_224
- vit_base_patch16_224

# Swin Transformer 계열
- swin_tiny_patch4_window7_224
- swin_small_patch4_window7_224
- swin_base_patch4_window7_224

# ConvNeXt 계열
- convnext_tiny
- convnext_small
- convnext_base
```

전체 목록: `timm.list_models('*', pretrained=True)`

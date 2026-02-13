# WandB 설정 가이드

## 버전별 WandB 설정

각 버전은 동일한 프로젝트 내에서 다른 실험(run)으로 추적됩니다.

### 공통 설정
```yaml
project_name: "receipt_ocr_detection"  # 모든 버전 공통
wandb: True                             # WandB 활성화
```

### 버전별 exp_version

| 버전 | exp_version | 설명 |
|------|-------------|------|
| **v0** | `v0_resnet34_base` | ResNet34 + 기본 최적화 |
| **v1** | `v1_resnet34_high_resolution` | ResNet34 + 고해상도 (1024) |
| **v2** | `v2_efficientnet_b3` | EfficientNet-B3 |
| **v3** | `v3_efficientnet_b4` | EfficientNet-B4 (계획) |
| **v4** | `v4_vit_base` | Vision Transformer (계획) |

## WandB에서 추적되는 메트릭

### 학습 메트릭 (매 step마다)
- `train/loss`: 전체 손실
- `train/prob_loss`: Probability map 손실
- `train/thresh_loss`: Threshold map 손실
- `train/binary_loss`: Binary map 손실

### 검증 메트릭 (매 epoch마다) ⭐ 핵심!
- **`val/hmean`**: H-Mean (최종 점수)
- **`val/precision`**: Precision
- **`val/recall`**: Recall
- `val/loss`: 검증 손실

### 테스트 메트릭 (학습 완료 후)
- **`test/hmean`**: 최종 H-Mean
- **`test/precision`**: 최종 Precision
- **`test/recall`**: 최종 Recall

### 시스템 메트릭
- GPU 사용률
- GPU 메모리
- CPU 사용률
- 학습 시간

## WandB 대시보드에서 비교하는 방법

### 1. 프로젝트 페이지 접속
https://wandb.ai/imeanseo_/receipt_ocr_detection

### 2. 여러 실험(run) 선택
- 왼쪽 체크박스로 v0, v1, v2 선택
- 자동으로 비교 차트 생성

### 3. 커스텀 차트 만들기

#### 메트릭 비교 차트
```
X축: epoch
Y축: val/hmean (또는 val/precision, val/recall)
Run: v0, v1, v2
```

#### 추천 차트 구성
1. **H-Mean 비교**: epoch별 val/hmean 추이
2. **Precision vs Recall**: 두 메트릭 동시 비교
3. **Loss 추이**: train/loss와 val/loss
4. **학습 속도**: step당 시간

### 4. 최종 점수 비교 테이블

| Run | val/hmean | val/precision | val/recall | test/hmean |
|-----|-----------|---------------|------------|------------|
| v0 | 최고값 | 최고값 | 최고값 | 최종값 |
| v1 | 최고값 | 최고값 | 최고값 | 최종값 |
| v2 | 최고값 | 최고값 | 최고값 | 최종값 |

## 로깅 위치

메트릭 로깅은 `ocr/lightning_modules/ocr_pl.py`에서 자동으로 처리됩니다:

```python
# Validation epoch 종료 시 (라인 74-76)
self.log('val/recall', recall, on_epoch=True, prog_bar=True)
self.log('val/precision', precision, on_epoch=True, prog_bar=True)
self.log('val/hmean', hmean, on_epoch=True, prog_bar=True)

# Test epoch 종료 시 (라인 108-110)
self.log('test/recall', recall, on_epoch=True, prog_bar=True)
self.log('test/precision', precision, on_epoch=True, prog_bar=True)
self.log('test/hmean', hmean, on_epoch=True, prog_bar=True)
```

## WandB 로그인

처음 실행 시:
```bash
wandb login
# API 키 입력: https://wandb.ai/authorize
```

또는 환경변수로:
```bash
export WANDB_API_KEY="your_api_key"
```

## 오프라인 모드

인터넷 없이 로컬에만 저장:
```bash
wandb offline
```

나중에 동기화:
```bash
wandb sync wandb/run-XXXXXXXX-XXXXXXXX
```

## 팁

1. **실험 비교**: 대시보드에서 여러 run을 선택하면 자동으로 비교 차트 생성
2. **최고 점수 추적**: val/hmean을 기준으로 자동 정렬
3. **하이퍼파라미터 비교**: Config 탭에서 버전별 설정 차이 확인
4. **체크포인트 연결**: Model checkpoint와 WandB run ID 연결
5. **노트 추가**: 각 run에 메모 추가 가능 (예: "batch_size=4로 OOM 해결")

## 실시간 모니터링

학습 중 브라우저에서 실시간으로:
- Loss 그래프
- GPU 사용률
- epoch별 H-Mean 변화
- 예상 완료 시간

## 알림 설정

WandB 대시보드에서 알림 설정 가능:
- H-Mean > 0.98 달성 시 이메일
- 학습 완료 시 알림
- 에러 발생 시 알림

# Receipt Text Detection (패스트캠퍼스 AI16)

영수증 텍스트 검출 대회용 프로젝트. **DBNet** 기반 Detection 모델의 학습·예측·제출 파이프라인.

---

## 프로젝트 개요

- **목표**: Baseline H-Mean 0.8818 → 0.90 이상 개선
- **평가**: **CLEval** (Character-Level Evaluation), **H-Mean** 기준 순위
- **제출 형식**: POLY(다각형) 방식, 이미지당 최대 500개 글자 영역
- **데이터**: 영수증 이미지 (train 3,272 / val 404 / test 413장)
- **구조**: 베이스라인 코드는 수정하지 않고, **v0, v1, v2, …** 별도 디렉터리에서 개선 버전 관리

> 실험 과정·멘토링 질문·팀원 설명용 정리는 **`cursor_ocr.md`** (대화록 export) 참고.

---

## 최종 결과 요약

| 제출명 | 리더보드 | 최종 | 비고 |
|--------|----------|------|------|
| **V3_ResNet50** | **0.9843** | **0.9825** | **최고 성능 (권장)** |
| V4_EDAver2 | 0.9748 | 0.9698 | EDA 반영 실험 |

- **Best 제출**: V3_ResNet50 (ResNet50 + UNet + DB Head, 1024 해상도, epoch 19)
- 앙상블·TTA는 영수증 도메인 특성상 이득 없어 **단일 모델**로 제출

---

## 버전 히스토리 (Baseline → v0 → v4)

| 버전 | 설명 | 주요 변경 | 성능/비고 |
|------|------|------------|-----------|
| **Baseline** | `baseline_code/` | ResNet18, QUAD, 640 | H-Mean 0.8818 |
| **v0** | Baseline 직후 | ResNet34, **POLY**(CLEval 대응), box_thresh 0.3, AdamW, CosineAnnealing, Augmentation | **리더보드 H-Mean 0.9450** |
| **v1** | v0 개선 | **1024×1024** 해상도, batch 8 + grad accum 2, box_thresh 0.25, prob_map loss 7.0 | **리더보드 H-Mean 0.9786** |
| **v2** | `v2/` | **EfficientNet-B3** 백본, decoder 채널 조정, batch 4 + accum 4 | 경량화·다양한 백본 실험 |
| **v3** | `v3/` | **ResNet50** 백본, 1024, DB Head(thresh 0.22, box_thresh 0.35), 후처리 튜닝 | **리더보드 0.9843 (최종 best)** |
| **v4** | `v4/` | configs.TAR 기반, EDA 반영 설정 (팀원 공유·실험용) | 리더보드 0.9748 |

- **v0·v1**: 별도 폴더 없이 문서(`v2/`, `v3/` 내 `BASELINE_CHANGES.md`, `V1_CHANGES.md`, `IMPROVEMENT_PLAN.md`)로 정리되어 있으며, 코드 라인은 v2/v3에 흡수됨.
- **재현·제출**: **v3** 기준으로 학습·예측·CSV 생성.

### 개선 요약 (cursor_ocr.md 기준)

- **v0**: POLY 적용, Recall 향상(후처리·백본·증강·최적화) → Recall 0.82 → 0.92
- **v1**: 고해상도 1024 + 메모리 관리(batch↓, grad accum) → Recall 0.92 → 0.98
- **v2~v4**: 백본·설정 다양화 후, v3 ResNet50에서 최고 점수 확정

---

## 디렉터리 구조

```
imeanseo_ocr/
├── README.md              # 이 파일 (최종 정리)
├── cursor_ocr.md          # 실험 대화록·팀원 설명용 정리 (export)
├── baseline_code/         # 베이스라인 (ResNet18, 640) — 수정 금지
├── v2/                    # v2: EfficientNet-B3 실험
├── v3/                    # ★ 최종 베스트 (ResNet50, 제출·스크립트)
├── v4/                    # v4: EDA 반영 (configs.TAR 기반)
├── EXPERIMENT_PLAN.md     # 실험 로드맵
└── WANDB_GUIDE.md         # WandB 사용 가이드 (선택)
```

---

## 실행 방법 (v3 기준)

### 환경
```bash
cd v3
pip install -r requirements.txt
```
데이터 경로: `configs/preset/datasets/db.yaml` 의 `dataset_base_path` 수정  
또는 환경변수 `OCR_PROJECT_ROOT` (예: `/root/imeanseo_ocr`) 설정.

### 학습
```bash
cd v3
python runners/train.py preset=example
```

### 제출용 CSV 생성 (best 체크포인트 → predict → CSV)
```bash
cd v3
python scripts/run_submission.py "outputs/ocr_training/checkpoints/epoch=19-step=5460.ckpt"
# 또는 최신 체크포인트 자동 사용
python scripts/run_submission.py
```
→ `outputs/ocr_training/submissions/submission.csv` 생성.

### JSON → CSV만 변환
```bash
cd v3
python ocr/utils/convert_submission.py -J {json_path} -O {output_path}
```

---

## 모델 구조 (v3 Best)

- **Backbone**: ResNet50 (timm)
- **Decoder**: UNet
- **Head**: DB Head (Differentiable Binarization)
- **입력 해상도**: 1024×1024 (학습·추론 동일)
- **평가**: CLEval (Character-Level Evaluation)

---

## 참고 자료

- [DBNet](https://github.com/MhLiao/DB)
- [CLEval](https://github.com/clovaai/CLEval)
- [Hydra](https://hydra.cc/docs/intro/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- v3 상세 실행·설정: `v3/README.md`
- 버전별 변경사항: `v2/`, `v3/` 내 `*_CHANGES.md`, `BASELINE_CHANGES.md`
- 실험 과정·설명 정리: **`cursor_ocr.md`**

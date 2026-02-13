#!/bin/bash
# epoch5 체크포인트로 제출용 CSV 만드는 스크립트
# 사용법: 학습 잠시 중단한 뒤 (Ctrl+C) 이 스크립트 실행
#   cd /root/imeanseo_ocr/v4 && bash run_epoch5_submission.sh

set -e
cd "$(dirname "$0")"
CKPT="outputs/ocr_training/checkpoints/epoch=5-step=4908.ckpt"
OUT_CSV="outputs/ocr_training/submissions/submission_epoch5.csv"

export OP_CONFIG_DIR="$(pwd)/configs"
# predict 시 GPU 메모리 적게 쓰도록 배치 1
python runners/predict.py preset=example "checkpoint_path=${CKPT}" dataloaders.predict_dataloader.batch_size=1

# 방금 생성된 JSON 찾아서 CSV로 변환
JSON=$(ls -t outputs/ocr_training/submissions/*.json 2>/dev/null | head -1)
if [ -z "$JSON" ]; then
  echo "생성된 JSON을 찾을 수 없습니다."
  exit 1
fi
echo "변환: $JSON -> $OUT_CSV"
python -c "
from ocr.utils.convert_submission import convert_json_to_csv
import sys
n, path = convert_json_to_csv('$JSON', '$OUT_CSV', force=True)
print(f'제출 CSV 생성 완료: {path} (이미지 {n}개)')
"

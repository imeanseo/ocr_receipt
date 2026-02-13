#!/usr/bin/env python3
"""
Best (또는 지정) 체크포인트로 predict 실행 후 제출용 CSV 생성.

사용법 (v3 디렉터리에서):
  # 체크포인트 경로 지정
  python scripts/run_submission.py outputs/ocr_training/checkpoints/epoch=5-step=546.ckpt

  # 최신 체크포인트 자동 선택 (checkpoints/ 내 가장 최근 .ckpt)
  python scripts/run_submission.py

  # preset 지정 (기본: example)
  python scripts/run_submission.py outputs/ocr_training/checkpoints/best.ckpt preset=example
"""
import os
import sys
import subprocess
from pathlib import Path

# v3 루트로 이동
SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
os.chdir(V3_ROOT)
sys.path.insert(0, os.getcwd())

CHECKPOINT_DIR = V3_ROOT / "outputs" / "ocr_training" / "checkpoints"
SUBMISSION_DIR = V3_ROOT / "outputs" / "ocr_training" / "submissions"
DEFAULT_CSV = SUBMISSION_DIR / "submission.csv"


def find_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None
    ckpts = list(CHECKPOINT_DIR.glob("*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def find_latest_submission_json():
    if not SUBMISSION_DIR.exists():
        return None
    jsons = list(SUBMISSION_DIR.glob("*.json"))
    if not jsons:
        return None
    return max(jsons, key=lambda p: p.stat().st_mtime)


def main():
    args = [a for a in sys.argv[1:] if a.startswith("preset=") or a.startswith("checkpoint_path=")]
    rest = [a for a in sys.argv[1:] if a not in args]

    checkpoint_path = None
    preset = "example"
    for a in args:
        if a.startswith("checkpoint_path="):
            checkpoint_path = a.split("=", 1)[1].strip()
        elif a.startswith("preset="):
            preset = a.split("=", 1)[1].strip()

    # 위치 인자로 체크포인트 경로를 넘긴 경우
    if not checkpoint_path and rest:
        candidate = Path(rest[0])
        if candidate.suffix == ".ckpt" or "ckpt" in candidate.name:
            checkpoint_path = str(candidate)
        rest = rest[1:]

    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("체크포인트를 찾을 수 없습니다. 다음처럼 경로를 지정하세요:")
            print("  python scripts/run_submission.py outputs/ocr_training/checkpoints/epoch=5-step=546.ckpt")
            sys.exit(1)
        checkpoint_path = str(checkpoint_path)
        print(f"사용 체크포인트 (최신): {checkpoint_path}")

    # Hydra에 넘길 때 경로는 절대경로로. '='가 있으면 오버라이드로 파싱되므로 따옴표로 감쌈
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(Path(checkpoint_path).resolve())

    cmd = [
        sys.executable, "runners/predict.py",
        f"preset={preset}",
        f'checkpoint_path="{checkpoint_path}"',
    ]

    print("실행:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=V3_ROOT)
    if ret.returncode != 0:
        sys.exit(ret.returncode)

    json_path = find_latest_submission_json()
    if not json_path:
        print("예측 결과 JSON을 찾을 수 없습니다:", SUBMISSION_DIR)
        sys.exit(1)

    # CSV 출력 경로 (인자로 받거나 기본값)
    csv_path = rest[0] if rest and rest[0].endswith(".csv") else str(DEFAULT_CSV)
    if not os.path.isabs(csv_path):
        csv_path = str(V3_ROOT / csv_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    from ocr.utils.convert_submission import convert_json_to_csv
    result = convert_json_to_csv(str(json_path), csv_path, force=True)
    if result:
        n, out = result
        print(f"제출 CSV 생성 완료: {out} (총 {n}행)")
    else:
        print("CSV 변환 취소됨 (기존 파일 덮어쓰기 거부)")


if __name__ == "__main__":
    main()

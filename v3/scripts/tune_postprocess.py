#!/usr/bin/env python3
"""
후처리(thresh, box_thresh) 그리드 서치 — validation set으로 best 조합 찾기.

사용법 (v3 디렉터리에서):
  python scripts/tune_postprocess.py "outputs/ocr_training/checkpoints/epoch=19-step=5460.ckpt"

  또는 체크포인트 경로에 = 가 있으면 따옴표로 감싸기:
  python scripts/tune_postprocess.py 'checkpoint_path="/root/imeanseo_ocr/v3/outputs/ocr_training/checkpoints/epoch=19-step=5460.ckpt"'
"""
import os
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
os.chdir(V3_ROOT)
sys.path.insert(0, os.getcwd())

import torch
import lightning.pytorch as pl
from tqdm import tqdm

# Hydra compose로 config 로드 (runners와 동일한 config 사용)
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd

CONFIG_DIR = str(V3_ROOT / "configs")


def get_config(checkpoint_path):
    """preset=example + checkpoint_path 로 config 생성."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.2"):
        # checkpoint_path에 = 가 있으면 Hydra가 파싱하므로 따옴표로 감싼 문자열로 전달
        overrides = [
            "preset=example",
            f'checkpoint_path="{checkpoint_path}"' if "=" in checkpoint_path else f"checkpoint_path={checkpoint_path}",
        ]
        cfg = compose(config_name="test", overrides=overrides)
    return cfg


def run_grid_search(checkpoint_path, full_grid=False):
    pl.seed_everything(42, workers=True)

    config = get_config(checkpoint_path)
    from ocr.lightning_modules import get_pl_modules_by_cfg

    model_module, data_module = get_pl_modules_by_cfg(config)

    # 체크포인트 로드 (한 번만)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_module.load_state_dict(ckpt["state_dict"], strict=True)
    model_module.eval()

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

    # 그리드: 현재 0.22, 0.35 기준. 기본 4조합(~8분), --full 시 12조합(~25분)
    if full_grid:
        thresh_list = [0.18, 0.20, 0.22, 0.25]
        box_thresh_list = [0.30, 0.35, 0.40]
    else:
        thresh_list = [0.20, 0.22]
        box_thresh_list = [0.35, 0.40]

    postprocess = model_module.model.head.postprocess

    results_grid = []
    for thresh in thresh_list:
        for box_thresh in box_thresh_list:
            postprocess.thresh = thresh
            postprocess.box_thresh = box_thresh

            # validate: ckpt_path 없이 이미 로드된 모델 사용
            metrics_list = trainer.validate(model_module, data_module, ckpt_path=None)
            m = metrics_list[0]

            def _to_float(v):
                return v.item() if hasattr(v, "item") else float(v)

            results_grid.append({
                "thresh": thresh,
                "box_thresh": box_thresh,
                "precision": _to_float(m["val/precision"]),
                "recall": _to_float(m["val/recall"]),
                "hmean": _to_float(m["val/hmean"]),
            })

    # 표 출력
    print("\n[후처리 그리드 서치 결과]")
    print("thresh \\ box_thresh |", " | ".join(f"{b:5.2f}" for b in box_thresh_list))
    print("-" * (10 + 9 * len(box_thresh_list)))
    for thresh in thresh_list:
        row = [f"{r['hmean']:.4f}" for r in results_grid if r["thresh"] == thresh]
        print(f"   {thresh:.2f}        |", " | ".join(row))

    best = max(results_grid, key=lambda x: x["hmean"])
    print(f"\n► Best (val hmean): thresh={best['thresh']}, box_thresh={best['box_thresh']}")
    print(f"  precision={best['precision']:.4f}, recall={best['recall']:.4f}, hmean={best['hmean']:.4f}")

    print("\n  db_head.yaml 에 반영할 값:")
    print(f"  thresh: {best['thresh']}")
    print(f"  box_thresh: {best['box_thresh']}")

    return results_grid, best


def main():
    parser = argparse.ArgumentParser(description="후처리 thresh/box_thresh 그리드 서치")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        nargs="?",
        default=None,
        help="체크포인트 경로 (예: outputs/ocr_training/checkpoints/epoch=19-step=5460.ckpt)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 그리드 (12조합, ~25분). 기본은 4조합(~8분).",
    )
    args = parser.parse_args()

    if not args.checkpoint_path:
        # 기본: 최신 체크포인트
        ckpt_dir = V3_ROOT / "outputs" / "ocr_training" / "checkpoints"
        if not ckpt_dir.exists():
            print("체크포인트 디렉터리가 없습니다:", ckpt_dir)
            sys.exit(1)
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            print("체크포인트가 없습니다.")
            sys.exit(1)
        checkpoint_path = str(max(ckpts, key=lambda p: p.stat().st_mtime))
        print("사용 체크포인트 (최신):", checkpoint_path)
    else:
        checkpoint_path = args.checkpoint_path
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str((V3_ROOT / checkpoint_path).resolve())
        if not Path(checkpoint_path).exists():
            print("파일 없음:", checkpoint_path)
            sys.exit(1)

    run_grid_search(checkpoint_path, full_grid=args.full)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TTA(Test Time Augmentation): 원본 + Horizontal Flip 예측 후 NMS로 병합해 제출용 CSV 생성.

사용법 (v3 디렉터리에서):
  python scripts/run_tta.py outputs/ocr_training/checkpoints/epoch=19-step=5460.ckpt

  python scripts/run_tta.py checkpoint_path="path/to.ckpt" preset=example
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

# v3 루트
SCRIPT_DIR = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_DIR.parent
os.chdir(V3_ROOT)
sys.path.insert(0, os.getcwd())

import torch
import lightning.pytorch as pl
import hydra
from tqdm import tqdm
from shapely.geometry import Polygon as ShapelyPolygon

CONFIG_DIR = os.environ.get("OP_CONFIG_DIR") or "../configs"
SUBMISSION_DIR = V3_ROOT / "outputs" / "ocr_training" / "submissions"


def _flip_matrix(w):
    """flipped image coords (x_f, y_f) -> unflipped (x_u = w-1-x_f, y_u = y_f). (3,3) homogeneous."""
    return np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _polygon_to_shapely(points):
    if not points or len(points) < 3:
        return None
    try:
        p = ShapelyPolygon(points)
        if p.is_valid and not p.is_empty:
            return p
        return p.buffer(0) if not p.is_empty else None
    except Exception:
        return None


def _polygon_iou(a, b):
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def nms_merge_polygons(polygons_list, iou_thresh=0.5):
    """
    polygons_list: list of polygons, each polygon = list of [x,y] or list of (x,y).
    Returns: list of polygons (list of points) after NMS.
    """
    items = []
    for p in polygons_list:
        pts = [tuple(pt) if isinstance(pt, (list, np.ndarray)) else pt for pt in p]
        if len(pts) < 3:
            continue
        sp = _polygon_to_shapely(pts)
        if sp is not None:
            items.append((pts, sp))
    if not items:
        return []
    keep = []
    for i, (pts, sp) in enumerate(items):
        suppressed = False
        for j in keep:
            _, sj = items[j]
            if _polygon_iou(sp, sj) >= iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return [items[i][0] for i in keep]


def merge_tta_batch(boxes_orig, boxes_flip, iou_thresh=0.5):
    """이미지 하나에 대해 원본 예측과 flip 예측을 NMS로 병합. 각 boxes는 list of list of [x,y]."""
    all_boxes = list(boxes_orig) + list(boxes_flip)
    return nms_merge_polygons(all_boxes, iou_thresh=iou_thresh)


@hydra.main(config_path=CONFIG_DIR, config_name="predict", version_base="1.2")
def main(config):
    pl.seed_everything(config.get("seed", 42), workers=True)

    from ocr.lightning_modules import get_pl_modules_by_cfg

    model_module, data_module = get_pl_modules_by_cfg(config)
    ckpt_path = config.get("checkpoint_path")
    if not ckpt_path:
        print("checkpoint_path가 필요합니다. 예: python scripts/run_tta.py checkpoint_path=outputs/.../epoch=19-step=5460.ckpt")
        sys.exit(1)
    if not os.path.isabs(ckpt_path):
        ckpt_path = str(Path(ckpt_path).resolve())

    state = torch.load(ckpt_path, map_location="cpu")
    model_module.load_state_dict(state["state_dict"], strict=True)
    model_module.eval()

    device = next(model_module.parameters()).device
    predict_loader = data_module.predict_dataloader()

    submission = OrderedDict(images=OrderedDict())
    iou_thresh = getattr(config, "tta_iou_thresh", 0.5)

    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="TTA predict"):
            # 배치를 device로
            images = batch["images"].to(device)
            filenames = batch["image_filename"]
            inverse_matrix = batch["inverse_matrix"]
            B, _, H, W = images.shape

            # 원본 예측
            batch_orig = {
                "images": images,
                "image_filename": filenames,
                "inverse_matrix": inverse_matrix,
            }
            pred_orig = model_module(batch_orig)
            boxes_orig_batch, _ = model_module.model.get_polygons_from_maps(batch_orig, pred_orig)

            # Flip 후 예측: inverse_matrix를 flip 반영해 원본 이미지 좌표로 맞춤
            flip_mat = _flip_matrix(W)
            inverse_np = [np.array(inv, dtype=np.float32) if not isinstance(inv, np.ndarray) else inv.astype(np.float32) for inv in inverse_matrix]
            inverse_flip = [inv @ flip_mat for inv in inverse_np]
            batch_flip = {
                "images": torch.flip(images, dims=[3]),
                "image_filename": filenames,
                "inverse_matrix": inverse_flip,
            }
            pred_flip = model_module(batch_flip)
            boxes_flip_batch, _ = model_module.model.get_polygons_from_maps(batch_flip, pred_flip)

            for idx in range(B):
                fn = filenames[idx]
                boxes_orig = boxes_orig_batch[idx]  # list of list of [x,y]
                boxes_flip = boxes_flip_batch[idx]
                merged = merge_tta_batch(boxes_orig, boxes_flip, iou_thresh=iou_thresh)
                words = OrderedDict()
                for i, box in enumerate(merged):
                    words[f"{i + 1:04d}"] = OrderedDict(points=box)
                submission["images"][fn] = OrderedDict(words=words)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = SUBMISSION_DIR / f"tta_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"TTA 제출 JSON 저장: {json_path}")

    csv_path = SUBMISSION_DIR / f"submission_tta_{timestamp}.csv"
    from ocr.utils.convert_submission import convert_json_to_csv
    convert_json_to_csv(str(json_path), str(csv_path), force=True)
    print(f"TTA 제출 CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()

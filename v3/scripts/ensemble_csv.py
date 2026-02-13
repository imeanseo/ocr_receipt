#!/usr/bin/env python3
"""
여러 제출 CSV를 앙상블합니다. 이미지별로 폴리곤을 합친 뒤 NMS(IoU 기준)로 중복을 제거해
단일 제출용 CSV를 만듭니다.

사용법 (v3 디렉터리에서):
  python scripts/ensemble_csv.py \
    outputs/ocr_training/submissions/submission_epoch19.csv \
    outputs/ocr_training/submissions/submission_epoch15.csv \
    -o outputs/ocr_training/submissions/submission_ensemble.csv

  # IoU 임계값 변경 (기본 0.5)
  python scripts/ensemble_csv.py a.csv b.csv -o out.csv --iou-thresh 0.4
"""
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


def parse_polygon(s: str):
    """'x1 y1 x2 y2 ...' -> list of (x,y)"""
    parts = s.strip().split()
    if len(parts) < 6:  # 최소 3점
        return None
    xs = [float(parts[i]) for i in range(0, len(parts), 2)]
    ys = [float(parts[i + 1]) for i in range(0, len(parts), 2)]
    return list(zip(xs, ys))


def polygon_to_shapely(points):
    if not points or len(points) < 3:
        return None
    try:
        p = ShapelyPolygon(points)
        if p.is_valid and not p.is_empty:
            return p
        return p.buffer(0) if not p.is_empty else None
    except Exception:
        return None


def polygon_iou(a: ShapelyPolygon, b: ShapelyPolygon):
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _bboxes_overlap(a: ShapelyPolygon, b: ShapelyPolygon, margin=2):
    """빠른 제거: bbox가 겹치지 않으면 IoU 0"""
    if a is None or b is None:
        return False
    try:
        (xmin1, ymin1, xmax1, ymax1) = a.bounds
        (xmin2, ymin2, xmax2, ymax2) = b.bounds
        return not (xmax1 + margin < xmin2 or xmax2 + margin < xmin1 or
                    ymax1 + margin < ymin2 or ymax2 + margin < ymin1)
    except Exception:
        return True


def nms_polygons(polygons_list, iou_thresh=0.5):
    """
    polygons_list: list of (points, shapely_poly) or list of shapely.
    반환: 남길 폴리곤 인덱스 리스트 (순서 유지, 중복 제거). 면적 큰 순으로 유지.
    """
    if not polygons_list:
        return []
    items = []
    for i, p in enumerate(polygons_list):
        if isinstance(p, (list, tuple)):
            points, sp = p if len(p) == 2 else (p, polygon_to_shapely(p))
        else:
            sp = p
            points = list(p.exterior.coords)[:-1] if sp and not sp.is_empty else []
        if sp is None or sp.is_empty:
            continue
        items.append((i, points, sp))
    if not items:
        return []
    # 면적 큰 순 정렬 (큰 박스 우선 유지)
    items.sort(key=lambda x: x[2].area, reverse=True)
    keep = []
    for i, pts, sp in items:
        suppressed = False
        for j in keep:
            _, _, sj = items[j]
            if not _bboxes_overlap(sp, sj):
                continue
            if polygon_iou(sp, sj) >= iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return keep


def polygon_to_str(points):
    """list of (x,y) -> 'x1 y1 x2 y2 ...' (정수로 출력)"""
    return " ".join(f"{int(round(x))} {int(round(y))}" for x, y in points)


def load_csv_polygons(path: Path):
    """CSV 한 개 로드: { filename: [ polygon_str, ... ] }"""
    df = pd.read_csv(path)
    assert "filename" in df.columns and "polygons" in df.columns
    out = defaultdict(list)
    for _, row in df.iterrows():
        fn = row["filename"]
        poly_str = row["polygons"]
        if pd.isna(poly_str) or not str(poly_str).strip():
            out[fn] = []
            continue
        for part in str(poly_str).split("|"):
            part = part.strip()
            if part:
                out[fn].append(part)
    return dict(out)


def main():
    parser = argparse.ArgumentParser(description="앙상블: 여러 제출 CSV를 NMS로 병합")
    parser.add_argument("csv_files", nargs="+", type=Path, help="제출 CSV 경로들")
    parser.add_argument("-o", "--output", type=Path, required=True, help="출력 CSV 경로")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="NMS IoU 임계값 (이 이상 겹치면 하나만 유지, 기본 0.5)")
    args = parser.parse_args()

    # 모든 filename 수집 (첫 번째 CSV 기준으로 통일)
    all_filenames = set()
    per_file_polygons = defaultdict(list)  # filename -> [ (points, shapely), ... ]

    for csv_path in args.csv_files:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        data = load_csv_polygons(csv_path)
        for fn, poly_strs in data.items():
            all_filenames.add(fn)
            for s in poly_strs:
                points = parse_polygon(s)
                if points:
                    sp = polygon_to_shapely(points)
                    if sp is not None:
                        per_file_polygons[fn].append((points, sp))

    rows = []
    for fn in sorted(all_filenames):
        polys = per_file_polygons.get(fn, [])
        if not polys:
            rows.append({"filename": fn, "polygons": ""})
            continue
        keep_indices = nms_polygons(polys, iou_thresh=args.iou_thresh)
        kept = [polygon_to_str(polys[i][0]) for i in keep_indices]
        rows.append({"filename": fn, "polygons": "|".join(kept)})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"앙상블 CSV 저장: {out_path} (이미지 {len(rows)}개)")


if __name__ == "__main__":
    main()

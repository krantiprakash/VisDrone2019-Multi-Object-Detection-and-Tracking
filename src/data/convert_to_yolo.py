import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.paths import get_paths, verify_paths


# ── label definitions ─────────────────────────────────────────
VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# VisDrone class id (1-10) -> YOLO class id (0-9)
LABEL_MAP = {raw: idx for idx, raw in enumerate(sorted(VISDRONE_CLASSES.keys()))}


# ── core functions ────────────────────────────────────────────
def parse_visdrone_annotation(ann_path: Path):
    """
    Parse a single VisDrone DET annotation file.
    Format per line: x, y, w, h, score, category, truncation, occlusion

    Returns:
        boxes   : list of valid (class_id_raw, x, y, w, h)
        skipped : count of lines skipped
    """
    boxes   = []
    skipped = 0

    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                skipped += 1
                continue
            try:
                x   = int(parts[0])
                y   = int(parts[1])
                w   = int(parts[2])
                h   = int(parts[3])
                score = int(parts[4])
                cat   = int(parts[5])
            except ValueError:
                skipped += 1
                continue

            if score != 1:
                skipped += 1
                continue
            if cat not in VISDRONE_CLASSES:
                skipped += 1
                continue
            if w <= 0 or h <= 0:
                skipped += 1
                continue

            boxes.append((cat, x, y, w, h))

    return boxes, skipped


def convert_to_yolo_format(boxes: list, img_w: int, img_h: int):
    """
    Convert VisDrone boxes to YOLO normalized format.
    Input  : (cat, x_topleft, y_topleft, w, h) in pixels
    Output : list of strings "class cx cy w h" normalized to [0, 1]
    """
    yolo_lines = []
    for cat, x, y, w, h in boxes:
        cx     = (x + w / 2) / img_w
        cy     = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        cx     = max(0.0, min(1.0, cx))
        cy     = max(0.0, min(1.0, cy))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        yolo_class = LABEL_MAP[cat]
        yolo_lines.append(
            f"{yolo_class} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"
        )
    return yolo_lines


def get_image_size(img_path: Path):
    """
    Read image width and height without loading full pixel data.
    Raises exception if file is corrupted or unreadable.
    """
    with Image.open(img_path) as img:
        w, h = img.size
    return w, h


# ── per-split conversion ──────────────────────────────────────
def convert_split(img_dir: Path, ann_dir: Path, label_dir: Path, split_name: str):
    """
    Convert one dataset split (train or val).
    Writes one YOLO .txt label file per image into label_dir.
    Returns summary dict.
    """
    label_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted([f for f in ann_dir.iterdir() if f.suffix == ".txt"])

    if len(ann_files) == 0:
        raise FileNotFoundError(
            f"No annotation files found in {ann_dir}"
        )

    total_images   = 0
    skipped_images = 0
    total_boxes    = 0
    skipped_boxes  = 0
    class_counts   = defaultdict(int)

    print(f"\n[{split_name}] Converting {len(ann_files)} annotations...")

    for ann_file in tqdm(ann_files, desc=split_name):
        img_file = img_dir / (ann_file.stem + ".jpg")

        if not img_file.exists():
            skipped_images += 1
            continue

        try:
            img_w, img_h = get_image_size(img_file)
        except Exception:
            skipped_images += 1
            continue

        raw_boxes, n_skipped = parse_visdrone_annotation(ann_file)
        skipped_boxes += n_skipped

        label_path = label_dir / (ann_file.stem + ".txt")

        if len(raw_boxes) == 0:
            label_path.write_text("")
            total_images += 1
            continue

        yolo_lines = convert_to_yolo_format(raw_boxes, img_w, img_h)

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        total_images += 1
        total_boxes  += len(yolo_lines)

        for cat, *_ in raw_boxes:
            class_counts[VISDRONE_CLASSES[cat]] += 1

    summary = {
        "split"         : split_name,
        "total_images"  : total_images,
        "skipped_images": skipped_images,
        "total_boxes"   : total_boxes,
        "skipped_boxes" : skipped_boxes,
        "class_counts"  : dict(
            sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        ),
    }
    return summary


# ── det.yaml generation ───────────────────────────────────────
def generate_det_yaml(paths: dict, out_path: Path):
    """
    Generate det.yaml required by YOLO training.
    Uses absolute paths to avoid working-directory issues on Kaggle.
    """
    det_yaml = {
        "path" : str(paths["det_train"].parent),
        "train": str(paths["det_train_images"]),
        "val"  : str(paths["det_val_images"]),
        "nc"   : len(VISDRONE_CLASSES),
        "names": [VISDRONE_CLASSES[k] for k in sorted(VISDRONE_CLASSES)],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        yaml.dump(det_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n[det.yaml] Saved to {out_path}")


# ── summary printer ───────────────────────────────────────────
def print_summary(summaries: list):
    print("\n" + "=" * 55)
    print("CONVERSION SUMMARY")
    print("=" * 55)
    for s in summaries:
        print(f"\nSplit          : {s['split']}")
        print(f"Images done    : {s['total_images']}")
        print(f"Images skipped : {s['skipped_images']}")
        print(f"Boxes written  : {s['total_boxes']}")
        print(f"Boxes skipped  : {s['skipped_boxes']}")
        print("Class counts   :")
        for cls, cnt in s["class_counts"].items():
            print(f"  {cls:<20} : {cnt}")
    print("=" * 55)


# ── entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    paths = get_paths()
    verify_paths(paths)

    summaries = []

    train_summary = convert_split(
        img_dir    = paths["det_train_images"],
        ann_dir    = paths["det_train_ann"],
        label_dir  = paths["det_train_labels"],
        split_name = "DET-Train",
    )
    summaries.append(train_summary)

    val_summary = convert_split(
        img_dir    = paths["det_val_images"],
        ann_dir    = paths["det_val_ann"],
        label_dir  = paths["det_val_labels"],
        split_name = "DET-Val",
    )
    summaries.append(val_summary)

    generate_det_yaml(
        paths    = paths,
        out_path = paths["out_data"] / "det.yaml",
    )

    print_summary(summaries)
    print("\n[done] Conversion complete.")
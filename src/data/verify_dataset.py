import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.paths import get_paths, verify_paths


# ── label definitions ─────────────────────────────────────────
YOLO_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

SEED       = 42
N_SAMPLES  = 20
BOX_COLOR  = "red"
TEXT_COLOR = "white"
TEXT_BG    = "red"


# ── core functions ────────────────────────────────────────────
def read_yolo_label(label_path: Path):
    """
    Read a YOLO label file.
    Returns list of (class_id, cx, cy, w, h) all as floats.
    Returns empty list if file is empty or missing.
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx  = float(parts[1])
                cy  = float(parts[2])
                w   = float(parts[3])
                h   = float(parts[4])
            except ValueError:
                continue
            boxes.append((cls, cx, cy, w, h))
    return boxes


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """
    Convert YOLO normalized (cx, cy, w, h) to pixel (x1, y1, x2, y2).
    """
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h

    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))

    return x1, y1, x2, y2


def draw_and_save(img_path: Path, label_path: Path, save_path: Path):
    """
    Draw YOLO bounding boxes on image and save to save_path.
    Converts PIL image to numpy array before closing file handle
    so matplotlib has no dependency on open file.
    Returns number of boxes drawn.
    """
    with Image.open(img_path) as img:
        img_w, img_h = img.size
        img_array = np.array(img.convert("RGB"))

    boxes = read_yolo_label(label_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(img_array)

    for cls, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        rect = patches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=1.5,
            edgecolor=BOX_COLOR,
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = (
            YOLO_CLASS_NAMES[cls] if 0 <= cls < len(YOLO_CLASS_NAMES) else str(cls)
        )
        ax.text(
            x1, max(0.0, y1 - 4),
            class_name,
            fontsize=6,
            color=TEXT_COLOR,
            bbox=dict(facecolor=TEXT_BG, alpha=0.7, pad=1, linewidth=0),
        )

    ax.set_title(f"{img_path.name} | boxes: {len(boxes)}", fontsize=9)
    ax.axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return len(boxes)


# ── per-split verification ────────────────────────────────────
def verify_split(img_dir: Path, label_dir: Path, save_dir: Path, split_name: str):
    """
    Randomly sample N_SAMPLES images from a split,
    draw YOLO boxes, and save annotated images to save_dir.
    """
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix == ".jpg"])

    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")

    random.seed(SEED)
    samples = random.sample(img_files, min(N_SAMPLES, len(img_files)))

    print(f"\n[{split_name}] Verifying {len(samples)} samples...")

    for img_file in tqdm(samples, desc=split_name):
        label_file = label_dir / (img_file.stem + ".txt")
        save_file  = save_dir  / (img_file.stem + "_verify.png")

        n_boxes = draw_and_save(img_file, label_file, save_file)
        print(f"  {img_file.name} | boxes drawn: {n_boxes}")

    print(f"[{split_name}] Saved to {save_dir}")


# ── entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    paths = get_paths()
    verify_paths(paths)

    base_save = paths["out_data"] / "verify_samples"

    verify_split(
        img_dir    = paths["det_train_images"],
        label_dir  = paths["det_train_labels"],
        save_dir   = base_save / "train",
        split_name = "DET-Train",
    )

    verify_split(
        img_dir    = paths["det_val_images"],
        label_dir  = paths["det_val_labels"],
        save_dir   = base_save / "val",
        split_name = "DET-Val",
    )

    print(f"\n[done] All verify samples saved to {base_save}")
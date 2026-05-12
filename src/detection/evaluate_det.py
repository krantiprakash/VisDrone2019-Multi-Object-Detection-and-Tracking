import sys
import json
import yaml
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.paths import get_paths, verify_paths


# ── label definitions ─────────────────────────────────────────
VISDRONE_CLASSES = {
    1 : "pedestrian",
    2 : "people",
    3 : "bicycle",
    4 : "car",
    5 : "van",
    6 : "truck",
    7 : "tricycle",
    8 : "awning-tricycle",
    9 : "bus",
    10: "motor",
}

# VisDrone raw class id -> COCO category id (1-indexed)
LABEL_MAP     = {raw: idx for idx, raw in enumerate(sorted(VISDRONE_CLASSES.keys()), start=1)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ── settings loader ───────────────────────────────────────────
def load_settings(settings_path: Path) -> dict:
    """Load hyperparameters from settings.yaml."""
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {settings_path}")
    with open(settings_path, "r") as f:
        return yaml.safe_load(f)


# ── COCO ground truth builder ─────────────────────────────────
def build_coco_gt(img_dir: Path, ann_dir: Path) -> dict:
    """
    Build COCO-format ground truth from VisDrone DET annotations.
    Filters: score==1, class in VISDRONE_CLASSES, w>0, h>0.
    Returns COCO gt dict with keys: images, annotations, categories.
    """
    ann_files = sorted([f for f in ann_dir.iterdir() if f.suffix == ".txt"])

    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {ann_dir}")

    categories = [
        {"id": LABEL_MAP[raw], "name": name}
        for raw, name in VISDRONE_CLASSES.items()
    ]

    images      = []
    annotations = []
    ann_id      = 1

    print(f"\n[GT] Building COCO ground truth from {len(ann_files)} annotations...")

    for img_id, ann_file in enumerate(tqdm(ann_files, desc="Building GT"), start=1):
        img_file = img_dir / (ann_file.stem + ".jpg")

        if not img_file.exists():
            continue

        with Image.open(img_file) as img:
            img_w, img_h = img.size

        images.append({
            "id"       : img_id,
            "file_name": img_file.name,
            "width"    : img_w,
            "height"   : img_h,
        })

        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    score, cat  = int(parts[4]), int(parts[5])
                except ValueError:
                    continue

                if score != 1:
                    continue
                if cat not in VISDRONE_CLASSES:
                    continue
                if w <= 0 or h <= 0:
                    continue

                annotations.append({
                    "id"         : ann_id,
                    "image_id"   : img_id,
                    "category_id": LABEL_MAP[cat],
                    "bbox"       : [x, y, w, h],
                    "area"       : float(w * h),
                    "iscrowd"    : 0,
                })
                ann_id += 1

    print(f"[GT] Images: {len(images)} | Annotations: {len(annotations)}")

    return {
        "images"     : images,
        "annotations": annotations,
        "categories" : categories,
    }


# ── SAHI inference ────────────────────────────────────────────
def run_sahi_inference(
    model_path : Path,
    img_dir    : Path,
    img_id_map : dict,
    sahi_cfg   : dict,
    device     : str,
) -> list:
    """
    Run SAHI sliced inference on all images in img_dir.
    img_id_map: dict mapping filename -> image_id (from COCO GT)
    Returns list of COCO-format detection dicts.
    """
    detection_model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
        model_path           = str(model_path),
        confidence_threshold = sahi_cfg["confidence"],
        device               = device,
    )

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix == ".jpg"])
    coco_dt   = []

    print(f"\n[SAHI] Running sliced inference on {len(img_files)} images...")
    print(f"  slice    : {sahi_cfg['slice_height']}x{sahi_cfg['slice_width']}")
    print(f"  overlap  : {sahi_cfg['overlap_ratio']}")
    print(f"  conf     : {sahi_cfg['confidence']}")

    for img_file in tqdm(img_files, desc="SAHI inference"):
        if img_file.name not in img_id_map:
            continue

        img_id = img_id_map[img_file.name]

        result = get_sliced_prediction(
            image                       = str(img_file),
            detection_model             = detection_model,
            slice_height                = sahi_cfg["slice_height"],
            slice_width                 = sahi_cfg["slice_width"],
            overlap_height_ratio        = sahi_cfg["overlap_ratio"],
            overlap_width_ratio         = sahi_cfg["overlap_ratio"],
            perform_standard_pred       = True,
            postprocess_type            = "NMM",
            postprocess_match_metric    = "IOU",
            postprocess_match_threshold = 0.5,
            verbose                     = 0,
        )

        for obj in result.object_prediction_list:
            bbox   = obj.bbox
            x1     = bbox.minx
            y1     = bbox.miny
            w      = bbox.maxx - bbox.minx
            h      = bbox.maxy - bbox.miny
            score  = obj.score.value
            cat_id = obj.category.id + 1

            if w <= 0 or h <= 0:
                continue

            coco_dt.append({
                "image_id"   : img_id,
                "category_id": cat_id,
                "bbox"       : [x1, y1, w, h],
                "score"      : float(score),
            })

    print(f"[SAHI] Total detections: {len(coco_dt)}")
    return coco_dt


# ── pycocotools evaluation ────────────────────────────────────
def evaluate_coco(coco_gt_dict: dict, coco_dt_list: list) -> dict:
    """
    Run pycocotools COCOeval on GT and predictions.
    Returns dict with all required metrics.
    """
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    if len(coco_dt_list) == 0:
        print("[EVAL] WARNING: No detections found. Returning zero metrics.")
        return {
            "mAP50"    : 0.0,
            "mAP5095"  : 0.0,
            "mAR50"    : 0.0,
            "mAR5095"  : 0.0,
            "precision": 0.0,
            "recall"   : 0.0,
        }

    coco_dt   = coco_gt.loadRes(coco_dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats    = coco_eval.stats
    prec_arr = coco_eval.eval["precision"]
    rec_arr  = coco_eval.eval["recall"]

    precision = float(np.mean(prec_arr[prec_arr > -1])) if np.any(prec_arr > -1) else 0.0
    recall    = float(np.mean(rec_arr[rec_arr > -1]))   if np.any(rec_arr > -1)  else 0.0

    results = {
        "mAP50"    : round(float(stats[1]), 4),
        "mAP5095"  : round(float(stats[0]), 4),
        "mAR50"    : round(float(stats[6]), 4),
        "mAR5095"  : round(float(stats[8]), 4),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
    }

    # per-class mAP50
    per_class = {}
    for cat_id, cat_info in coco_gt.cats.items():
        coco_eval_cls               = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cls.params.catIds = [cat_id]
        coco_eval_cls.evaluate()
        coco_eval_cls.accumulate()
        coco_eval_cls.summarize()
        if len(coco_eval_cls.stats) > 1:
            per_class[cat_info["name"]] = round(float(coco_eval_cls.stats[1]), 4)
        else:
            per_class[cat_info["name"]] = 0.0

    results["per_class_mAP50"] = per_class
    return results


# ── summary printer ───────────────────────────────────────────
def print_results(results: dict, split: str = "test") -> None:
    print("\n" + "=" * 50)
    print(f"DETECTION EVALUATION RESULTS (DET-{split})")
    print("=" * 50)
    print(f"  mAP@0.50       : {results['mAP50']}")
    print(f"  mAP@0.50:0.95  : {results['mAP5095']}")
    print(f"  mAR@0.50       : {results['mAR50']}")
    print(f"  mAR@0.50:0.95  : {results['mAR5095']}")
    print(f"  Precision      : {results['precision']}")
    print(f"  Recall         : {results['recall']}")
    print("\n  Per-class mAP@0.50:")
    for cls, val in results["per_class_mAP50"].items():
        print(f"    {cls:<20} : {val}")
    print("=" * 50)


# ── entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import torch

    paths = get_paths()
    verify_paths(paths)

    settings_path = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    settings      = load_settings(settings_path)
    sahi_cfg      = settings["sahi"]

    on_kaggle = os.path.exists("/kaggle/input")
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    if on_kaggle:
        model_path = Path("/kaggle/input/datasets/krantiprakash/visdrone-weights-v1/Weights/best.pt")
    else:
        model_path = paths["out_detection"] / "yolo26x_visdrone" / "weights" / "best.pt"

    # set EVAL_SPLIT to "test" or "val"
    EVAL_SPLIT = "test"
    # EVAL_SPLIT = "val"

    if EVAL_SPLIT == "test":
        eval_img_dir = paths["det_test_images"]
        eval_ann_dir = paths["det_test_ann"]
        result_fname = "detection_results_test.json"
    else:
        eval_img_dir = paths["det_val_images"]
        eval_ann_dir = paths["det_val_ann"]
        result_fname = "detection_results_val.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"best.pt not found at {model_path}\n"
            f"Run src/detection/train_yolo.py first."
        )

    print(f"\n[eval] Model     : {model_path}")
    print(f"[eval] Device    : {device}")
    print(f"[eval] Split     : {EVAL_SPLIT}")
    print(f"[eval] Image dir : {eval_img_dir}")

    coco_gt_dict = build_coco_gt(
        img_dir = eval_img_dir,
        ann_dir = eval_ann_dir,
    )

    img_id_map = {img["file_name"]: img["id"] for img in coco_gt_dict["images"]}

    coco_dt_list = run_sahi_inference(
        model_path = model_path,
        img_dir    = eval_img_dir,
        img_id_map = img_id_map,
        sahi_cfg   = sahi_cfg,
        device     = device,
    )

    results = evaluate_coco(coco_gt_dict, coco_dt_list)

    print_results(results, split=EVAL_SPLIT)

    out_path = paths["out_evaluation"] / result_fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")
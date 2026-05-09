import sys
import shutil
import yaml
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.paths import get_paths, verify_paths


# ── loaders ───────────────────────────────────────────────────
def load_settings(settings_path: Path) -> dict:
    """
    Load hyperparameters from settings.yaml.
    Raises FileNotFoundError if settings file is missing.
    """
    if not settings_path.exists():
        raise FileNotFoundError(
            f"settings.yaml not found at {settings_path}"
        )
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
    return settings


def validate_det_yaml(det_yaml_path: Path) -> None:
    """
    Confirm det.yaml exists and contains required keys.
    Raises FileNotFoundError or KeyError with clear message if invalid.
    """
    if not det_yaml_path.exists():
        raise FileNotFoundError(
            f"det.yaml not found at {det_yaml_path}\n"
            f"Run src/data/convert_to_yolo.py first."
        )
    with open(det_yaml_path, "r") as f:
        det = yaml.safe_load(f)

    for key in ["train", "val", "nc", "names"]:
        if key not in det:
            raise KeyError(
                f"det.yaml is missing required key: '{key}'"
            )


# ── loss curves ───────────────────────────────────────────────
def plot_loss_curves(results_csv: Path, save_dir: Path) -> None:
    """
    Read results.csv from Ultralytics training output and save
    separate train and val loss curves to save_dir.
    """
    if not results_csv.exists():
        print("[curves] results.csv not found, skipping curve plots.")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    if "epoch" not in df.columns:
        print("[curves] epoch column not found in results.csv, skipping.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # train loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    if "train/box_loss" in df.columns:
        ax.plot(df["epoch"], df["train/box_loss"], label="box_loss")
    if "train/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
    if "train/dfl_loss" in df.columns:
        ax.plot(df["epoch"], df["train/dfl_loss"], label="dfl_loss")
    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "train_loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"[curves] Train loss curve saved to {save_dir / 'train_loss_curve.png'}")

    # val loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    if "val/box_loss" in df.columns:
        ax.plot(df["epoch"], df["val/box_loss"], label="box_loss")
    if "val/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["val/cls_loss"], label="cls_loss")
    if "val/dfl_loss" in df.columns:
        ax.plot(df["epoch"], df["val/dfl_loss"], label="dfl_loss")
    ax.set_title("Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "val_loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"[curves] Val loss curve saved to {save_dir / 'val_loss_curve.png'}")


# ── download helper ───────────────────────────────────────────
def save_outputs_for_download(train_out_dir: Path, download_dir: Path) -> None:
    """
    Copy key training outputs to a single flat folder for easy
    download from Kaggle output panel.
    Files copied: best.pt, results.csv, train_loss_curve.png,
    val_loss_curve.png, results.png, confusion_matrix.png.
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        train_out_dir / "weights" / "best.pt",
        train_out_dir / "results.csv",
        train_out_dir / "results.png",
        train_out_dir / "confusion_matrix.png",
        train_out_dir.parent / "curves" / "train_loss_curve.png",
        train_out_dir.parent / "curves" / "val_loss_curve.png",
    ]

    print("\n[download] Copying outputs for download...")
    for src in files_to_copy:
        if src.exists():
            dst = download_dir / src.name
            shutil.copy2(src, dst)
            print(f"  copied: {src.name}")
        else:
            print(f"  missing: {src.name}")

    print(f"[download] All outputs saved to {download_dir}")


# ── training ──────────────────────────────────────────────────
def train(paths: dict, settings: dict) -> None:
    """
    Fine-tune YOLO26x on VisDrone-DET using parameters from settings.yaml.
    Saves best checkpoint and plots to output/detection/.
    """
    det_cfg  = settings["detection"]
    det_yaml = paths["out_data"] / "det.yaml"

    validate_det_yaml(det_yaml)

    out_dir = paths["out_detection"]
    out_dir.mkdir(parents=True, exist_ok=True)

    last_pt   = out_dir / "yolo26x_visdrone" / "weights" / "last.pt"
    device    = det_cfg["device"] if torch.cuda.is_available() else "cpu"

    if last_pt.exists():
        print(f"\n[train] Resuming from checkpoint: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(
            resume  = True,
            epochs  = det_cfg["epochs"],
            device  = device,
            project = str(out_dir),
            name    = "yolo26x_visdrone",
            exist_ok= True,
        )
    else:
        print("\n[train] No checkpoint found. Starting fresh fine-tuning...")
        model = YOLO(det_cfg["model"])

        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  total params : {total_params:,}")
        print(f"  note         : all params fine-tuned, requires_grad set internally by Ultralytics")

        print("[train] Starting fine-tuning on VisDrone-DET...")
        print(f"  model    : {det_cfg['model']}")
        print(f"  imgsz    : {det_cfg['imgsz']}")
        print(f"  epochs   : {det_cfg['epochs']}")
        print(f"  batch    : {det_cfg['batch']}")
        print(f"  lr0      : {det_cfg['lr0']}")
        print(f"  patience : {det_cfg['patience']}")
        print(f"  workers  : {det_cfg['workers']}")
        print(f"  device   : {device}")
        print(f"  data     : {det_yaml}")
        print(f"  save_dir : {out_dir}\n")

        model.train(
            data       = str(det_yaml),
            epochs     = det_cfg["epochs"],
            imgsz      = det_cfg["imgsz"],
            batch      = det_cfg["batch"],
            lr0        = det_cfg["lr0"],
            patience   = det_cfg["patience"],
            workers    = det_cfg["workers"],
            device     = device,
            project    = str(out_dir),
            name       = "yolo26x_visdrone",
            exist_ok   = True,
            pretrained = True,
            verbose    = True,
        )

    train_out_dir = out_dir / "yolo26x_visdrone"
    best_pt       = train_out_dir / "weights" / "best.pt"

    if best_pt.exists():
        print(f"\n[train] Best checkpoint saved: {best_pt}")
    else:
        print("\n[train] WARNING: best.pt not found. Check training logs.")

    curves_dir = out_dir / "curves"
    plot_loss_curves(train_out_dir / "results.csv", curves_dir)

    download_dir = out_dir / "download"
    save_outputs_for_download(train_out_dir, download_dir)


# ── entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    paths = get_paths()
    verify_paths(paths)

    settings_path = Path(__file__).resolve().parents[2] / "configs" / "settings.yaml"
    settings      = load_settings(settings_path)

    train(paths, settings)
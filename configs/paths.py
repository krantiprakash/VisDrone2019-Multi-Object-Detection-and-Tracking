import os
from pathlib import Path


def get_paths() -> dict:
    """
    Auto-detect runtime environment and return all project paths.
    Supports: local Windows/Linux development and Kaggle GPU environment.
    """
    if os.path.exists("/kaggle/input"):
        base_data = Path("/kaggle/input/datasets/krantiprakash/dataset/Dataset")
        base_work = Path("/kaggle/working")
        base_root = Path("/kaggle/working")
    else:
        base_root = Path(__file__).resolve().parents[1]
        base_data = base_root / "Dataset"
        base_work = base_root / "output"

    paths = {
        # dataset roots
        "det_train"       : base_data / "VisDrone2019-DET-train",
        "det_val"         : base_data / "VisDrone2019-DET-val",
        "det_test"        : base_data / "VisDrone2019-DET-test",
        "mot_val"         : base_data / "VisDrone2019-MOT-val",

        # DET train subfolders
        "det_train_images": base_data / "VisDrone2019-DET-train" / "images",
        "det_train_ann"   : base_data / "VisDrone2019-DET-train" / "annotations",
        "det_train_labels": base_data / "VisDrone2019-DET-train" / "labels",

        # DET val subfolders
        "det_val_images"  : base_data / "VisDrone2019-DET-val" / "images",
        "det_val_ann"     : base_data / "VisDrone2019-DET-val" / "annotations",
        "det_val_labels"  : base_data / "VisDrone2019-DET-val" / "labels",

        # DET test subfolders
        "det_test_images" : base_data / "VisDrone2019-DET-test" / "images",
        "det_test_ann"    : base_data / "VisDrone2019-DET-test" / "annotations",
        "det_test_labels" : base_data / "VisDrone2019-DET-test" / "labels",

        # MOT val subfolders
        "mot_val_seq"     : base_data / "VisDrone2019-MOT-val" / "sequences",
        "mot_val_ann"     : base_data / "VisDrone2019-MOT-val" / "annotations",

        # output folders per stage
        "out_data"        : base_work / "data",
        "out_detection"   : base_work / "detection",
        "out_reid"        : base_work / "reid",
        "out_tracking"    : base_work / "tracking",
        "out_evaluation"  : base_work / "evaluation",
        "out_export"      : base_work / "export",

        # project root
        "root"            : base_root,
    }

    # create all output directories automatically
    output_keys = [k for k in paths if k.startswith("out_")]
    for key in output_keys:
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def verify_paths(paths: dict) -> None:
    """
    Verify required dataset input paths exist.
    Test paths are optional — skipped if not found.
    Raises FileNotFoundError if any required path is missing.
    """
    required_keys = [
        "det_train_images",
        "det_train_ann",
        "det_val_images",
        "det_val_ann",
        "mot_val_seq",
        "mot_val_ann",
    ]

    for key in required_keys:
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required dataset path [{key}]: {path}\n"
                f"Check your Dataset folder structure."
            )

    print("[paths] All required dataset paths verified.")

    optional_keys = ["det_test_images", "det_test_ann"]
    for key in optional_keys:
        path = paths[key]
        if path.exists():
            print(f"[paths] Optional path found     [{key}]: {path}")
        else:
            print(f"[paths] Optional path not found [{key}]: {path} (skipping)")


if __name__ == "__main__":
    paths = get_paths()
    verify_paths(paths)
    print("\nResolved paths:")
    for key, val in paths.items():
        print(f"  {key:<20} : {val}")
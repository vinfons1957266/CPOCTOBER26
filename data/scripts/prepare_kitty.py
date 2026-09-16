from pathlib import Path
import re
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent
RAW_ROOT   = DATA_DIR / "raw" / "val_selection_cropped"
OUT_ROOT   = DATA_DIR / "kitti_ood"

MAX_DEPTH_M = 80.0

# Estrae l'identificativo comune (drive + frame + camera) ignorando
# il token "image" / "groundtruth_depth" che differisce tra le due cartelle.
_KEY_RE = re.compile(r"(.+_sync)_(?:image|groundtruth_depth)_(\d+_image_\d+)")


def _extract_key(filename: str) -> str:
    m = _KEY_RE.match(filename)
    if not m:
        raise ValueError(f"Nome file inatteso, pattern non riconosciuto: {filename}")
    return f"{m.group(1)}__{m.group(2)}"


def convert_kitti_validation(raw_root: Path, out_root: Path) -> None:
    image_dir = raw_root / "image"
    depth_dir = raw_root / "groundtruth_depth"

    rgb_files   = {_extract_key(p.name): p for p in image_dir.glob("*.png")}
    depth_files = {_extract_key(p.name): p for p in depth_dir.glob("*.png")}

    common_keys = sorted(set(rgb_files) & set(depth_files))
    missing = (set(rgb_files) ^ set(depth_files))
    if missing:
        print(f"[WARN] {len(missing)} file senza coppia corrispondente, saltati.")
    if not common_keys:
        raise FileNotFoundError(f"Nessuna coppia trovata in {image_dir} / {depth_dir}")

    out_rgb   = out_root / "rgb"
    out_depth = out_root / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    for i, key in enumerate(common_keys):
        rgb = Image.open(rgb_files[key]).convert("RGB")
        rgb.save(out_rgb / f"{i:04d}.png")

        depth_raw = np.array(Image.open(depth_files[key]), dtype=np.uint16)
        depth_m   = depth_raw.astype(np.float32) / 256.0
        depth_8bit = np.clip(depth_m / MAX_DEPTH_M * 255, 0, 255).astype(np.uint8)
        Image.fromarray(depth_8bit, mode="L").save(out_depth / f"{i:04d}.png")

    print(f"Convertite {len(common_keys)} coppie in {out_root}")


if __name__ == "__main__":
    convert_kitti_validation(RAW_ROOT, OUT_ROOT)
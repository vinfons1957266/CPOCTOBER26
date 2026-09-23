# data/scripts/prepare_nyu.py
from pathlib import Path
import h5py, numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent   # .../data/scripts
DATA_DIR   = SCRIPT_DIR.parent                  # .../data
RAW_PATH   = DATA_DIR / "raw" / "nyu_depth_v2_labeled.mat"
OUT_ROOT   = DATA_DIR / "nyu_depth_v2"

def convert_nyu_mat(mat_path: Path, out_root: Path, train_ratio: float = 0.8):
    with h5py.File(mat_path, "r") as f:
        images, depths = f["images"], f["depths"]
        n = images.shape[0]
        n_train = int(n * train_ratio)

        for split, idx_range in [("train", range(0, n_train)),
                                   ("test",  range(n_train, n))]:
            rgb_dir   = out_root / split / "rgb"
            depth_dir = out_root / split / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            for i, idx in enumerate(idx_range):
                rgb = np.transpose(images[idx], (2, 1, 0)).astype(np.uint8)
                Image.fromarray(rgb).save(rgb_dir / f"{i:04d}.png")

                depth_m = depths[idx].T
                depth_8bit = np.clip(depth_m / 10.0 * 255, 0, 255).astype(np.uint8)
                Image.fromarray(depth_8bit, mode="L").save(depth_dir / f"{i:04d}.png")

if __name__ == "__main__":
    convert_nyu_mat(RAW_PATH, OUT_ROOT)
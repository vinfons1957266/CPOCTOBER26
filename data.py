"""
=============================================================================
Sezione 4 — DATA
=============================================================================
Dataset custom (NYU Depth V2 ID, KITTI OOD) e funzione get_dataloaders().
Entrambi i dataset supportano fallback sintetico se i dati reali non sono
presenti sul disco.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


# ===========================================================================
# NYU Depth V2 — In-Distribution (indoor)
# ===========================================================================

class NYUDepthV2Dataset(Dataset):
    """
    Dataset In-Distribution — scene indoor.

    Se la directory root contiene dati reali (sottocartelle rgb/ e depth/),
    vengono caricati.  Altrimenti viene generato un **fallback sintetico**
    (immagini RGB casuali + mappe di profondità smooth nell'intervallo
    0.5–10 m).
    """

    def __init__(
        self,
        root: str = "./data/nyu_depth_v2",
        split: str = "train",
        num_samples: int = 800,
        transform=None,
        depth_transform=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.num_samples = num_samples
        self.transform = transform
        self.depth_transform = depth_transform

        # Tentativo di individuare dati reali --------------------------------
        self.real_data = False
        self.rgb_paths: list[str] = []
        self.depth_paths: list[str] = []

        split_dir = os.path.join(root, split)
        rgb_dir   = os.path.join(split_dir, "rgb")
        depth_dir = os.path.join(split_dir, "depth")

        if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
            rgb_files   = sorted(os.listdir(rgb_dir))
            depth_files = sorted(os.listdir(depth_dir))
            if len(rgb_files) > 0 and len(rgb_files) == len(depth_files):
                self.rgb_paths   = [os.path.join(rgb_dir, f)   for f in rgb_files]
                self.depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
                self.num_samples = len(self.rgb_paths)
                self.real_data = True

        if not self.real_data:
            print(f"[NYUDepthV2Dataset] Synthetic fallback "
                  f"({split}, n={self.num_samples})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.real_data:
            from PIL import Image
            rgb   = Image.open(self.rgb_paths[idx]).convert("RGB")
            depth = Image.open(self.depth_paths[idx]).convert("L")
            rgb   = rgb.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            depth = depth.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))

            rgb   = transforms.ToTensor()(rgb)            # (3, H, W) [0,1]
            depth = transforms.ToTensor()(depth).float()   # (1, H, W) [0,1]
            depth = depth * 10.0                           # scala a ~metri
        else:
            # --- Generazione sintetica indoor ---
            rng    = np.random.RandomState(idx)
            rgb_np = rng.rand(3, IMAGE_SIZE[0], IMAGE_SIZE[1]).astype(np.float32)
            rgb    = torch.from_numpy(rgb_np)

            # Profondità sintetica smooth (rumore a bassa frequenza) [0.5, 10.0] m
            base  = rng.rand(1, IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8).astype(np.float32)
            depth = torch.from_numpy(base)
            depth = F.interpolate(
                depth.unsqueeze(0), size=IMAGE_SIZE,
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            depth = depth * 9.5 + 0.5

        if self.transform is not None:
            rgb = self.transform(rgb)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        return rgb, depth


# ===========================================================================
# KITTI — Out-of-Distribution (outdoor)
# ===========================================================================

class KITTIOODDataset(Dataset):
    """
    Dataset Out-of-Distribution — scene outdoor (guida autonoma).

    Se la directory root contiene dati reali vengono caricati, altrimenti
    fallback sintetico con statistiche diverse (più luminose, profondità
    1–80 m).
    """

    def __init__(
        self,
        root: str = "./data/kitti_ood",
        num_samples: int = 200,
        transform=None,
        depth_transform=None,
    ):
        super().__init__()
        self.root = root
        self.num_samples = num_samples
        self.transform = transform
        self.depth_transform = depth_transform

        # Tentativo di individuare dati reali --------------------------------
        self.real_data = False
        self.rgb_paths: list[str] = []
        self.depth_paths: list[str] = []

        rgb_dir   = os.path.join(root, "rgb")
        depth_dir = os.path.join(root, "depth")

        if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
            rgb_files   = sorted(os.listdir(rgb_dir))
            depth_files = sorted(os.listdir(depth_dir))
            if len(rgb_files) > 0 and len(rgb_files) == len(depth_files):
                self.rgb_paths   = [os.path.join(rgb_dir, f)   for f in rgb_files]
                self.depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
                self.num_samples = len(self.rgb_paths)
                self.real_data = True

        if not self.real_data:
            print(f"[KITTIOODDataset] Synthetic fallback (n={self.num_samples})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.real_data:
            from PIL import Image
            rgb   = Image.open(self.rgb_paths[idx]).convert("RGB")
            depth = Image.open(self.depth_paths[idx]).convert("L")
            rgb   = rgb.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            depth = depth.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))

            rgb   = transforms.ToTensor()(rgb)
            depth = transforms.ToTensor()(depth).float()
            depth = depth * 80.0  # range profondità outdoor
        else:
            # --- Generazione sintetica outdoor ---
            rng    = np.random.RandomState(idx + 100_000)
            rgb_np = (rng.rand(3, IMAGE_SIZE[0], IMAGE_SIZE[1]).astype(np.float32)
                      * 0.6 + 0.4)
            rgb    = torch.from_numpy(rgb_np)

            base  = rng.rand(1, IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8).astype(np.float32)
            depth = torch.from_numpy(base)
            depth = F.interpolate(
                depth.unsqueeze(0), size=IMAGE_SIZE,
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            depth = depth * 79.0 + 1.0  # [1.0, 80.0] m

        if self.transform is not None:
            rgb = self.transform(rgb)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        return rgb, depth


# ===========================================================================
# DataLoader factory
# ===========================================================================

def get_dataloaders(
    nyu_root:    str = "./data/nyu_depth_v2",
    kitti_root:  str = "./data/kitti_ood",
    nyu_train_n: int = 800,
    nyu_test_n:  int = 200,
    kitti_ood_n: int = 200,
) -> tuple:
    """
    Costruisce e restituisce i DataLoader per:
        - train  (NYU Depth V2, ID)
        - test   (NYU Depth V2, ID)
        - ood    (KITTI, OOD)

    Returns:
        (train_loader, id_test_loader, ood_test_loader)
    """
    # Normalizzazione ImageNet per l'encoder pre-addestrato
    img_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_ds = NYUDepthV2Dataset(
        root=nyu_root, split="train",
        num_samples=nyu_train_n,
        transform=img_transform,
    )
    test_ds = NYUDepthV2Dataset(
        root=nyu_root, split="test",
        num_samples=nyu_test_n,
        transform=img_transform,
    )
    ood_ds = KITTIOODDataset(
        root=kitti_root,
        num_samples=kitti_ood_n,
        transform=img_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    id_test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    ood_test_loader = DataLoader(
        ood_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, id_test_loader, ood_test_loader

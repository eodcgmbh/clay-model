import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import math

from typing import Optional, Tuple

import yaml
from box import Box
from torchvision.transforms import v2
import rasterio

class EmbeddingDatasetGFM3(Dataset):
    """
    Dataset that loads pre/post Clay embeddings, corresponding GFM labels/masks,
    and paired pre/post SAR chips ("images") for auxiliary fusion.
    Returns a dict with keys used by embedding_gfm_classifier_v3:
      - pre_embedding, post_embedding: [N_patches, D]
      - pre_image, post_image: [C, H, W] normalized
      - label: [H, W] (uint8 0/1)
      - ignore_mask: [H, W] (bool)
    """

    def __init__(
        self,
        embeddings_dir: str,
        label_dir: str,
        chips_dir: Optional[str] = None,
        metadata_path: Optional[str] = None,
        platform: Optional[str] = None,
        target_size: Tuple[int, int] = (224, 224),
        max_samples: Optional[int] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.label_dir = Path(label_dir)
        self.chips_dir = (Path(chips_dir) if chips_dir is not None else self.embeddings_dir.parent)
        self.target_size = target_size

        # Optional normalization for SAR chips
        self.transform = None
        if metadata_path is not None and platform is not None:
            meta = Box(yaml.safe_load(open(metadata_path)))
            mean = list(meta[platform].bands.mean.values())
            std = list(meta[platform].bands.std.values())
            self.transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

        # Pair pre/post embedding files within the directory
        all_embed_files = [p.name for p in self.embeddings_dir.glob("*.npy")]
        key_to_embed = {}
        for fname in all_embed_files:
            key = self._key_from_name(fname)
            key_to_embed.setdefault(key, []).append(fname)
        paired_embed_items = [(k, sorted(v)) for k, v in key_to_embed.items() if len(v) == 2]
        paired_embed_items.sort(key=lambda x: x[0])
        self.pre_embedding_files = [v[0] for _, v in paired_embed_items]
        self.post_embedding_files = [v[1] for _, v in paired_embed_items]

        # Build label and mask file mapping based on shared chip index
        all_label_files = [p for p in self.label_dir.glob("*FLOOD*.npy")]
        all_mask_files = [p for p in self.label_dir.glob("*EXCLAYER*.npy")]
        all_water_files = [p for p in self.label_dir.glob("*REFERENCE_WATER*.npy")]

        def match_by_index(src_name: str, candidates):
            chip_idx = self._index_from_name(src_name)
            matches = [p for p in candidates if chip_idx in p.name]
            if not matches:
                raise FileNotFoundError(f"No match for {src_name} using key '{chip_idx}'")
            return str(matches[0].name)

        # Align labels and masks to pre embeddings
        self.labels = [match_by_index(name, all_label_files) for name in self.pre_embedding_files]
        self.masks = [match_by_index(name, all_mask_files) for name in self.pre_embedding_files]
        self.water_refs = [match_by_index(name, all_water_files) for name in self.pre_embedding_files]

        # Pair pre/post chip files; try to find two files per chip index
        all_chip_files = [p.name for p in self.chips_dir.glob("*.tif")]
        index_to_chipfiles = {}
        for fname in all_chip_files:
            idx = self._index_from_name(fname)
            index_to_chipfiles.setdefault(idx, []).append(fname)
        # For each embedding pair (ordered), fetch two corresponding chip files
        self.pre_chip_files = []
        self.post_chip_files = []
        for emb_name in self.pre_embedding_files:
            idx = self._index_from_name(emb_name)
            cand = sorted(index_to_chipfiles.get(idx, []))
            if len(cand) < 2:
                raise FileNotFoundError(f"Expected 2 chip files for chip '{idx}', found {len(cand)}")
            # Heuristics: choose first as pre, second as post
            self.pre_chip_files.append(cand[0])
            self.post_chip_files.append(cand[1])

        # Basic checks
        n = len(self.pre_embedding_files)
        assert n > 0, f"No embedding pairs found in directory {embeddings_dir}"
        assert (
            len(self.post_embedding_files) == n
            and len(self.labels) == n
            and len(self.masks) == n
            and len(self.water_refs) == n
            and len(self.pre_chip_files) == n
            and len(self.post_chip_files) == n
        ), "Mismatch between files counts"

        # Optional sub-sampling
        if max_samples is not None:
            self.pre_embedding_files = self.pre_embedding_files[:max_samples]
            self.post_embedding_files = self.post_embedding_files[:max_samples]
            self.labels = self.labels[:max_samples]
            self.masks = self.masks[:max_samples]
            self.water_refs = self.water_refs[:max_samples]
            self.pre_chip_files = self.pre_chip_files[:max_samples]
            self.post_chip_files = self.post_chip_files[:max_samples]

    def __len__(self):
        return len(self.pre_embedding_files)

    def _key_from_name(self, src_name: str) -> str:
        # Use everything after 'chip' token if present; else stem
        name = Path(src_name).name
        return name.split("chip", 1)[1] if "chip" in name else Path(name).stem

    def _index_from_name(self, src_name: str) -> str:
        name = Path(src_name).name
        if "chip_" in name:
            # token immediately after 'chip_'
            token = name.split("chip_", 1)[1]
            for char in token:
                if char.isnumeric(): 
                    continue
                else:
                    return token.split(char, 1)[0]
        # Fallback: stem
        return Path(name).stem

    def __getitem__(self, idx):
        pre_embedding_name = self.embeddings_dir / self.pre_embedding_files[idx]
        post_embedding_name = self.embeddings_dir / self.post_embedding_files[idx]
        label_name = self.label_dir / self.labels[idx]
        mask_name = self.label_dir / self.masks[idx]
        water_ref_name = self.label_dir / self.water_refs[idx]
        pre_chip_name = self.chips_dir / self.pre_chip_files[idx]
        post_chip_name = self.chips_dir / self.post_chip_files[idx]

        # Load arrays
        pre_embedding = np.load(pre_embedding_name).astype(np.float32)
        post_embedding = np.load(post_embedding_name).astype(np.float32)
        
        # Load GeoTIFF chips with metadata
        with rasterio.open(pre_chip_name) as src:
            pre_img = src.read().astype(np.float32)
        
        with rasterio.open(post_chip_name) as src:
            post_img = src.read().astype(np.float32)
        
        # Load label numpy files
        label = np.load(label_name).astype(bool).astype(np.uint8)
        excl_mask = np.load(mask_name).astype(bool)
        water_ref = np.load(water_ref_name) == 1
        
        ignore_mask = excl_mask | (water_ref)

        # To tensors
        pre_embedding_t = torch.from_numpy(pre_embedding)
        post_embedding_t = torch.from_numpy(post_embedding)
        pre_img_t = torch.from_numpy(pre_img)
        post_img_t = torch.from_numpy(post_img)

        # Normalize SAR chips if transform is provided
        if self.transform is not None:
            pre_img_t = self.transform(pre_img_t)
            post_img_t = self.transform(post_img_t)

        sample = {
            "pre_embedding": pre_embedding_t,
            "post_embedding": post_embedding_t,
            "pre_image": pre_img_t,
            "post_image": post_img_t,
            "label": torch.from_numpy(label),
            "ignore_mask": torch.from_numpy(ignore_mask),
            "pre_embedding_name": self.pre_embedding_files[idx],
        }
        return sample


class EmbeddingDataModuleGFM3(L.LightningDataModule):
    """
    Lightning DataModule for pre-computed embeddings paired with SAR chips.
    Produces batches compatible with EmbeddingClassifierGFM v3.
    """

    def __init__(
        self,
        train_embedd_dir,
        train_label_dir,
        val_embedd_dir,
        val_label_dir,
        train_chips_dir=None,
        val_chips_dir=None,
        test_embedd_dir=None,
        test_label_dir=None,
        test_chips_dir=None,
        metadata_path: Optional[str] = None,
        platform: Optional[str] = None,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 16,
        num_workers: int = 8,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.train_embedd_dir = train_embedd_dir
        self.train_label_dir = train_label_dir    
        self.train_chips_dir = train_chips_dir
        self.val_embedd_dir = val_embedd_dir
        self.val_label_dir = val_label_dir
        self.val_chips_dir = val_chips_dir
        self.test_embedd_dir = test_embedd_dir
        self.test_label_dir = test_label_dir
        self.test_chips_dir = test_chips_dir
        self.metadata_path = metadata_path
        self.platform = platform
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples

    def setup(self, stage: Optional[str] = None):
        if stage in {"fit", None}:
            self.trn_ds = EmbeddingDatasetGFM3(
                self.train_embedd_dir,
                self.train_label_dir,
                self.train_chips_dir,
                self.metadata_path,
                self.platform,
                self.target_size,
                self.max_samples,
            )
            self.val_ds = EmbeddingDatasetGFM3(
                self.val_embedd_dir,
                self.val_label_dir,
                self.val_chips_dir,
                self.metadata_path,
                self.platform,
                self.target_size,
                self.max_samples,
            )
        elif stage == "test":
            if (
                self.test_embedd_dir is None
                or self.test_label_dir is None
            ):
                raise ValueError("Test directories must be provided for test stage")
            self.test_ds = EmbeddingDatasetGFM3(
                self.test_embedd_dir,
                self.test_label_dir,
                self.test_chips_dir,
                self.metadata_path,
                self.platform,
                self.target_size,
                self.max_samples,
            )
        else:
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )



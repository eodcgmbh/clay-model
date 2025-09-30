
import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from typing import Optional, Tuple

class EmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed Clay embeddings and corresponding GFM labels
    and exclusion masks (ignore regions).
    """

    def __init__(
            self,
            embeddings_dir: str,
            label_dir: str,
            target_size: Tuple[int, int] = (224, 224)
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.label_dir = Path(label_dir)
        self.target_size = target_size
        
        # Find embedding files
        self.embedding_files = [embedding_path.name for embedding_path in self.embeddings_dir.glob("*.npy")]

        # Build label and mask file mapping based on shared chip index
        all_label_files = [p for p in self.label_dir.glob("*FLOOD*.npy")]
        all_mask_files = [p for p in self.label_dir.glob("*EXCLAYER*.npy")]

        def match_by_index(src_name: str, candidates):
            if "chip" in src_name:
                chip_idx = src_name.split("chip", 1)[1]
            else:
                chip_idx = Path(src_name).stem
            matches = [p for p in candidates if chip_idx in p.name]
            if not matches:
                raise FileNotFoundError(f"No match for {src_name} using key '{chip_idx}' in {[p.name for p in candidates[:3]]}...")
            return str(matches[0].name)

        self.labels = [match_by_index(name, all_label_files) for name in self.embedding_files]
        self.masks = [match_by_index(name, all_mask_files) for name in self.embedding_files]

        # Basic checks
        assert len(self.embedding_files) > 0, f"No embedding files found in directory {embeddings_dir}"
        assert len(self.embedding_files) == len(self.labels) == len(self.masks), \
            "Mismatch between embeddings, labels, and masks counts"
    
    def __len__(self):
        return len(self.embedding_files)
    
    def __getitem__(self, idx):

        embedding_name = self.embeddings_dir / self.embedding_files[idx]
        label_name = self.label_dir / self.labels[idx]
        mask_name = self.label_dir / self.masks[idx]

        embedding = np.load(embedding_name).astype(np.float32)
        label = np.load(label_name)
        excl_mask = np.load(mask_name)
        
        sample = {
            "embedding": torch.from_numpy(embedding),
            "label": torch.from_numpy(label),
            "ignore_mask": torch.from_numpy(excl_mask),
            "embedding_name": self.embedding_files[idx]
        }
        return sample

class EmbeddingDataModule(L.LightningDataModule):
    """
    Lightning DataModule for pre-computed embeddings
    """
    
    def __init__(
            self,
            train_embedd_dir,
            train_label_dir,
            val_embedd_dir,
            val_label_dir,
            test_embedd_dir=None,
            test_label_dir=None,
            target_size: Tuple[int, int] = (224, 224),
            batch_size: int = 16,
            num_workers: int = 8
    ):
        """
        Args:
            embeddings_dir: Directory containing saved embeddings
            label_dir: Directory containing labels
            target_size: Target size for labels (original image size)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            train_split: Fraction of data to use for training
            val_split: Fraction of data to use for validation
        """
        super().__init__()        
        self.train_embedd_dir = train_embedd_dir
        self.train_label_dir = train_label_dir
        self.val_embedd_dir = val_embedd_dir
        self.val_label_dir = val_label_dir
        self.test_embedd_dir = test_embedd_dir
        self.test_label_dir = test_label_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets"""
        
        if stage in {"fit", None}:
            self.trn_ds = EmbeddingDataset(
                self.train_embedd_dir,
                self.train_label_dir,
                self.target_size,
            )
            self.val_ds = EmbeddingDataset(
                self.val_embedd_dir,
                self.val_label_dir,
                self.target_size,
            )
        elif stage == "test":
            if self.test_embedd_dir is None or self.test_label_dir is None:
                raise ValueError("Test directories must be provided for test stage")
            self.test_ds = EmbeddingDataset(
                self.test_embedd_dir,
                self.test_label_dir,
                self.target_size,
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
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
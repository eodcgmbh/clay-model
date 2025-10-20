from typing import Optional, Tuple

import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rasterio as rio


class EmbeddingDatasetAI4B(Dataset):
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
        
        # Find and pair pre/post embedding files within a single directory
        all_embed_files = [p.name for p in self.embeddings_dir.glob("*.npy")]
        # key_to_files = {}
        # for fname in all_embed_files:
        #     key = fname.split("_", 3)[1]
        #     key_to_files.setdefault(key, []).append(fname)
        # paired_items = [(k, sorted(v)) for k, v in key_to_files.items() if len(v) == 2]
        # paired_items.sort(key=lambda x: x[0])
        # self.pre_embedding_files = [v[0] for _, v in paired_items]
        # self.post_embedding_files = [v[1] for _, v in paired_items]
        self.embeddings_files = sorted(all_embed_files)

        def key_from_name(src_name: str) -> str:
            
            return src_name.split("_", 3)[1]

        def match_by_index(src_name: str, candidates):
            chip_idx = key_from_name(src_name)
            matches = [p for p in candidates if chip_idx in p.name]
            if not matches:
                raise FileNotFoundError(f"No match for {src_name} using key '{chip_idx}'")
            return str(matches[0].name)

        all_label_files = [p for p in self.label_dir.glob("*.tif")]
        self.labels = [match_by_index(name, all_label_files) for name in self.embeddings_files]

        # Basic checks
        assert len(self.embeddings_files) > 0, f"No embedding pairs found in directory {embeddings_dir}"

    
    def __len__(self):
        return len(self.embeddings_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            dict: Sample containing pre/post embeddings, labels, and ignore mask
        """
        embedding_name = self.embeddings_dir / self.embeddings_files[idx]
        
        label_name = self.label_dir / self.labels[idx]

        embedding = np.load(embedding_name).astype(np.float32)

        with rio.open(label_name) as src:
            label = src.read(
                out_shape=(
                    src.count,
                    self.target_size[0],
                    self.target_size[1]
                ),
                resampling=rio.enums.Resampling.nearest
            )[:2,...].astype(np.uint8) # Keep only first two channels (boundary and field)
        
        sample = {
            "embedding": torch.from_numpy(embedding),
            "label": torch.from_numpy(label),
            "embedding_name": self.embeddings_files[idx]
        }
        return sample

class EmbeddingDataModuleAI4B(L.LightningDataModule):
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
            train_embedd_dir: Directory containing training embeddings
            train_label_dir: Directory containing training labels
            val_embedd_dir: Directory containing validation embeddings
            val_label_dir: Directory containing validation labels
            test_embedd_dir: Directory containing test embeddings
            test_label_dir: Directory containing test labels
            target_size: Target size for labels (original image size)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading (for CPU only)
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
        """
        Set up datasets.
        
        Args:
            stage: Stage identifier ('fit', 'test', or None)
        """
        
        if stage in {"fit", None}:
            self.trn_ds = EmbeddingDatasetAI4B(
                self.train_embedd_dir,
                self.train_label_dir,
                self.target_size,
            )
            self.val_ds = EmbeddingDatasetAI4B(
                self.val_embedd_dir,
                self.val_label_dir,
                self.target_size,
            )
        elif stage == "test":
            if self.test_embedd_dir is None or self.test_label_dir is None:
                raise ValueError("Test directories must be provided for test stage")
            self.test_ds = EmbeddingDatasetAI4B(
                self.test_embedd_dir,
                self.test_label_dir,
                self.target_size,
            )
        else:
            raise NotImplementedError()
    
    def train_dataloader(self):
        """Create DataLoader for training data."""
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Create DataLoader for validation data."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Create DataLoader for test data."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True
        )
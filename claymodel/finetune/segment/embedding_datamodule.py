
import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from typing import Optional, Tuple

class EmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed Clay embeddings and corresponding labels
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
        
        # Find matching embedding and label files        
        self.embedding_files = [embedding_path.name for embedding_path in self.embeddings_dir.glob("*.npy")]
        
        self.labels = [chip.replace("_naip-new_", "_lc_").replace("_emb", "") for chip in self.embedding_files]
        
        # Verify matching files
        assert len(self.embedding_files) == len(self.labels), \
            f"Mismatch: {len(self.embedding_files)} embedding files vs {len(self.labels)} label files"

        assert len(self.embedding_files) > 0, "No embedding files found in directory {embeddings_dir}"        
    
    def __len__(self):
        return len(self.embedding_files)
    
    def __getitem__(self, idx):

        embedding_name = self.embeddings_dir / self.embedding_files[idx]
        label_name = self.label_dir / self.labels[idx]

        embedding = np.load(embedding_name).astype(np.float32)
        label = np.load(label_name)

        # Remap labels to match desired classes
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: -1} # 15 is no data and should be ignored
        remapped_label = np.vectorize(label_mapping.get)(label)
        
        sample = {
            "embedding": torch.from_numpy(embedding),
            "label": torch.from_numpy(remapped_label[0]),            
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
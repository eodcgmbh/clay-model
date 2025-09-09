"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class ChesapeakeDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, chip_dir, label_dir, metadata, platform, max_samples):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        self.gsd = torch.tensor(metadata[platform].gsd)
        self.waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))

        # Load chip and label file names
        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npy")]
        self.chips.sort()

        if max_samples is not None:
            rng = np.random.RandomState(42)
            rng.shuffle(self.chips)
            self.chips = self.chips[:max_samples]           
        
        self.labels = [re.sub("_naip-new_", "_lc_", chip) for chip in self.chips]

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ],
        )

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip_name = self.chip_dir / self.chips[idx]
        label_name = self.label_dir / self.labels[idx]

        chip = np.load(chip_name).astype(np.float32)
        label = np.load(label_name)

        # Remap labels to match desired classes
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: -1}
        remapped_label = np.vectorize(label_mapping.get)(label)

        sample = {
            "pixels": self.transform(torch.from_numpy(chip)),
            "label": torch.from_numpy(remapped_label[0]),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
            "waves": self.waves,
            "gsd" : self.gsd,
            "chip_name": self.chips[idx],
            "chip_dir": str(self.chip_dir)
        }
        return sample


class ChesapeakeDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        parent_data_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
        max_samples,
        predict_ds="train", # "train", "val" or "test"
    ):
        super().__init__()
        self.train_chip_dir = Path(parent_data_dir) / "train" / "chips"        
        self.train_label_dir = Path(parent_data_dir) / "train" / "labels"
        self.val_chip_dir = Path(parent_data_dir) / "val" / "chips"
        self.val_label_dir = Path(parent_data_dir) / "val" / "labels"
        self.test_chip_dir = Path(parent_data_dir) / "test" / "chips"
        self.test_label_dir = Path(parent_data_dir) / "test" / "labels"
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform
        self.max_samples = max_samples
        self.predict_ds = predict_ds

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('predict').
        """
        if stage in {"predict", None}:
            if self.predict_ds=="val":
                self.predict_chip_dir = self.val_chip_dir
                self.predict_label_dir = self.val_label_dir
            elif self.predict_ds=="test":
                self.predict_chip_dir = self.test_chip_dir
                self.predict_label_dir = self.test_label_dir
            else: # if self.predict_ds=="train":
                self.predict_chip_dir = self.train_chip_dir
                self.predict_label_dir = self.train_label_dir

            self.prd_ds = ChesapeakeDataset(
                self.predict_chip_dir,
                self.predict_label_dir,
                self.metadata,
                self.platform,
                self.max_samples
            )
    
    def predict_dataloader(self):
        """
        Create DataLoader for prediction data. For now it is train eventually should be changed

        Returns:
            DataLoader: DataLoader for prediction dataset.
        """
        return DataLoader(
            self.prd_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

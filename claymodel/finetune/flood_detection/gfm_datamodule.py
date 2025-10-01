"""
DataModule for the GFM flood detection two-stream segmentation tasks.

This implementation handles paired pre- and post-event chips and corresponding
labels for flood segmentation. Pixels are exposed as two streams
(`pre_pixels`, `post_pixels`).
"""

from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class GFMDataset(Dataset):
    """
    Dataset class for the GFM flood segmentation dataset with two streams
    (pre-event and post-event).

    Args:
        chip_dir (str): Directory containing the image chips.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, chips_dir, metadata, platform):
        self.chips_dir = Path(chips_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        self.gsd = torch.tensor(metadata[platform].gsd)
        self.waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))

        self.chips = [p for p in self.chips_dir.glob("*.npy")]

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
        chip = np.load(self.chips[idx]).astype(np.float32)

        sample = {            
            "pixels": self.transform(torch.from_numpy(chip)),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
            "waves": self.waves,
            "gsd" : self.gsd,
            "chip_name": str(self.chips[idx].name),
            "chip_dir": str(self.chips[idx].parent),
        }
        return sample


class GFMDataModule(L.LightningDataModule):
    """
    DataModule class for the GFM flood dataset with two-stream chips.

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
        train_split_name="train",
        val_split_name="val",
        test_split_name="test",
    ):
        super().__init__()
        self.parent_data_dir = Path(parent_data_dir)
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name
        self.test_split_name = test_split_name
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform
        self.max_samples = max_samples
        self.predict_ds = predict_ds

    # def _label_dir(self, split_name: str) -> Path:
    #     return self.parent_data_dir / split_name / "labels"

    def _chips_dir(self, split_name: str) -> Path:
        return self.parent_data_dir / split_name / "chips"

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('predict').
        """
        if stage in {"predict", None}:
            if self.predict_ds=="val":
                split_name = self.val_split_name
            elif self.predict_ds=="test":
                split_name = self.test_split_name
            else: # if self.predict_ds=="train":
                split_name = self.train_split_name

            split_dir = self.parent_data_dir / split_name
            chips_dir = self._chips_dir(split_name)

            self.prd_ds = GFMDataset(
                chips_dir,
                self.metadata,
                self.platform,
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

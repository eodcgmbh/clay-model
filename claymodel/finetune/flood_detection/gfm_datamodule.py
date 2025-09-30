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
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, pre_event_dir, post_event_dir, label_dir, metadata, platform, max_samples):
        self.pre_chip_dir = Path(pre_event_dir)
        self.post_chip_dir = Path(post_event_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        self.gsd = torch.tensor(metadata[platform].gsd)
        self.waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))

        # Load chip and label file names
        self.pre_chips = [chip_path.name for chip_path in self.pre_chip_dir.glob("*.npy")]
        self.pre_chips.sort()
        self.post_chips = [chip_path.name for chip_path in self.post_chip_dir.glob("*.npy")]
        self.post_chips.sort()
        assert len(self.pre_chips) == len(self.post_chips), "Number of pre-event and post-event chips must be the same"

        if max_samples is not None and max_samples > 0 and max_samples < len(self.pre_chips):
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(self.pre_chips))[:max_samples]
            self.pre_chips = [self.pre_chips[i] for i in perm]
            self.post_chips = [self.post_chips[i] for i in perm]
                
        # all_label_files = [p for p in self.label_dir.glob("*FLOOD*.npy")]
        # all_mask_files = [p for p in self.label_dir.glob("*EXCLAYER*.npy")]
        # chip_indeces = [chip.split("chip")[1] for chip in self.pre_chips]
        # self.labels = [str(list(filter(lambda x: chip_idx in str(x), all_label_files))[0].name) for chip_idx in chip_indeces]
        # self.masks = [str(list(filter(lambda x: chip_idx in str(x), all_mask_files))[0].name) for chip_idx in chip_indeces]

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
        return len(self.pre_chips) + len(self.post_chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        if idx >= len(self.pre_chips):
            chip_name = self.post_chip_dir / self.post_chips[idx - len(self.pre_chips)]
        else:
            chip_name = self.pre_chip_dir  / self.pre_chips[idx]
        # pre_chip_name = self.pre_chip_dir / self.pre_chips[idx]
        # post_chip_name = self.post_chip_dir / self.post_chips[idx]
        # label_name = self.label_dir / self.labels[idx]
        # mask_name = self.label_dir / self.masks[idx]

        # pre_chip = np.load(pre_chip_name).astype(np.float32)
        # post_chip = np.load(post_chip_name).astype(np.float32)
        # label = np.load(label_name)
        # excl_mask = np.load(mask_name)
        chip = np.load(chip_name).astype(np.float32)

        sample = {
            # "pre_pixels": self.transform(torch.from_numpy(pre_chip)),
            # "post_pixels": self.transform(torch.from_numpy(post_chip)),
            "pixels": self.transform(torch.from_numpy(chip)),
            # "label": torch.from_numpy(label),
            # "ignore_mask": torch.from_numpy(excl_mask),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
            "waves": self.waves,
            "gsd" : self.gsd,
            # "pre_chip_name": self.pre_chips[idx],
            # "pre_chip_dir": str(self.pre_chip_dir),
            "chip_name": chip_name.name,
            "chip_dir": str(chip_name.parent),
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

    def _label_dir(self, split_name: str) -> Path:
        return self.parent_data_dir / split_name / "labels"

    def _find_pre_post_dirs(self, split_dir: Path):
        """Discover the two chips_SIG0_* directories and return (pre_dir, post_dir).

        Assumes exactly two subdirectories starting with 'chips_SIG0_'.
        The order is determined by sorted name (first is pre, second is post).
        """
        candidates = [p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("chips_SIG0_")]
        assert len(candidates) == 2, f"Expected 2 'chips_SIG0_*' dirs under {split_dir}, found {len(candidates)}"
        candidates.sort(key=lambda p: p.name)
        return candidates[0], candidates[1]

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
            predict_label_dir = self._label_dir(split_name)

            pre_dir, post_dir = self._find_pre_post_dirs(split_dir)

            self.prd_ds = GFMDataset(
                pre_dir,
                post_dir,
                predict_label_dir,
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

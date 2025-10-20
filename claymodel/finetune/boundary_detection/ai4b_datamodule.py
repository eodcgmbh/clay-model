"""
DataModule for the GFM flood detection two-stream segmentation tasks.

This implementation handles paired pre- and post-event chips and corresponding
labels for flood segmentation. Pixels are exposed as two streams
(`pre_pixels`, `post_pixels`).
"""

from pathlib import Path
import math

from datetime import datetime
import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import rasterio as rio
import xarray as xr
import pandas as pd

def coords_to_bounds(coords) -> rio.coords.BoundingBox:
    """
    Convert xarray coordinates to rasterio bounds.

    Args:
        coords: xarray coordinates object

    Returns:
        rasterio.coords.BoundingBox: Bounding box with (left, bottom, right, top)
    """
    x = coords['x'].values
    y = coords['y'].values
    left = float(np.min(x))
    right = float(np.max(x))
    bottom = float(np.min(y))
    top = float(np.max(y))
    return rio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)

def normalize_latlon(lat, lon):
    """
    Normalize latitude and longitude coordinates using sinusoidal encoding.
    This helps neural networks better handle the circular nature of coordinates.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        tuple: (sin(lat), cos(lat), sin(lon), cos(lon)) - normalized coordinates
    """
    lat = lat * np.pi / 180  # Convert to radians
    lon = lon * np.pi / 180  # Convert to radians

    return (math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon))

def bounds_to_latlon_tensor(bounds):
    """
    Convert rasterio bounds to a normalized tensor format.

    Args:
        bounds: rasterio.bounds.BoundingBox object (left, bottom, right, top)

    Returns:
        torch.Tensor: [sin(lat), cos(lat), sin(lon), cos(lon)] - normalized coordinates
    """
    # Bounds are in order: left, bottom, right, top
    left, bottom, right, top = bounds
    lat = (bottom + top) / 2  # Center latitude
    lon = (left + right) / 2  # Center longitude
    lat_lon_norm = normalize_latlon(lat, lon)
    return torch.tensor(lat_lon_norm, dtype=torch.float32)


def normalize_timestamp(date):
    """
    Normalize timestamp using sinusoidal encoding for cyclical features.
    This helps neural networks better handle the cyclical nature of time.

    Args:
        date: datetime object

    Returns:
        tuple: (sin(week), cos(week), sin(hour), cos(hour)) - normalized time features
    """
    week = date.isocalendar().week * 2 * np.pi / 52  # Normalize week to [0, 2π]
    hour = date.hour * 2 * np.pi / 24  # Normalize hour to [0, 2π]

    return (math.sin(week), math.cos(week), math.sin(hour), math.cos(hour))

def date_to_tensor(times):
    """
    Convert array of datetimes to normalized tensor format using sinusoidal encoding.

    Args:
        times: array-like of datetime objects

    Returns:
        torch.Tensor: [sin(week), cos(week), sin(hour), cos(hour)] - normalized time features
    """
    time_features = [normalize_timestamp(pd.to_datetime(t)) for t in times]
    return torch.tensor(time_features, dtype=torch.float32)



class AI4BDataset(Dataset):
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
        # self.labels_dir = Path(labels_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        self.gsd = torch.tensor(metadata[platform].gsd)
        self.waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))
        self.chips = [p for p in self.chips_dir.glob("*.nc")]
        # self.labels = [p for p in self.labels_dir.glob("*.tif")]

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
        ds = xr.open_dataset(self.chips[idx], engine='netcdf4')
        # dims = np.array(list(ds.dims.values()))
        
        times = ds.time.values
        coords = ds.coords

        img_np = np.array([ds[d].values for d in ["B2", "B3", "B4", "B8"]])
        img_np = np.swapaxes(img_np, 0, 1)  # Shape: (T, C, H, W)

        chip_latlon = bounds_to_latlon_tensor(coords_to_bounds(coords))
        chip_times = date_to_tensor(times)

        sample = {
            "pixels": self.transform(torch.from_numpy(img_np)),  # Shape: (C, H, W)
            "times": chip_times,
            "latlon": chip_latlon,
            "waves": self.waves,
            "gsd" : self.gsd,
            "chip_name": str(self.chips[idx].name),
            "chip_dir": str(self.chips[idx].parent),
        }
        ds.close()

        return sample


class AI4BDataModule(L.LightningDataModule):
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
        train_split_name="FR_train",
        val_split_name="FR_val",
        test_split_name="ES",
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
    #     return self.parent_data_dir / split_name / "masks"

    def _chips_dir(self, split_name: str) -> Path:
        return self.parent_data_dir / split_name / "images"

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
            # masks_dir = self._label_dir(split_name)


            self.prd_ds = AI4BDataset(
                chips_dir,
                # masks_dir,
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

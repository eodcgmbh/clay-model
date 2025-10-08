"""
DataModule for the GFM flood detection U-Net model.

This implementation handles paired pre- and post-event chips and corresponding
labels for flood segmentation using U-Net architecture.
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
from typing import Optional, Tuple


def extract_geospatial_metadata(geotiff_path):
    """
    Extract geospatial metadata from a GeoTIFF file.
    
    Args:
        geotiff_path: Path to the GeoTIFF file
        
    Returns:
        dict: Dictionary containing bounds, transform, crs, and acquisition date
    """
    with rio.open(geotiff_path) as src:
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        
        # Extract acquisition date from tags if available
        acquisition_date = None
        if hasattr(src, 'tags') and 'ACQUISITION_DATE' in src.tags():
            acquisition_date = src.tags()['ACQUISITION_DATE']
        
        return {
            'bounds': bounds,
            'transform': transform,
            'crs': crs,
            'acquisition_date': acquisition_date
        }


def normalize_latlon(lat, lon):
    """
    Normalize latitude and longitude to [-1, 1] range.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        tuple: Normalized (lat, lon) values
    """
    # Simple normalization - you might want to adjust based on your data
    norm_lat = lat / 90.0  # Assuming lat is in [-90, 90]
    norm_lon = lon / 180.0  # Assuming lon is in [-180, 180]
    return norm_lat, norm_lon


def bounds_to_latlon_tensor(bounds):
    """
    Convert bounds to normalized lat/lon tensor.
    
    Args:
        bounds: Bounding box (left, bottom, right, top)
        
    Returns:
        torch.Tensor: 4-element tensor with normalized coordinates
    """
    left, bottom, right, top = bounds
    center_lat = (bottom + top) / 2
    center_lon = (left + right) / 2
    
    norm_lat, norm_lon = normalize_latlon(center_lat, center_lon)
    
    return torch.tensor([norm_lat, norm_lon, norm_lat, norm_lon], dtype=torch.float32)


def date_to_tensor(date_str):
    """
    Convert date string to tensor representation.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        torch.Tensor: 4-element tensor with date components
    """
    try:
        if date_str:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year / 3000.0  # Normalize year
            month = date_obj.month / 12.0  # Normalize month
            day = date_obj.day / 31.0  # Normalize day
            day_of_year = date_obj.timetuple().tm_yday / 365.0  # Normalize day of year
            
            return torch.tensor([year, month, day, day_of_year], dtype=torch.float32)
    except ValueError:
        pass
    
    return torch.zeros(4, dtype=torch.float32)


class GFMUNetDataset(Dataset):
    """
    Dataset class for the GFM flood segmentation dataset with two streams
    (pre-event and post-event) and labels.

    Args:
        chips_dir (str): Directory containing the image chips.
        labels_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
        file_extension (str): File extension for chips ('npy' or 'tif').
    """

    def __init__(self, chips_dir, labels_dir, metadata, platform, file_extension="tif"):
        self.chips_dir = Path(chips_dir)
        self.labels_dir = Path(labels_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        self.gsd = torch.tensor(metadata[platform].gsd)
        self.waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))
        self.file_extension = file_extension
        
        # Get all chip files
        self.chips = [p for p in self.chips_dir.glob("*." + self.file_extension)]
        
        # Pair pre/post chips and match with labels
        self._pair_chips_and_labels()

    def _pair_chips_and_labels(self):
        """Pair pre/post chips and match with corresponding labels."""
        # Group chips by their base name (without pre/post suffix)
        chip_groups = {}
        for chip_path in self.chips:
            name = chip_path.stem
            # Extract base name by removing pre/post indicators
            if "_pre_" in name or "_post_" in name:
                base_name = name.replace("_pre_", "_").replace("_post_", "_")
            elif "pre" in name.lower() and "post" in name.lower():
                # Handle different naming conventions
                parts = name.split("_")
                base_parts = []
                for part in parts:
                    if part.lower() in ["pre", "post"]:
                        break
                    base_parts.append(part)
                base_name = "_".join(base_parts)
            else:
                # Fallback: use chip index
                if "chip_" in name:
                    base_name = name.split("chip_")[1].split("_")[0]
                else:
                    base_name = name
            
            if base_name not in chip_groups:
                chip_groups[base_name] = []
            chip_groups[base_name].append(chip_path)
        
        # Create pairs and match with labels
        self.chip_pairs = []
        self.label_files = []
        self.mask_files = []
        self.water_files = []
        
        for base_name, chip_paths in chip_groups.items():
            if len(chip_paths) >= 2:
                # Sort to ensure consistent ordering
                chip_paths.sort()
                pre_chip = chip_paths[0]
                post_chip = chip_paths[1]
                
                # Find corresponding label files
                label_file = self._find_label_file(base_name, "FLOOD")
                mask_file = self._find_label_file(base_name, "EXCLAYER")
                water_file = self._find_label_file(base_name, "REFERENCE_WATER")
                
                if label_file and mask_file and water_file:
                    self.chip_pairs.append((pre_chip, post_chip))
                    self.label_files.append(label_file)
                    self.mask_files.append(mask_file)
                    self.water_files.append(water_file)

    def _find_label_file(self, base_name, label_type):
        """Find label file matching the base name and type."""
        label_files = list(self.labels_dir.glob(f"*{label_type}*.npy"))
        for label_file in label_files:
            if base_name in label_file.name or any(part in label_file.name for part in base_name.split("_")):
                return label_file
        return None

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose([
            v2.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.chip_pairs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the images, labels, and additional information.
        """
        pre_chip_path, post_chip_path = self.chip_pairs[idx]
        label_path = self.label_files[idx]
        mask_path = self.mask_files[idx]
        water_path = self.water_files[idx]

        if self.file_extension == "npy":
            pre_chip = np.load(pre_chip_path).astype(np.float32)
            post_chip = np.load(post_chip_path).astype(np.float32)
            
            sample = {
                "pre_image": self.transform(torch.from_numpy(pre_chip)),
                "post_image": self.transform(torch.from_numpy(post_chip)),
                "time": torch.zeros(4),  # Placeholder for time information
                "latlon": torch.zeros(4),  # Placeholder for latlon information
                "waves": self.waves,
                "gsd": self.gsd,
                "chip_name": str(pre_chip_path.name),
                "chip_dir": str(pre_chip_path.parent),
            }
        elif self.file_extension == "tif":
            with rio.open(pre_chip_path) as src:
                pre_chip = src.read().astype(np.float32)
                pre_metadata = extract_geospatial_metadata(pre_chip_path)
            
            with rio.open(post_chip_path) as src:
                post_chip = src.read().astype(np.float32)
                post_metadata = extract_geospatial_metadata(post_chip_path)

            pre_latlon = bounds_to_latlon_tensor(pre_metadata['bounds'])
            pre_time = date_to_tensor(pre_metadata['acquisition_date'])
            post_latlon = bounds_to_latlon_tensor(post_metadata['bounds'])
            post_time = date_to_tensor(post_metadata['acquisition_date'])

            sample = {
                "pre_image": self.transform(torch.from_numpy(pre_chip)),
                "post_image": self.transform(torch.from_numpy(post_chip)),
                "pre_time": pre_time,
                "post_time": post_time,
                "pre_latlon": pre_latlon,
                "post_latlon": post_latlon,
                "waves": self.waves,
                "gsd": self.gsd,
                "chip_name": str(pre_chip_path.name),
                "chip_dir": str(pre_chip_path.parent),
            }

        # Load labels and masks
        label = np.load(label_path).astype(bool).astype(np.uint8)  # Binary mask: 1 for flood, 0 for no flood
        excl_mask = np.load(mask_path).astype(bool)
        water_ref = np.load(water_path) == 1  # Water reference is 1 for permanent water
        ignore_mask = excl_mask | (water_ref)  # Exclude both exclusion areas and water areas

        sample.update({
            "label": torch.from_numpy(label),
            "ignore_mask": torch.from_numpy(ignore_mask),
        })

        return sample


class GFMUNetDataModule(L.LightningDataModule):
    """
    DataModule class for the GFM flood dataset with U-Net model.

    Args:
        parent_data_dir (str): Parent directory containing train/val/test splits.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
        max_samples (int): Maximum number of samples to use.
        train_split_name (str): Name of training split directory.
        val_split_name (str): Name of validation split directory.
        test_split_name (str): Name of test split directory.
        file_extension (str): File extension for chips ('npy' or 'tif').
    """

    def __init__(
        self,
        parent_data_dir,
        metadata_path,
        batch_size,
        num_workers,
        max_samples,
        platform,
        train_split_name="train",
        val_split_name="val",
        test_split_name="test",
        file_extension="tif",
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
        self.file_extension = file_extension

    def _chips_dir(self, split_name: str) -> Path:
        return self.parent_data_dir / split_name / "chips"

    def _labels_dir(self, split_name: str) -> Path:
        return self.parent_data_dir / split_name / "labels"

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.

        Args:
            stage (str): Stage identifier ('fit', 'test', or None).
        """
        if stage in {"fit", None}:
            train_chips_dir = self._chips_dir(self.train_split_name)
            train_labels_dir = self._labels_dir(self.train_split_name)
            val_chips_dir = self._chips_dir(self.val_split_name)
            val_labels_dir = self._labels_dir(self.val_split_name)

            self.train_ds = GFMUNetDataset(
                train_chips_dir,
                train_labels_dir,
                self.metadata,
                self.platform,
                self.file_extension,
            )
            
            self.val_ds = GFMUNetDataset(
                val_chips_dir,
                val_labels_dir,
                self.metadata,
                self.platform,
                self.file_extension,
            )

            # Apply max_samples if specified
            if self.max_samples is not None:
                self.train_ds.chip_pairs = self.train_ds.chip_pairs[:self.max_samples]
                self.train_ds.label_files = self.train_ds.label_files[:self.max_samples]
                self.train_ds.mask_files = self.train_ds.mask_files[:self.max_samples]
                self.train_ds.water_files = self.train_ds.water_files[:self.max_samples]

        elif stage == "test":
            test_chips_dir = self._chips_dir(self.test_split_name)
            test_labels_dir = self._labels_dir(self.test_split_name)

            self.test_ds = GFMUNetDataset(
                test_chips_dir,
                test_labels_dir,
                self.metadata,
                self.platform,
                self.file_extension,
            )

    def train_dataloader(self):
        """Create DataLoader for training data."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create DataLoader for validation data."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create DataLoader for test data."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

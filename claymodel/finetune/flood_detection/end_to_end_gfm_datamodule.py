"""
DataModule for end-to-end flood detection with raw image chips.
Loads paired pre/post SAR images, labels, and masks for training.
"""

from pathlib import Path
from datetime import datetime
import math

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import rasterio as rio


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
    Normalize latitude and longitude coordinates using sinusoidal encoding.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        tuple: (sin(lat), cos(lat), sin(lon), cos(lon))
    """
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon))


def bounds_to_latlon_tensor(bounds):
    """
    Convert rasterio bounds to a normalized tensor format.
    
    Args:
        bounds: rasterio.bounds.BoundingBox object (left, bottom, right, top)
        
    Returns:
        torch.Tensor: [sin(lat), cos(lat), sin(lon), cos(lon)]
    """
    left, bottom, right, top = bounds
    lat = (bottom + top) / 2
    lon = (left + right) / 2
    lat_lon_norm = normalize_latlon(lat, lon)
    return torch.tensor(lat_lon_norm, dtype=torch.float32)


def normalize_timestamp(date):
    """
    Normalize timestamp using sinusoidal encoding for cyclical features.
    
    Args:
        date: datetime object
        
    Returns:
        tuple: (sin(week), cos(week), sin(hour), cos(hour))
    """
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week), math.sin(hour), math.cos(hour))


def date_to_tensor(date_str):
    """
    Convert date string to normalized tensor format.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        torch.Tensor: [sin(week), cos(week), sin(hour), cos(hour)]
    """
    if date_str is None:
        return torch.zeros(4, dtype=torch.float32)
    
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return torch.tensor(normalize_timestamp(date_obj), dtype=torch.float32)
    except ValueError:
        return torch.zeros(4, dtype=torch.float32)


class EndToEndGFMDataset(Dataset):
    """
    Dataset for end-to-end flood detection with paired pre/post raw images.
    
    Args:
        chips_dir: Directory containing image chips (.tif files)
        label_dir: Directory containing labels and masks (.npy files)
        metadata: Box object with normalization parameters
        platform: Platform identifier (e.g., 'sentinel-1-rtc')
        target_size: Target image size (H, W)
        max_samples: Optional limit on number of samples
    """
    
    def __init__(self, chips_dir, label_dir, metadata, platform, target_size=(224, 224), max_samples=None):
        self.chips_dir = Path(chips_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.platform = platform
        self.target_size = target_size
        
        # Create normalization transform
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )
        
        # Get metadata for Clay encoder
        self.gsd = torch.tensor(metadata[platform].gsd)
        self.waves = torch.tensor(list(metadata[platform].bands.wavelength.values()))
        
        # Find all chip files and pair them
        all_chip_files = [p.name for p in self.chips_dir.glob("*.tif")]
        
        # Group chips by index to pair pre/post
        index_to_chips = {}
        for fname in all_chip_files:
            idx = self._index_from_name(fname)
            index_to_chips.setdefault(idx, []).append(fname)
        
        # Keep only indices with exactly 2 chips (pre and post)
        paired_items = [(idx, sorted(chips)) for idx, chips in index_to_chips.items() if len(chips) == 2]
        paired_items.sort(key=lambda x: x[0])
        
        if len(paired_items) == 0:
            raise RuntimeError(f"No paired chips found in {chips_dir}")
        
        self.chip_indices = [idx for idx, _ in paired_items]
        self.pre_chip_files = [chips[0] for _, chips in paired_items]
        self.post_chip_files = [chips[1] for _, chips in paired_items]
        
        # Find corresponding labels and masks
        all_label_files = [p for p in self.label_dir.glob("*FLOOD*.npy")]
        all_mask_files = [p for p in self.label_dir.glob("*EXCLAYER*.npy")]
        all_water_files = [p for p in self.label_dir.glob("*REFERENCE_WATER*.npy")]
        
        def match_by_index(chip_idx, candidates):
            matches = [p for p in candidates if chip_idx in p.name]
            if not matches:
                raise FileNotFoundError(f"No match for chip index '{chip_idx}'")
            return str(matches[0].name)
        
        self.label_files = [match_by_index(idx, all_label_files) for idx in self.chip_indices]
        self.mask_files = [match_by_index(idx, all_mask_files) for idx in self.chip_indices]
        self.water_files = [match_by_index(idx, all_water_files) for idx in self.chip_indices]
        
        # Apply max_samples limit if specified
        if max_samples is not None:
            self.chip_indices = self.chip_indices[:max_samples]
            self.pre_chip_files = self.pre_chip_files[:max_samples]
            self.post_chip_files = self.post_chip_files[:max_samples]
            self.label_files = self.label_files[:max_samples]
            self.mask_files = self.mask_files[:max_samples]
            self.water_files = self.water_files[:max_samples]
        
        print(f"📊 Dataset initialized: {len(self)} paired samples from {chips_dir}")
    
    def _index_from_name(self, filename):
        """Extract chip index from filename."""
        name = Path(filename).name
        if "chip_" in name:
            token = name.split("chip_", 1)[1]
            for char in token:
                if not char.isnumeric():
                    return token.split(char, 1)[0]
        return Path(name).stem
    
    def create_transforms(self, mean, std):
        """Create normalization transforms."""
        return v2.Compose([v2.Normalize(mean=mean, std=std)])
    
    def __len__(self):
        return len(self.pre_chip_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Sample with pre/post images (as Clay input format), labels, and masks
        """
        # Load pre and post chips
        pre_chip_path = self.chips_dir / self.pre_chip_files[idx]
        post_chip_path = self.chips_dir / self.post_chip_files[idx]
        
        with rio.open(pre_chip_path) as src:
            pre_img = src.read().astype(np.float32)
            pre_metadata = extract_geospatial_metadata(pre_chip_path)
        
        with rio.open(post_chip_path) as src:
            post_img = src.read().astype(np.float32)
            post_metadata = extract_geospatial_metadata(post_chip_path)
        
        # Load labels and masks
        label_path = self.label_dir / self.label_files[idx]
        mask_path = self.label_dir / self.mask_files[idx]
        water_path = self.label_dir / self.water_files[idx]
        
        label = np.load(label_path).astype(bool).astype(np.uint8)
        excl_mask = np.load(mask_path).astype(bool)
        water_ref = np.load(water_path) == 1
        ignore_mask = excl_mask | water_ref
        
        # Convert to tensors and normalize
        pre_img_t = torch.from_numpy(pre_img)
        post_img_t = torch.from_numpy(post_img)
        
        pre_img_t = self.transform(pre_img_t)
        post_img_t = self.transform(post_img_t)
        
        # Extract geospatial metadata
        pre_latlon = bounds_to_latlon_tensor(pre_metadata['bounds'])
        pre_time = date_to_tensor(pre_metadata['acquisition_date'])
        post_latlon = bounds_to_latlon_tensor(post_metadata['bounds'])
        post_time = date_to_tensor(post_metadata['acquisition_date'])
        
        # Format as Clay encoder expects
        sample = {
            "pre_image": {
                "pixels": pre_img_t,
                "time": pre_time,
                "latlon": pre_latlon,
                "waves": self.waves,
                "gsd": self.gsd,
            },
            "post_image": {
                "pixels": post_img_t,
                "time": post_time,
                "latlon": post_latlon,
                "waves": self.waves,
                "gsd": self.gsd,
            },
            "label": torch.from_numpy(label),
            "ignore_mask": torch.from_numpy(ignore_mask),
            "chip_index": self.chip_indices[idx],
        }
        
        return sample


class EndToEndGFMDataModule(L.LightningDataModule):
    """
    DataModule for end-to-end flood detection with raw images.
    
    Args:
        parent_data_dir: Parent directory containing train/val/test splits
        metadata_path: Path to metadata YAML file
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        platform: Platform identifier (e.g., 'sentinel-1-rtc')
        target_size: Target image size (H, W)
        max_samples: Optional limit on number of samples per split
        train_split_name: Name of training split directory
        val_split_name: Name of validation split directory
        test_split_name: Name of test split directory
    """
    
    def __init__(
        self,
        parent_data_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
        target_size=(224, 224),
        max_samples=None,
        train_split_name="train",
        val_split_name="val",
        test_split_name="test",
    ):
        super().__init__()
        self.parent_data_dir = Path(parent_data_dir)
        self.metadata_path = metadata_path
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform
        self.target_size = target_size
        self.max_samples = max_samples
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name
        self.test_split_name = test_split_name
    
    def _chips_dir(self, split_name):
        return self.parent_data_dir / split_name / "chips"
    
    def _labels_dir(self, split_name):
        return self.parent_data_dir / split_name / "labels"
    
    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage: Stage identifier ('fit', 'test', or None)
        """
        if stage in {"fit", None}:
            self.train_ds = EndToEndGFMDataset(
                chips_dir=self._chips_dir(self.train_split_name),
                label_dir=self._labels_dir(self.train_split_name),
                metadata=self.metadata,
                platform=self.platform,
                target_size=self.target_size,
                max_samples=self.max_samples,
            )
            self.val_ds = EndToEndGFMDataset(
                chips_dir=self._chips_dir(self.val_split_name),
                label_dir=self._labels_dir(self.val_split_name),
                metadata=self.metadata,
                platform=self.platform,
                target_size=self.target_size,
                max_samples=self.max_samples,
            )
        
        if stage == "test":
            self.test_ds = EndToEndGFMDataset(
                chips_dir=self._chips_dir(self.test_split_name),
                label_dir=self._labels_dir(self.test_split_name),
                metadata=self.metadata,
                platform=self.platform,
                target_size=self.target_size,
                max_samples=self.max_samples,
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

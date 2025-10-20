import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_TRAIN_DIR = Path("data/AI4B/FR")  # Source for train/val split
SOURCE_TEST_DIR = Path("data/AI4B/ES")  # Source for test (optional, set to None to disable)
OUTPUT_BASE_DIR = Path("data/AI4B/")
VAL_RATIO = 0.2  # Configurable validation split ratio

# Maximum samples per dataset (set to None for no limit)
MAX_TRAIN_SAMPLES = 200  # e.g., 1000 to limit to 1000 training samples
MAX_VAL_SAMPLES = 40    # e.g., 200 to limit to 200 validation samples
MAX_TEST_SAMPLES = 100   # e.g., 100 to limit to 100 test samples

# Create output directories
TRAIN_DIR = OUTPUT_BASE_DIR / ("FR_train_" + str(MAX_TRAIN_SAMPLES) if MAX_TRAIN_SAMPLES else "FR_train")
VAL_DIR = OUTPUT_BASE_DIR / ("FR_val_" + str(MAX_VAL_SAMPLES) if MAX_VAL_SAMPLES else "FR_val")
TEST_DIR = OUTPUT_BASE_DIR / ("ES_" + str(MAX_TEST_SAMPLES) if MAX_TEST_SAMPLES else "ES")

# Create train and val directories
for split_dir in [TRAIN_DIR, VAL_DIR]:
    split_dir.mkdir(exist_ok=True, parents=True)
    (split_dir / "masks").mkdir(exist_ok=True, parents=True)
    (split_dir / "images").mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
np.random.seed(42)

def get_pairs(source_dir):
    """Get matching mask-image pairs from a source directory."""
    chips_dir = source_dir / "images"
    labels_dir = source_dir / "masks"
    
    # Verify directories exist
    if not chips_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {chips_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {labels_dir}")
    
    labels_files = list(labels_dir.glob("*.tif"))
    print(f"Found {len(labels_files)} mask files in {source_dir.name}")
    
    # Check for corresponding image files
    image_files = list(chips_dir.glob("*.nc"))
    print(f"Found {len(image_files)} .nc image files in {source_dir.name}")
    
    # Build list of (mask, image) pairs where both exist
    pairs = []
    for mask_path in labels_files:
        image_path = chips_dir / (mask_path.stem.replace("label", "") + ".nc")
        if image_path.exists():
            pairs.append((mask_path, image_path))
        else:
            print(f"Warning: image not found for mask {mask_path.name}")
    
    return pairs

# Process train/val split
print("=" * 60)
print("Processing TRAIN/VAL split...")
print("=" * 60)

train_val_pairs = get_pairs(SOURCE_TRAIN_DIR)

if not train_val_pairs:
    raise RuntimeError("No matching mask-image pairs found in train source. Aborting.")

# Shuffle and split train/val
num_total = len(train_val_pairs)
indices = np.random.permutation(num_total)
num_val = int(np.floor(num_total * VAL_RATIO))
val_idx = set(indices[:num_val])
train_idx = set(indices[num_val:])

# Apply max sample limits for train and val
if MAX_TRAIN_SAMPLES is not None and len(train_idx) > MAX_TRAIN_SAMPLES:
    train_idx = set(list(train_idx)[:MAX_TRAIN_SAMPLES])

if MAX_VAL_SAMPLES is not None and len(val_idx) > MAX_VAL_SAMPLES:
    val_idx = set(list(val_idx)[:MAX_VAL_SAMPLES])

print(f"Total pairs: {num_total}")
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Copy train/val files
for i in tqdm(range(num_total), desc="Copying train/val files"):
    # Skip if index not in any split (due to max sample limits)
    if i not in train_idx and i not in val_idx:
        continue
    
    mask_src, img_src = train_val_pairs[i]
    
    if i in val_idx:
        dst_mask = VAL_DIR / "masks" / mask_src.name
        dst_img = VAL_DIR / "images" / img_src.name
    else:
        dst_mask = TRAIN_DIR / "masks" / mask_src.name
        dst_img = TRAIN_DIR / "images" / img_src.name

    shutil.copy2(mask_src, dst_mask)
    shutil.copy2(img_src, dst_img)

# Process test set (if enabled)
if SOURCE_TEST_DIR is not None and SOURCE_TEST_DIR.exists():
    print("\n" + "=" * 60)
    print("Processing TEST set...")
    print("=" * 60)
    
    # Create test directory
    TEST_DIR.mkdir(exist_ok=True, parents=True)
    (TEST_DIR / "masks").mkdir(exist_ok=True, parents=True)
    (TEST_DIR / "images").mkdir(exist_ok=True, parents=True)
    
    test_pairs = get_pairs(SOURCE_TEST_DIR)
    
    if not test_pairs:
        print("Warning: No matching mask-image pairs found in test source. Skipping test set.")
    else:
        # Apply max sample limit for test
        num_test = len(test_pairs)
        if MAX_TEST_SAMPLES is not None and num_test > MAX_TEST_SAMPLES:
            # Shuffle and cap
            test_indices = np.random.permutation(num_test)[:MAX_TEST_SAMPLES]
            test_pairs = [test_pairs[i] for i in test_indices]
            num_test = len(test_pairs)
        
        print(f"Test: {num_test}")
        
        # Copy test files
        for mask_src, img_src in tqdm(test_pairs, desc="Copying test files"):
            dst_mask = TEST_DIR / "masks" / mask_src.name
            dst_img = TEST_DIR / "images" / img_src.name
            
            shutil.copy2(mask_src, dst_mask)
            shutil.copy2(img_src, dst_img)
else:
    print("\n" + "=" * 60)
    print("No test source directory specified or found. Skipping test set.")
    print("=" * 60)

print("\nDataset split complete.")
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIR = Path("data/GFM/ny/train")
OUTPUT_BASE_DIR = Path("data/GFM/ny")
VAL_RATIO = 0.2  # Configurable validation split ratio
BALANCE_TRAIN_SET = True  # Whether to reduce no-flood chips in training set
BALANCE_VAL_SET = False  # Whether to reduce no-flood chips in validation set
NO_FLOOD_KEEP_RATIO = 0.5  # Ratio of no-flood chips to keep if balancing

# Create output directories
TRAIN_DIR = OUTPUT_BASE_DIR / ("train_split_balanced_" + str(BALANCE_TRAIN_SET).lower())
VAL_DIR = OUTPUT_BASE_DIR / ("val_split_balanced_" + str(BALANCE_VAL_SET).lower())

for split_dir in [TRAIN_DIR, VAL_DIR]:
    split_dir.mkdir(exist_ok=True, parents=True)
    (split_dir / "labels").mkdir(exist_ok=True, parents=True)
    (split_dir / "chips").mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Analyzing dataset...")

chips_dir = SOURCE_DIR / "chips"

labels_dir = SOURCE_DIR / "labels"

# Discover flooding files
flooding_files = list(labels_dir.glob("*FLOOD*.npy"))

# Categorize chips
chips_with_flood = []
chips_without_flood = []

print("Categorizing chips...")
for f in tqdm(flooding_files, desc="Processing chips"):
    flooding = np.load(f)
    idx = f.stem.split("chip_")[-1]
    
    # Get corresponding files
    exclayer_file = labels_dir / (f.stem.replace("ENSEMBLE_FLOOD", "ENSEMBLE_EXCLAYER") + ".npy")
    lc_file = labels_dir / (f.stem.replace("ENSEMBLE_FLOOD", "REFERENCE_WATER") + ".npy")

    img_files = sorted(list(chips_dir.glob("*chip_" + str(idx) + "*")))
    
    # Check if chip has flooding
    exclayer = np.load(exclayer_file)
    valid_mask = (exclayer == 0)
    has_flood = (flooding[valid_mask] == 1).sum() > 0
    
    chip_info = {
        'idx': idx,
        'flood_file': f,
        'exclayer_file': exclayer_file,
        'lc_file': lc_file,
        'imgs_files': [img_files[0], img_files[1]]
    }
    
    if has_flood:
        chips_with_flood.append(chip_info)
    else:
        chips_without_flood.append(chip_info)

print(f"\nOriginal dataset:")
print(f"  - Chips with flood: {len(chips_with_flood)}")
print(f"  - Chips without flood: {len(chips_without_flood)}")
print(f"  - Total: {len(flooding_files)}")

# Shuffle both categories
np.random.shuffle(chips_with_flood)
np.random.shuffle(chips_without_flood)

# Split into train/val for each category
n_val_flood = int(len(chips_with_flood) * VAL_RATIO)
n_val_no_flood = int(len(chips_without_flood) * VAL_RATIO)

val_chips_flood = chips_with_flood[:n_val_flood]
train_chips_flood = chips_with_flood[n_val_flood:]

val_chips_no_flood = chips_without_flood[:n_val_no_flood]
train_chips_no_flood = chips_without_flood[n_val_no_flood:]

# Optionally balance training set by reducing no-flood chips
if BALANCE_TRAIN_SET:
    n_no_flood_to_keep = int(len(train_chips_no_flood) * NO_FLOOD_KEEP_RATIO)
    train_chips_no_flood = np.random.choice(
        train_chips_no_flood,
        size=n_no_flood_to_keep,
        replace=False
    ).tolist()
    print(f"\nBalancing training set: keeping {n_no_flood_to_keep}/{len(chips_without_flood) - n_val_no_flood} no-flood chips")

# Optionally balance validation set by reducing no-flood chips
if BALANCE_VAL_SET:
    n_no_flood_to_keep = int(len(val_chips_no_flood) * NO_FLOOD_KEEP_RATIO)
    val_chips_no_flood = np.random.choice(
        val_chips_no_flood,
        size=n_no_flood_to_keep,
        replace=False
    ).tolist()
    print(f"\nBalancing validation set: keeping {n_no_flood_to_keep}/{len(chips_without_flood) - n_val_no_flood} no-flood chips")

# Combine chips for each split
train_chips = train_chips_flood + train_chips_no_flood
val_chips = val_chips_flood + val_chips_no_flood

print(f"\n{'='*60}")
print(f"FINAL SPLIT (val_ratio={VAL_RATIO}):")
print(f"{'='*60}")
print(f"\nTraining set:")
print(f"  - Chips with flood: {len(train_chips_flood)}")
print(f"  - Chips without flood: {len(train_chips_no_flood)}")
print(f"  - Total: {len(train_chips)}")

print(f"\nValidation set:")
print(f"  - Chips with flood: {len(val_chips_flood)}")
print(f"  - Chips without flood: {len(val_chips_no_flood)}")
print(f"  - Total: {len(val_chips)}")

print(f"\nClass balance:")
print(f"  - Train flood ratio: {len(train_chips_flood)/len(train_chips)*100:.1f}%")
print(f"  - Val flood ratio: {len(val_chips_flood)/len(val_chips)*100:.1f}%")

# Function to copy chip files
def copy_chip_files(chips, destination_dir, desc):
    for chip in tqdm(chips, desc=desc):
        # Copy label files
        for src_file in [chip['flood_file'], chip['exclayer_file'], chip['lc_file']]:
            dst_file = destination_dir / "labels" / src_file.name
            shutil.copy2(src_file, dst_file)
        
        # Copy image files
        for img_file in chip["imgs_files"]:
            dst_file = destination_dir / "chips" / img_file.name
            shutil.copy2(img_file, dst_file)

# Copy files to respective directories
print(f"\nCopying files...")
copy_chip_files(train_chips, TRAIN_DIR, "Copying training chips")
copy_chip_files(val_chips, VAL_DIR, "Copying validation chips")

print(f"\n✓ Dataset split created:")
print(f"  - Training set: {TRAIN_DIR}")
print(f"  - Validation set: {VAL_DIR}")

# Print statistics about land/water distribution
def analyze_landcover(chips, split_name):
    print(f"\n{split_name} land cover distribution:")
    land_dominant = 0
    water_dominant = 0
    
    for chip in chips:
        lc = np.load(chip['lc_file'])
        exclayer = np.load(chip['exclayer_file'])
        valid_mask = (exclayer == 0)
        
        land_pixels = (lc[valid_mask] == 0).sum()
        water_pixels = (lc[valid_mask] == 1).sum()
        
        if land_pixels > water_pixels:
            land_dominant += 1
        else:
            water_dominant += 1
    
    total = len(chips)
    print(f"  - Land dominant: {land_dominant} ({land_dominant/total*100:.1f}%)")
    print(f"  - Water dominant: {water_dominant} ({water_dominant/total*100:.1f}%)")

analyze_landcover(train_chips, "Training")
analyze_landcover(val_chips, "Validation")
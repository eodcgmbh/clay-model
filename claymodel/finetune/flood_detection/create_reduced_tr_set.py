import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIR = Path("data/GFM/ny/train")
OUTPUT_DIR = Path("data/GFM/ny/train_balanced")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "chips").mkdir(exist_ok=True, parents=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Analyzing dataset...")

labels_dir = SOURCE_DIR / "labels"
chips_dir = SOURCE_DIR / "chips"

# Discover flooding files
flooding_files = list(labels_dir.glob("*FLOOD*.npy"))

# Categorize chips
chips_with_flood = []
chips_without_flood = []

for f in flooding_files:
    flooding = np.load(f)
    idx = f.stem.split("chip_")[-1]
    
    # Get corresponding files
    exclayer_file = labels_dir / (f.stem.replace("ENSEMBLE_FLOOD", "ENSEMBLE_EXCLAYER") + ".npy")
    lc_file = labels_dir / (f.stem.replace("ENSEMBLE_FLOOD", "REFERENCE_WATER") + ".npy")

    img_files = sorted(list(chips_dir.glob("*chip_" + str(idx) + "*")))
    assert len(img_files) == 2, f"Expected 2 chip files for idx {idx}, found {len(img_files)}"
    
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

print(f"Chips with flood: {len(chips_with_flood)}")
print(f"Chips without flood: {len(chips_without_flood)}")

# Sample half of the no-flood chips
n_no_flood_to_keep = len(chips_without_flood) // 2
sampled_no_flood = np.random.choice(
    chips_without_flood, 
    size=n_no_flood_to_keep, 
    replace=False
)

# Combine all chips to keep
chips_to_copy = chips_with_flood + list(sampled_no_flood)

print(f"\nNew dataset composition:")
print(f"  - Chips with flood: {len(chips_with_flood)}")
print(f"  - Chips without flood (sampled): {n_no_flood_to_keep}")
print(f"  - Total chips: {len(chips_to_copy)}")
print(f"  - Reduction: {len(flooding_files)} → {len(chips_to_copy)} ({len(chips_to_copy)/len(flooding_files)*100:.1f}%)")

# Copy files
print("\nCopying files...")
for chip in tqdm(chips_to_copy, desc="Copying chips"):
    # Copy all three files for this chip
    for src_file in [chip['flood_file'], chip['exclayer_file'], chip['lc_file']]:
        dst_file = OUTPUT_DIR / "labels" / src_file.name
        shutil.copy2(src_file, dst_file)
    for img_file in chip["imgs_files"]:
        dst_file = OUTPUT_DIR / "chips" / img_file.name
        shutil.copy2(img_file, dst_file)

print(f"\n✓ Balanced dataset created at: {OUTPUT_DIR}")

# Print some statistics about the sampled no-flood chips
print("\nNo-flood chips breakdown:")
land_count = 0
water_count = 0

for chip in sampled_no_flood:
    lc = np.load(chip['lc_file'])
    exclayer = np.load(chip['exclayer_file'])
    valid_mask = (exclayer == 0)
    
    land_pixels = (lc[valid_mask] == 0).sum()
    water_pixels = (lc[valid_mask] == 1).sum()
    
    if land_pixels > water_pixels:
        land_count += 1
    else:
        water_count += 1

print(f"  - Mostly land: {land_count} ({land_count/n_no_flood_to_keep*100:.1f}%)")
print(f"  - Mostly water: {water_count} ({water_count/n_no_flood_to_keep*100:.1f}%)")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns

DATA_DIR = Path("data/GFM/ny/val_split_balanced_false/labels")

# Configuration
output_dir = DATA_DIR.parent / "eda_results"
output_dir.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define output files
plot_path = output_dir / "flood_eda_visualization.png"
stats_path = output_dir / "flood_eda_statistics.txt"

# Discover available chip indices from flooding files
flooding_files = sorted(DATA_DIR.glob("*FLOOD*.npy"))
n_scenes = len(flooding_files)

# Initialize storage
stats = {
    'total_pixels': 0,
    'flooded_pixels': 0,
    'non_flooded_pixels': 0,
    'excluded_pixels': 0,
    'land_pixels': 0,
    'permanent_water_pixels': 0,
    'chips_with_flood': 0,
    'chips_without_flood': 0,
    'no_flood_mostly_land': 0,
    'no_flood_mostly_water': 0,
}

chip_details = []

print("Processing chips...")
for f in flooding_files:
    # Load data

    idx = f.stem.split("chip_")[-1]    

    exclayer_file = DATA_DIR / (f.stem.replace("ENSEMBLE_FLOOD", "ENSEMBLE_EXCLAYER") + ".npy")
    lc_file = DATA_DIR / (f.stem.replace("ENSEMBLE_FLOOD", "REFERENCE_WATER") + ".npy")

    flooding = np.load(f)
    exclayer = np.load(exclayer_file)
    lc = np.load(lc_file)
    
    # Get valid pixels (not excluded)
    valid_mask = (exclayer == 0)  # Assuming 0 means not excluded
    
    # Count pixels
    n_total = flooding.size
    n_excluded = (~valid_mask).sum()
    n_valid = valid_mask.sum()
    
    # Count flooding in valid areas
    flooded_pixels = (flooding[valid_mask] == 1).sum()
    non_flooded_pixels = (flooding[valid_mask] == 0).sum()
    
    # Count land/water in valid areas
    land_pixels = (lc[valid_mask] == 0).sum()  
    water_pixels = (lc[valid_mask] == 1).sum() 
    
    # Update global stats
    stats['total_pixels'] += n_total
    stats['excluded_pixels'] += n_excluded
    stats['flooded_pixels'] += flooded_pixels
    stats['non_flooded_pixels'] += non_flooded_pixels
    stats['land_pixels'] += land_pixels
    stats['permanent_water_pixels'] += water_pixels
    
    # Chip-level classification
    has_flood = flooded_pixels > 0
    if has_flood:
        stats['chips_with_flood'] += 1
    else:
        stats['chips_without_flood'] += 1
        # Classify no-flood chips by land/water dominance
        if land_pixels > water_pixels:
            stats['no_flood_mostly_land'] += 1
        else:
            stats['no_flood_mostly_water'] += 1
    
    # Store chip details for further analysis
    chip_details.append({
        'idx': idx,
        'has_flood': has_flood,
        'flood_ratio': flooded_pixels / n_valid if n_valid > 0 else 0,
        'land_ratio': land_pixels / n_valid if n_valid > 0 else 0,
        'excluded_ratio': n_excluded / n_total,
        'n_flooded': flooded_pixels,
        'n_land': land_pixels,
        'n_water': water_pixels,
    })

print("Processing complete!\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1. Overall Class Balance
ax1 = plt.subplot(2, 3, 1)
valid_pixels = stats['flooded_pixels'] + stats['non_flooded_pixels']
labels = ['Flooded', 'Non-Flooded', 'Excluded']
sizes = [stats['flooded_pixels'], stats['non_flooded_pixels'], stats['excluded_pixels']]
colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
explode = (0.05, 0, 0)

wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.2f%%',
                                     colors=colors, explode=explode, startangle=90)
ax1.set_title('Overall Pixel Distribution', fontsize=14, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 2. Class Imbalance (Valid Pixels Only)
ax2 = plt.subplot(2, 3, 2)
flood_ratio = stats['flooded_pixels'] / valid_pixels * 100
non_flood_ratio = stats['non_flooded_pixels'] / valid_pixels * 100
bars = ax2.bar(['Flooded', 'Non-Flooded'], 
               [flood_ratio, non_flood_ratio],
               color=['#ff6b6b', '#4ecdc4'], edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Class Imbalance (Valid Pixels)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}%', ha='center', va='bottom', fontweight='bold')

# 3. Chips with/without Flood
ax3 = plt.subplot(2, 3, 3)
chip_labels = ['With Flood', 'Without Flood']
chip_counts = [stats['chips_with_flood'], stats['chips_without_flood']]
bars = ax3.bar(chip_labels, chip_counts, color=['#e74c3c', '#3498db'], 
               edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Number of Chips', fontsize=12)
ax3.set_title('Chips by Flood Presence', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/n_scenes*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 4. No-Flood Chips: Land vs water
ax4 = plt.subplot(2, 3, 4)
no_flood_labels = ['Mostly Land', 'Mostly water']
no_flood_counts = [stats['no_flood_mostly_land'], stats['no_flood_mostly_water']]
colors_land_water = ['#8b4513', '#1e90ff']
bars = ax4.bar(no_flood_labels, no_flood_counts, color=colors_land_water,
               edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Number of Chips', fontsize=12)
ax4.set_title('No-Flood Chips: Land vs water Dominance', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    total_no_flood = stats['chips_without_flood']
    pct = height/total_no_flood*100 if total_no_flood > 0 else 0
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({pct:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 5. Distribution of Flood Ratios (chips with flood)
ax5 = plt.subplot(2, 3, 5)
chips_with_flood = [c for c in chip_details if c['has_flood']]
flood_ratios = [c['flood_ratio'] * 100 for c in chips_with_flood]
ax5.hist(flood_ratios, bins=50, color='#ff6b6b', edgecolor='black', alpha=0.7)
ax5.set_xlabel('Flood Coverage (%)', fontsize=12)
ax5.set_ylabel('Number of Chips', fontsize=12)
ax5.set_title('Flood Coverage Distribution\n(Chips with Flooding)', 
              fontsize=14, fontweight='bold')
ax5.axvline(np.median(flood_ratios), color='red', linestyle='--', 
            linewidth=2, label=f'Median: {np.median(flood_ratios):.2f}%')
ax5.legend()

# 6. Land vs water Distribution (All Chips)
ax6 = plt.subplot(2, 3, 6)
labels_lc = ['Land', 'water']
sizes_lc = [stats['land_pixels'], stats['permanent_water_pixels']]
colors_lc = ['#8b4513', '#1e90ff']
wedges, texts, autotexts = ax6.pie(sizes_lc, labels=labels_lc, autopct='%1.1f%%',
                                     colors=colors_lc, startangle=90)
ax6.set_title('Overall Land/water Distribution', fontsize=14, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PRINT STATISTICS
# ============================================================================

# Open file to save statistics
with open(stats_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DATASET STATISTICS\n")
    f.write("=" * 80 + "\n")
with open(stats_path, 'a') as f:
    f.write(f"\nTotal Chips: {n_scenes}\n")
    f.write(f"Total Pixels: {stats['total_pixels']:,}\n")
    f.write(f"Valid Pixels: {valid_pixels:,} ({valid_pixels/stats['total_pixels']*100:.2f}%)\n")
    f.write(f"Excluded Pixels: {stats['excluded_pixels']:,} ({stats['excluded_pixels']/stats['total_pixels']*100:.2f}%)\n")

with open(stats_path, 'a') as f:
    f.write(f"\n{'CLASS DISTRIBUTION (Valid Pixels)':^80}\n")
    f.write("-" * 80 + "\n")
    f.write(f"Flooded Pixels: {stats['flooded_pixels']:,} ({flood_ratio:.4f}%)\n")
    f.write(f"Non-Flooded Pixels: {stats['non_flooded_pixels']:,} ({non_flood_ratio:.4f}%)\n")
    f.write(f"Imbalance Ratio: 1:{stats['non_flooded_pixels']/stats['flooded_pixels']:.1f}\n")

with open(stats_path, 'a') as f:
    f.write(f"\n{'CHIP-LEVEL ANALYSIS':^80}\n")
    f.write("-" * 80 + "\n")
    f.write(f"Chips WITH Flood: {stats['chips_with_flood']} ({stats['chips_with_flood']/n_scenes*100:.2f}%)\n")
    f.write(f"Chips WITHOUT Flood: {stats['chips_without_flood']} ({stats['chips_without_flood']/n_scenes*100:.2f}%)\n")

with open(stats_path, 'a') as f:
    f.write(f"\n{'NO-FLOOD CHIPS BREAKDOWN':^80}\n")
    f.write("-" * 80 + "\n")
    if stats['chips_without_flood'] > 0:
        f.write(f"Mostly Land: {stats['no_flood_mostly_land']} ({stats['no_flood_mostly_land']/stats['chips_without_flood']*100:.2f}%)\n")
        f.write(f"Mostly water: {stats['no_flood_mostly_water']} ({stats['no_flood_mostly_water']/stats['chips_without_flood']*100:.2f}%)\n")
    else:
        f.write("No chips without flood\n")

with open(stats_path, 'a') as f:
    f.write(f"\n{'LAND/water DISTRIBUTION':^80}\n")
    f.write("-" * 80 + "\n")
    f.write(f"Land Pixels: {stats['land_pixels']:,} ({stats['land_pixels']/valid_pixels*100:.2f}%)\n")
    f.write(f"water Pixels: {stats['permanent_water_pixels']:,} ({stats['permanent_water_pixels']/valid_pixels*100:.2f}%)\n")

if chips_with_flood:
    with open(stats_path, 'a') as f:
        f.write(f"\n{'FLOOD SEVERITY (Chips with Flooding)':^80}\n")
        f.write("-" * 80 + "\n")
        flood_ratios_array = np.array(flood_ratios)
        f.write(f"Mean Flood Coverage: {np.mean(flood_ratios_array):.2f}%\n")
        f.write(f"Median Flood Coverage: {np.median(flood_ratios_array):.2f}%\n")
        f.write(f"Min Flood Coverage: {np.min(flood_ratios_array):.2f}%\n")
        f.write(f"Max Flood Coverage: {np.max(flood_ratios_array):.2f}%\n")

with open(stats_path, 'a') as f:
    f.write("\n" + "=" * 80 + "\n")

# Print path information
print(f"Results saved to:")
print(f"- Statistics: {stats_path}")
print(f"- Visualization: {plot_path}")
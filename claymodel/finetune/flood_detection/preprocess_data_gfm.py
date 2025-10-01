r"""
Flood Detection Data Processing Script
======================================

   Run the script as follows:
   python preprocess_data.py <data_dir> <output_dir> <chip_size>

   Example:
   python preprocess_data.py data/cvpr/files data/cvpr/ny 224
"""  # noqa E501

import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio

def read_chip_filter(input_dir, output_dir, chip_size, filter_exclusion_layer=None, filter_water=None, filter_nodata=None):
    """
    Reads all GeoTIFF files in a directory, creates chips of specified size,
    and saves them as numpy arrays.

    Args:
        input_dir (str or Path): Directory containing GeoTIFF files.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
        filter_exclusion_layer (function): Function to filter chips based on exclusion layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "chips", exist_ok=True)
    os.makedirs(output_dir / "labels", exist_ok=True)

    file_paths = list(Path(input_dir).glob("*.tif"))
    vh_img_paths = list(Path(input_dir).glob("SIG0*VH*.tif"))
    vv_img_paths = list(Path(input_dir).glob("SIG0*VV*.tif"))
    assert len(vv_img_paths) == 2, "There should be exactly two SIG0 VV images (pre and post flood)"
    assert len(vh_img_paths) == 2, "There should be exactly two SIG0 VH images (pre and post flood)"
    exclayer_path = [fp for fp in file_paths if "EXCLAYER" in fp.name][0]
    ref_water_path = [fp for fp in file_paths if "REFERENCE_WATER" in fp.name][0]
    flood_mask_path = [fp for fp in file_paths if "FLOOD" in fp.name][0]

    with rio.open(vh_img_paths[0]) as vh_0_src, \
         rio.open(vh_img_paths[1]) as vh_1_src, \
         rio.open(vv_img_paths[0]) as vv_0_src, \
         rio.open(vv_img_paths[1]) as vv_1_src, \
         rio.open(exclayer_path) as src3, \
         rio.open(ref_water_path) as src4, \
         rio.open(flood_mask_path) as src5:
        
        vh_0 = vh_0_src.read(1) * vh_0_src.scales[0]
        vh_1 = vh_1_src.read(1) * vh_1_src.scales[0]
        vv_0 = vv_0_src.read(1) * vv_0_src.scales[0]
        vv_1 = vv_1_src.read(1) * vv_1_src.scales[0]

        vv_0_flat = vv_0[vv_0!=vv_0_src.nodatavals[0] * vv_0_src.scales[0]].flatten()
        vv_1_flat = vv_1[vv_1!=vv_1_src.nodatavals[0] * vv_1_src.scales[0]].flatten()
        vv_flat = np.concatenate([vv_0_flat, vv_1_flat])
        vv_mean = np.mean(vv_flat)
        vv_std = np.std(vv_flat)

        vh_0_flat = vh_0[vh_0!=vh_0_src.nodatavals[0]* vh_0_src.scales[0]].flatten()
        vh_1_flat = vh_1[vh_1!=vh_1_src.nodatavals[0]* vh_1_src.scales[0]].flatten()
        vh_flat = np.concatenate([vh_0_flat, vh_1_flat])
        vh_mean = np.mean(vh_flat)
        vh_std = np.std(vh_flat)

        exclayer_data = src3.read(1)
        ref_water_data = src4.read(1)
        flood_mask = src5.read(1)

        n_chips_x = vh_0_src.width // chip_size
        n_chips_y = vh_0_src.height // chip_size

        assert n_chips_x == vh_1_src.width // chip_size == n_chips_x == src5.width // chip_size == src3.width // chip_size == src4.width // chip_size, "All images must have the same dimensions"
        assert n_chips_y == vh_1_src.height // chip_size == n_chips_y == src5.height // chip_size == src3.height // chip_size == src4.height // chip_size, "All images must have the same dimensions"                    

        plot_number = 0

        for i in range(n_chips_x):
            for j in range(n_chips_y):
                chip_index = i * n_chips_y + j                            

                x1, y1 = i * chip_size, j * chip_size
                x2, y2 = x1 + chip_size, y1 + chip_size

                vv_0_chip = vv_0[y1:y2, x1:x2]
                vv_1_chip = vv_1[y1:y2, x1:x2]
                vh_1_chip = vh_1[y1:y2, x1:x2]
                vh_0_chip = vh_0[y1:y2, x1:x2]

                img0_chip = np.zeros((2, chip_size, chip_size), dtype=vh_0.dtype)
                img1_chip = np.zeros((2, chip_size, chip_size), dtype=vh_1.dtype)
                img0_chip[0] = vv_0_chip
                img0_chip[1] = vh_0_chip
                img1_chip[0] = vv_1_chip
                img1_chip[1] = vh_1_chip

                exclayer_chip = exclayer_data[y1:y2, x1:x2]
                ref_water_chip = ref_water_data[y1:y2, x1:x2]
                flood_mask_chip = flood_mask[y1:y2, x1:x2]

                if filter_nodata is not None:
                    if not filter_nodata(vv_0_chip, no_data=vv_0_src.nodatavals) or not filter_nodata(vv_1_chip, no_data=vv_1_src.nodatavals) \
                        or not filter_nodata(vh_0_chip, no_data=vh_0_src.nodatavals) or not filter_nodata(vh_1_chip, no_data=vh_1_src.nodatavals):
                        continue
                if filter_exclusion_layer is not None:
                    if filter_exclusion_layer(exclayer_chip) is np.False_:
                        continue
                if filter_water is not None:
                    if filter_water(ref_water_chip) is np.False_:
                        continue

                img0_base_name = Path(vv_img_paths[0]).stem.replace("VV", "VV_VH")
                img0_chip_path = os.path.join(
                    output_dir / "chips",
                    f"{img0_base_name}_chip_{chip_index}.npy",
                )

                img1_base_name = Path(vv_img_paths[1]).stem.replace("VV", "VV_VH")
                img1_chip_path = os.path.join(
                    output_dir / "chips",
                    f"{img1_base_name}_chip_{chip_index}.npy",
                )

                exclayer_chip_path = os.path.join(
                    output_dir / "labels",
                    f"{Path(exclayer_path).stem}_chip_{chip_index}.npy",
                )
                ref_water_chip_path = os.path.join(
                    output_dir / "labels",
                    f"{Path(ref_water_path).stem}_chip_{chip_index}.npy",
                )
                flood_mask_chip_path = os.path.join(
                    output_dir / "labels",
                    f"{Path(flood_mask_path).stem}_chip_{chip_index}.npy",
                )

                np.save(img0_chip_path, img0_chip)
                np.save(img1_chip_path, img1_chip)
                np.save(exclayer_chip_path, exclayer_chip)
                np.save(ref_water_chip_path, ref_water_chip)
                np.save(flood_mask_chip_path, flood_mask_chip)

                if plot_number >= 10 or np.random.rand() > 0.05:
                    continue

                plot_number += 1
                #plot imges for visual inspection
                import matplotlib.pyplot as plt
                _, axs = plt.subplots(1, 5, figsize=(15, 5))
                axs[0].imshow(img0_chip[0], cmap='gray')
                axs[0].set_title('img-0 SIG0')
                axs[1].imshow(img1_chip[0], cmap='gray')
                axs[1].set_title('img-1 SIG0')
                axs[2].imshow(exclayer_chip, cmap='gray')
                axs[2].set_title('Exclusion Layer')
                axs[3].imshow(ref_water_chip, cmap='gray')
                axs[3].set_title('Reference Water')
                axs[4].imshow(flood_mask_chip, cmap='gray')
                axs[4].set_title('Flood Mask')

                png_chip_path = os.path.join(
                    output_dir,
                    f"chip_{chip_index}.png",
                )
                plt.savefig(png_chip_path)   # saves to file
                plt.close()                
    return {"vv_mean": vv_mean, "vv_std": vv_std, "vh_mean": vh_mean, "vh_std": vh_std}
    

def filter_exclayer(max_exclayer_percent):
    def filter_function(chip):
        exclayer_band = chip[0]
        total_pixels = exclayer_band.size
        exclayer_pixels = np.sum(exclayer_band > 0)  # Count non-zero pixels
        exclayer_percent = exclayer_pixels / total_pixels

        return (exclayer_percent <= max_exclayer_percent)==np.True_
    return filter_function


def filter_water(max_water_percent):
    def filter_function(chip):
        water_band = chip[0]
        total_pixels = water_band.size
        water_pixels = np.sum(water_band == 1)  # Count non-zero pixels
        water_percent = water_pixels / total_pixels

        return (water_percent <= max_water_percent)==np.True_
    return filter_function


def filter_nodata(max_nodata_percent):
    def filter_function(chip, no_data):
        img = chip[0]
        total_pixels = img.size
        # Assuming nodata is represented by 0 values
        # nodata_pixels = np.sum(np.isnan(img))  # Count nodata pixels
        nodata_pixels = np.sum(img==no_data)  # Count nodata pixels
        nodata_percent = nodata_pixels / total_pixels

        return (nodata_percent <= max_nodata_percent)==np.True_
    return filter_function


def main():
# def main(data_dir, output_dir, chip_size):
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """

    MAX_EXCLAYER_PERCENT = 0.75
    MAX_WATER_PERCENT = 0.75
    MAX_NODATA_PERCENT = 0.01

    if len(sys.argv) != 4:  # noqa: PLR2004
        print("Usage: python script.py <data_dir> <output_dir> <chip_size>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chip_size = int(sys.argv[3])
    # data_dir = Path(data_dir)
    # output_dir = Path(output_dir)
    # chip_size = int(chip_size)

    train_dir = data_dir / "train"
    train_output_dir = output_dir / "train"
    print(read_chip_filter(
        train_dir,
        train_output_dir,
        chip_size,
        filter_exclusion_layer=filter_exclayer(MAX_EXCLAYER_PERCENT),
        filter_water=filter_water(MAX_WATER_PERCENT),
        filter_nodata=filter_nodata(MAX_NODATA_PERCENT),
    ))

    test_dir = data_dir / "test"
    test_output_dir = output_dir / "test"
    print(read_chip_filter(
        test_dir,
        test_output_dir,
        chip_size,
        filter_exclusion_layer=filter_exclayer(MAX_EXCLAYER_PERCENT),
        filter_water=filter_water(MAX_WATER_PERCENT),
        filter_nodata=filter_nodata(MAX_NODATA_PERCENT),
    ))


if __name__ == "__main__":
    print(main())
    # print(main("data/GFM/files", "data/GFM/ny", 224))

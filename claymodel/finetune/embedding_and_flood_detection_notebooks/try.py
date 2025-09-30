# %% [markdown]
# # Image encoding

# %% [markdown]
# ## Preparing the dataset
# Make sure your dataset is divided in three folders: "train", "test", and "val". In each of these folders, there should be three sub-folders: chips and labels. In these two sub-folders there should be matching patches of PATHC_SIZE x PATCH_SIZE and format ".npy". Matching objects should have the same name, replacing "naip-new"(in the patches) by "lc"(in the labels).
# 
# If following this tutorial, directory structure should look like:
# ```
# data/
# └── cvpr/
#     ├── files/
#     │   ├── train/
#     │   ├── val/
#     │   └── test/
#     └── ny/
#         ├── train/
#         │   ├── chips/
#         │   └── labels/
#         ├── val/
#         │   ├── chips/
#         │   └── labels/
#         └── test/
#             ├── chips/
#             └── labels/
# ```

# %% [markdown]
# ## Using the model
# 
# 1. Download the Clay model checkpoint from [Huggingface model hub](https://huggingface.co/made-with-clay/Clay/blob/main/v1.5/clay-v1.5.ckpt) and save it in the `checkpoints/` directory.
# 
# 2. Run the model. Configuration file can be found in `configs/extract_embeddings_chesapeake.yaml`. Beaware, the function Encode_images_from_config needs to be run for all three folders "train", "val" and "test". For this you will need to modify the fields "predict_chip_dir", "predict_label_dir" and "output_dir" from the configuration file.

# %%
import os
import shutil
import subprocess
from pathlib import Path

def find_project_root(marker='claymodel'):
    current = Path.cwd()
    for path in [current] + list(current.parents):
        if (path / marker).exists():
            return path
    raise FileNotFoundError(f"Project root not found")

os.chdir(find_project_root())
print(f"Current working directory: {os.getcwd()}")

MODEL_DIR = "checkpoints/"
CHECKPOINT_FILENAME = "clay-v1.5.ckpt"
# CHECKPOINT_FILENAME = "Prithvi_EO_V2_300M.pt"

full_target_path = os.path.join(MODEL_DIR, CHECKPOINT_FILENAME)

if os.path.exists(full_target_path):
    print(f"✓ File already exists at: {full_target_path}")
else:
    print(f"✗ File not found at: {full_target_path}")
    
    if CHECKPOINT_FILENAME != "clay-v1.5.ckpt":
        print("Please download the model.")
        exit(1)

    print("Downloading checkpoint...")
    
    # Create target directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        # Download the file using wget
        download_url = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
        result = subprocess.run(
            ["wget", "-q", download_url],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Move the downloaded file to the target location
        if os.path.exists(CHECKPOINT_FILENAME):
            shutil.move(CHECKPOINT_FILENAME, full_target_path)
            print(f"✓ Successfully downloaded and moved to: {full_target_path}")
        else:
            print("✗ Download failed - file not found after wget")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

# %%
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningArgumentParser, instantiate_class

from claymodel.finetune.segment.embedding_extractor import ClayEmbeddingExtractor
from claymodel.finetune.flood_detection.gfm_datamodule import GFMDataModule

seed_everything(42)  # your seed here

def Encode_images_from_config(config_path, dataset="train"):
    
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(ClayEmbeddingExtractor, "model")
    parser.add_lightning_class_args(GFMDataModule, "data")
    parser.add_lightning_class_args(Trainer, "trainer")
    
    config = parser.parse_path(config_path)
    trainer_config = dict(config["trainer"])
    
    callbacks = []
    for callback_config in trainer_config["callbacks"]:
        if hasattr(callback_config, 'class_path') and hasattr(callback_config, 'init_args'):
            # This is a Namespace object from CLI parsing
            callback_config.init_args["model_path"] = config["model"]["ckpt_path"].replace('/', '_').split('.')[0]
            callback = instantiate_class((), callback_config)
            callbacks.append(callback)
        elif isinstance(callback_config, dict) and "class_path" in callback_config:
            # This is a dictionary configuration
            callback_config["model_path"] = config["model"]["ckpt_path"].replace('/', '_').split('.')[0]    
            callback = instantiate_class((), callback_config)
            callbacks.append(callback)
        else:
            # Already an instantiated callback
            callbacks.append(callback_config)
    
    print(callback_config)
    trainer_config["callbacks"] = callbacks        
    
    # Create instances
    model = ClayEmbeddingExtractor(**config["model"])
    config["data"]["predict_ds"] = dataset
    datamodule = GFMDataModule(**config["data"])
    trainer = Trainer(**trainer_config)

    trainer.predict(model, datamodule)   

    return config

# %%
CONFIG_PATH = 'configs/extract_embeddings_gfm.yaml'
# available_datasets = ["train", "val", "test"]
available_datasets = ["test"]

for ds in available_datasets:
    print(f"Encoding dataset: {ds}")
    Encode_images_from_config(CONFIG_PATH, dataset=ds)


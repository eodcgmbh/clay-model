# %% [markdown]
# # Train segmentation model
# 
# ## Training the segmentation head from embeddings
# 
# In this notebook you can train the segmentation model using pre-computed embeddings via LightningCLI using configurations in `configs/train_embedding_classifier.yaml`. Modify the batch size, learning rate, and other hyperparameters in the configuration file as needed.
# 
# This notebook uses a CSV logger by default for training, validation and test results. A [WandB logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger) can also be used. If you prefer the latter, please switch the loggers and update the entity of the WandB logger configuration in `configs/train_embedding_classifier.yaml`.
# 

# %%
import os
from pathlib import Path

def find_project_root(marker='claymodel'):
    current = Path.cwd()
    for path in [current] + list(current.parents):
        if (path / marker).exists():
            return path
    raise FileNotFoundError(f"Project root not found")

os.chdir(find_project_root())


# %%
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningArgumentParser, instantiate_class

# from claymodel.finetune.flood_detection.embedding_gfm_classifier import EmbeddingClassifierGFM
from claymodel.finetune.flood_detection.embedding_gfm_classifier_2 import EmbeddingClassifierGFM
from claymodel.finetune.flood_detection.embedding_datamodule_gfm import EmbeddingDataModuleGFM

from datetime import datetime

seed_everything(42)  # your seed here

def Train_segmentation_from_embeddings(config_path, test_after_training=False):
    
    objects = ["callbacks", "logger", "plugins"]

    # Create argument parser similar to LightningCLI
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(EmbeddingClassifierGFM, "model")
    parser.add_lightning_class_args(EmbeddingDataModuleGFM, "data")
    parser.add_lightning_class_args(Trainer, "trainer")
    
    # Parse the config file
    config = parser.parse_path(config_path)

    trainer_config = dict(config["trainer"])
    
    # Instantiate objects
    for object in objects:
        if object in trainer_config and trainer_config[object]:
            callbacks = []
            for callback_config in trainer_config[object]:
                if hasattr(callback_config, 'class_path') and hasattr(callback_config, 'init_args'):
                    
                    if object == "logger":
                        if callback_config.class_path == "lightning.pytorch.loggers.CSVLogger":
                            callback_config.init_args['version'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        elif callback_config.class_path == "lightning.pytorch.loggers.WandbLogger":
                            callback_config.init_args['name'] = "GFM_model_late_concat2_" + datetime.now().strftime("%Y%m%d_%H%M%S")

                    if object == "callbacks" and callback_config.class_path == "lightning.pytorch.callbacks.ModelCheckpoint":
                        callback_config.init_args['dirpath'] = os.path.join(callback_config.init_args['dirpath'], datetime.now().strftime("%Y%m%d_%H%M%S"))
                    callback = instantiate_class((), callback_config)
                    callbacks.append(callback)        

                elif isinstance(callback_config, dict) and "class_path" in callback_config:
                    # This is a dictionary configuration
                    callback = instantiate_class((), callback_config)
                    callbacks.append(callback)
                else:
                    # Already an instantiated callback
                    callbacks.append(callback_config)
            
            trainer_config[object] = callbacks

    # Create instances
    model = EmbeddingClassifierGFM(**config["model"])
    datamodule = EmbeddingDataModuleGFM(**config["data"])
    trainer = Trainer(**trainer_config)

    result = trainer.fit(model, datamodule)
    if test_after_training:
        result = trainer.test(model, datamodule)
        print(model.confusion_matrix)

    print("Results:", result)
    
    return model, datamodule, trainer, result


# %% [markdown]
# ## Training the model
# 
# Run the training process. The model will be trained using the configuration specified in `configs/train_embedding_classifier.yaml`. Set `test_after_training=True` to also run evaluation on the test set after training.
# 

# %%
CONFIG_PATH = "configs/train_embedding_flood_detection.yaml"
model, *_ = Train_segmentation_from_embeddings(CONFIG_PATH)
model.draw_graph().visual_graph

# %%
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

cm_normalized = model.confusion_matrix.numpy()
cm_normalized = cm_normalized.astype('float') / cm_normalized.sum(axis=1)[:, np.newaxis]

disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['No Flood', 'Flood'])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Normalized Confusion Matrix')
plt.show()

# %%
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_random_n_chips(model, config_path_yaml, n: int = 5, split: str = "test"):
    """
    Visualize random chips for the GFM setup using paired pre/post embeddings and binary flood labels.

    Shows: pre SIG0, post SIG0, true flood mask, predicted flood mask, and ignore mask.
    """
    with open(config_path_yaml, "r") as f:
        config_dict = yaml.safe_load(f)

    if split == "train":
        embedding_path = Path(config_dict["data"]["train_embedd_dir"])  # .../chips/embeddings_checkpoints_clay-v1/
        labels_path = Path(config_dict["data"]["train_label_dir"])     # .../labels/
    elif split == "val":
        embedding_path = Path(config_dict["data"]["val_embedd_dir"])
        labels_path = Path(config_dict["data"]["val_label_dir"]) 
    elif split == "test":
        embedding_path = Path(config_dict["data"]["test_embedd_dir"])
        labels_path = Path(config_dict["data"]["test_label_dir"]) 
    else:
        raise ValueError("split must be 'train', 'val' or 'test'")

    # Chips are one level above embeddings_checkpoints_clay-v1
    # e.g., .../chips/embeddings_checkpoints_clay-v1 -> .../chips
    chips_path = embedding_path.parent

    # Pair pre/post embeddings by chip index (same logic as EmbeddingDatasetGFM)
    all_embed_files = [p.name for p in embedding_path.glob("*.npy")]
    key_to_files = {}
    for fname in all_embed_files:
        key = fname.split("chip", 1)[1] if "chip" in fname else Path(fname).stem
        key_to_files.setdefault(key, []).append(fname)
    paired_items = [(k, sorted(v)) for k, v in key_to_files.items() if len(v) == 2]
    paired_items.sort(key=lambda x: x[0])

    if len(paired_items) == 0:
        raise RuntimeError(f"No paired embeddings found in {embedding_path}")

    if n > len(paired_items):
        raise ValueError(
            f"n ({n}) is greater than the number of available pairs ({len(paired_items)})"
        )

    selected = random.sample(paired_items, n)

    # Prepare predictions
    model.eval()

    # Show: Pre SIG0, Post SIG0, Overlapped confusion map (TP/TN/FP/FN + Excluded)
    _, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = np.expand_dims(axs, axis=0)

    for row_idx, (key, pair_files) in enumerate(selected):
        pre_name, post_name = pair_files[0], pair_files[1]
        pre_path = embedding_path / pre_name
        post_path = embedding_path / post_name

        # Derive chip index to find corresponding label and chip files
        chip_idx = pre_name.split("chip_", 1)[1].split("_", 1)[0] if "chip_" in pre_name else Path(pre_name).stem

        # Load embeddings
        pre_embedding = np.load(pre_path).astype(np.float32)
        post_embedding = np.load(post_path).astype(np.float32)

        # Predict
        with torch.no_grad():
            batch = {
                "pre_embedding": torch.from_numpy(pre_embedding).unsqueeze(0),
                "post_embedding": torch.from_numpy(post_embedding).unsqueeze(0),
            }
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            pred_mask = (probs > 0.5).astype(np.uint8)

        # Find and load labels/masks
        # FLOOD: 1 = flood, 0 = not flood
        # EXCLAYER: True to ignore
        # REFERENCE_WATER: 1 = permanent water (ignored)
        def find_first(pattern):
            matches = list((labels_path).glob(pattern))
            if len(matches) == 0:
                raise FileNotFoundError(f"No file matching {pattern} in {labels_path}")
            return matches[0]

        lbl_path = find_first(f"*FLOOD*chip_{chip_idx}*.npy")
        excl_path = find_first(f"*EXCLAYER*chip_{chip_idx}*.npy")
        water_path = find_first(f"*REFERENCE_WATER*chip_{chip_idx}*.npy")

        label = np.load(lbl_path).astype(np.uint8)
        excl = np.load(excl_path).astype(bool)
        water = (np.load(water_path) == 1)
        ignore_mask = (excl | water).astype(np.uint8)

        # Load corresponding raw chip images for visualization (pre/post)
        # Expect exactly 2 files per chip index in chips dir
        chip_files = sorted(list((chips_path).glob(f"*chip_{chip_idx}*.npy")))
        pre_img = None
        post_img = None
        if len(chip_files) >= 2:
            # Heuristic: first is pre, second is post after sort
            pre_img_np = np.load(chip_files[0])
            post_img_np = np.load(chip_files[1])
            # Display first band if multi-channel
            pre_img = pre_img_np[0] if pre_img_np.ndim == 3 else pre_img_np
            post_img = post_img_np[0] if post_img_np.ndim == 3 else post_img_np

        # Build overlapped confusion map with custom colors
        # Colors: TP=white, TN=black, FP=green, FN=orange, EXCLUDED=red
        h, w = label.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        excluded = ignore_mask.astype(bool)
        tp = (pred_mask == 1) & (label == 1) & (~excluded)
        tn = (pred_mask == 0) & (label == 0) & (~excluded)
        fp = (pred_mask == 1) & (label == 0) & (~excluded)
        fn = (pred_mask == 0) & (label == 1) & (~excluded)

        # Assign colors
        overlay[tn] = np.array([0.0, 0.0, 0.0], dtype=np.float32)       # black
        overlay[tp] = np.array([1.0, 1.0, 1.0], dtype=np.float32)       # white
        overlay[fp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)       # green
        overlay[fn] = np.array([1.0, 0.5, 0.0], dtype=np.float32)       # orange
        overlay[excluded] = np.array([1.0, 0.0, 0.0], dtype=np.float32) # red

        # Plot
        if pre_img is not None:
            axs[row_idx, 0].imshow(pre_img, cmap="gray")
            axs[row_idx, 0].set_title("Pre SIG0")
        else:
            axs[row_idx, 0].axis("off")
            axs[row_idx, 0].set_title("Pre SIG0 (n/a)")

        if post_img is not None:
            axs[row_idx, 1].imshow(post_img, cmap="gray")
            axs[row_idx, 1].set_title("Post SIG0")
        else:
            axs[row_idx, 1].axis("off")
            axs[row_idx, 1].set_title("Post SIG0 (n/a)")

        axs[row_idx, 2].imshow(overlay)
        axs[row_idx, 2].set_title("Overlap: TP W, TN K, FP G, FN O, EX R")

        for c in range(3):
            axs[row_idx, c].axis("off")

    plt.tight_layout()
    plt.show()


# %%
# Visualize random test samples
plot_random_n_chips(model, config_path_yaml=CONFIG_PATH, n=5, split="test")


# %%




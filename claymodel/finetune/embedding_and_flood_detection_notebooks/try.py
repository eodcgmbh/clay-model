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

from claymodel.finetune.segment.embedding_classifier import EmbeddingClassifier
from claymodel.finetune.segment.embedding_datamodule import EmbeddingDataModule

from datetime import datetime

seed_everything(42)  # your seed here

def Train_segmentation_from_embeddings(config_path, test_after_training=False):
    
    objects = ["callbacks", "logger", "plugins"]

    # Create argument parser similar to LightningCLI
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(EmbeddingClassifier, "model")
    parser.add_lightning_class_args(EmbeddingDataModule, "data")
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
                    # This is a Namespace object from CLI parsing
                    if object == "logger" and callback_config.class_path == "lightning.pytorch.loggers.CSVLogger":
                        callback_config.init_args['version'] = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    model = EmbeddingClassifier(**config["model"])
    datamodule = EmbeddingDataModule(**config["data"])
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
CONFIG_PATH = "configs/train_embedding_classifier.yaml"
model, *_ = Train_segmentation_from_embeddings(CONFIG_PATH)
model.model_graph.visual_graph


# %%




"""
Training script for end-to-end flood detection model.

This script trains a unified model that:
1. Encodes raw SAR images using Clay encoder (frozen)
2. Decodes embeddings back to spatial resolution
3. Classifies flood/no-flood pixels

The Clay encoder is frozen during training, only the decoder and classifier are trained.
"""

import os
from pathlib import Path
from datetime import datetime

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningArgumentParser, instantiate_class
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from claymodel.finetune.flood_detection.end_to_end_flood_classifier import EndToEndFloodClassifier
from claymodel.finetune.flood_detection.end_to_end_gfm_datamodule import EndToEndGFMDataModule


def find_project_root(marker='claymodel'):
    """Find project root directory."""
    current = Path.cwd()
    for path in [current] + list(current.parents):
        if (path / marker).exists():
            return path
    raise FileNotFoundError(f"Project root not found")


def train_end_to_end_model(config_path=None, test_after_training=False):
    """
    Train the end-to-end flood detection model.
    
    Args:
        config_path: Path to YAML configuration file (optional)
        test_after_training: Whether to run test after training
        
    Returns:
        tuple: (model, datamodule, trainer, result)
    """
    seed_everything(42)
    
    if config_path:
        # Parse configuration from YAML file
        parser = LightningArgumentParser()
        parser.add_lightning_class_args(EndToEndFloodClassifier, "model")
        parser.add_lightning_class_args(EndToEndGFMDataModule, "data")
        parser.add_lightning_class_args(Trainer, "trainer")
        
        config = parser.parse_path(config_path)
        
        # Instantiate callbacks, loggers, etc.
        trainer_config = dict(config["trainer"])
        
        for obj_type in ["callbacks", "logger", "plugins"]:
            if obj_type in trainer_config and trainer_config[obj_type]:
                instantiated = []
                for item_config in trainer_config[obj_type]:
                    if hasattr(item_config, 'class_path') and hasattr(item_config, 'init_args'):
                        # Add timestamp to loggers and checkpoints
                        if obj_type == "logger":
                            if item_config.class_path == "lightning.pytorch.loggers.CSVLogger":
                                item_config.init_args['version'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                            elif item_config.class_path == "lightning.pytorch.loggers.WandbLogger":
                                item_config.init_args['name'] = "EndToEnd_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        if obj_type == "callbacks" and item_config.class_path == "lightning.pytorch.callbacks.ModelCheckpoint":
                            item_config.init_args['dirpath'] = os.path.join(
                                item_config.init_args['dirpath'], 
                                datetime.now().strftime("%Y%m%d_%H%M%S")
                            )
                        
                        item = instantiate_class((), item_config)
                        instantiated.append(item)
                    elif isinstance(item_config, dict) and "class_path" in item_config:
                        item = instantiate_class((), item_config)
                        instantiated.append(item)
                    else:
                        instantiated.append(item_config)
                
                trainer_config[obj_type] = instantiated
        
        # Create instances
        model = EndToEndFloodClassifier(**config["model"])
        datamodule = EndToEndGFMDataModule(**config["data"])
        trainer = Trainer(**trainer_config)
    
    else:
        # Use default configuration
        print("⚠️  No config file provided, using default configuration")
        
        # Default model configuration
        model = EndToEndFloodClassifier(
            ckpt_path="checkpoints/clay-v1.5.ckpt",
            embedding_dim=1024,
            patch_size=8,
            target_size=(224, 224),
            hidden_dim=512,
            lr=1e-4,
            wd=1e-4,
            use_aspp=True,
            use_residual=True,
            focal_alpha=0.9,
            focal_gamma=2.0,
            freeze_encoder=True,
            fusion_strategy="early_concat_diff"
        )
        
        # Default datamodule configuration
        datamodule = EndToEndGFMDataModule(
            parent_data_dir="data/gfm",
            metadata_path="configs/metadata.yaml",
            batch_size=8,
            num_workers=4,
            platform="sentinel-1-rtc",
            target_size=(224, 224),
            max_samples=None,
        )
        
        # Default trainer configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                dirpath=f"checkpoints/end_to_end/{timestamp}",
                filename="best-{epoch:02d}-{val/iou:.4f}",
                monitor="val/iou",
                mode="max",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val/iou",
                patience=20,
                mode="max",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        
        logger = CSVLogger(
            save_dir="logs",
            name="end_to_end_flood_detection",
            version=timestamp,
        )
        
        trainer = Trainer(
            max_epochs=100,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            precision="16-mixed",
            gradient_clip_val=1.0,
        )
    
    # Log model config
    try:
        model_config = dict(config["model"]) if config_path else model.hparams
        for logger_instance in trainer.loggers if hasattr(trainer, 'loggers') else [trainer.logger]:
            if isinstance(logger_instance, WandbLogger):
                run = logger_instance.experiment
                if run is not None:
                    run.config.update({"model": model_config}, allow_val_change=True)
            elif isinstance(logger_instance, CSVLogger):
                logger_instance.log_hyperparams({"model": model_config})
    except Exception as e:
        print(f"⚠️  Could not log model config: {e}")
    
    # Train
    print("\n" + "="*80)
    print("🚀 Starting end-to-end training")
    print("="*80)
    print(f"📊 Training samples: {len(datamodule.train_ds) if hasattr(datamodule, 'train_ds') else 'Unknown'}")
    print(f"📊 Validation samples: {len(datamodule.val_ds) if hasattr(datamodule, 'val_ds') else 'Unknown'}")
    print(f"🔒 Clay encoder: FROZEN")
    print(f"🔓 Decoder + Classifier: TRAINABLE")
    print("="*80 + "\n")
    
    result = trainer.fit(model, datamodule)
    
    # Test if requested
    if test_after_training:
        print("\n" + "="*80)
        print("🧪 Running test evaluation")
        print("="*80 + "\n")
        result = trainer.test(model, datamodule)
    
    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)
    
    return model, datamodule, trainer, result


if __name__ == "__main__":
    import argparse
    
    # Change to project root
    try:
        os.chdir(find_project_root())
        print(f"📁 Working directory: {os.getcwd()}")
    except FileNotFoundError:
        print("⚠️  Could not find project root, using current directory")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train end-to-end flood detection model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test evaluation after training"
    )
    
    args = parser.parse_args()
    
    # Train model
    model, datamodule, trainer, result = train_end_to_end_model(
        config_path=args.config,
        test_after_training=args.test
    )
    
    print("\n📝 Training results:", result)

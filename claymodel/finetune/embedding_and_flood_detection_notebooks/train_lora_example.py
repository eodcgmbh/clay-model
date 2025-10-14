"""
Example training script for EndToEndFloodClassifierLoRA.

This script demonstrates how to use the LoRA-optimized flood classifier
for parameter-efficient fine-tuning of the Clay encoder.
"""

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from claymodel.finetune.flood_detection.end_to_end_flood_classifier_lora import EndToEndFloodClassifierLoRA
# Import your datamodule here
# from claymodel.finetune.flood_detection.your_datamodule import YourDataModule


def train_lora_model(
    ckpt_path: str,
    data_dir: str,
    output_dir: str = "./outputs",
    # LoRA hyperparameters
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_modules: list = ['to_q', 'to_v', 'to_out', 'net'],
    # Model hyperparameters
    embedding_dim: int = 1024,
    hidden_dim: int = 512,
    patch_size: int = 8,
    target_size: tuple = (224, 224),
    fusion_strategy: str = "early_concat_diff",
    use_aspp: bool = True,
    use_residual: bool = True,
    # Training hyperparameters
    lr: float = 1e-4,
    wd: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 100,
    # Focal loss hyperparameters
    focal_alpha: float = 0.9,
    focal_gamma: float = 2.0,
    # Hardware
    devices: int = 1,
    accelerator: str = "gpu",
    precision: str = "16-mixed",
    # Logging
    project_name: str = "flood-detection-lora",
    experiment_name: str = None,
):
    """
    Train the LoRA-optimized flood classifier.
    
    Args:
        ckpt_path: Path to Clay encoder checkpoint
        data_dir: Directory containing training data
        output_dir: Directory for saving outputs
        lora_rank: Rank of LoRA matrices (4-16 typical, lower = fewer params)
        lora_alpha: LoRA scaling factor (typically 2*rank)
        lora_dropout: Dropout for LoRA layers
        lora_target_modules: Which modules to apply LoRA to
        embedding_dim: Clay embedding dimension
        hidden_dim: Decoder hidden dimension
        patch_size: Patch size
        target_size: Target image size
        fusion_strategy: Temporal fusion strategy
        use_aspp: Use ASPP module
        use_residual: Use residual connections
        lr: Learning rate
        wd: Weight decay
        batch_size: Batch size
        max_epochs: Maximum epochs
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        devices: Number of devices
        accelerator: Accelerator type
        precision: Training precision
        project_name: W&B project name
        experiment_name: W&B experiment name
    """
    
    # Initialize model with LoRA
    model = EndToEndFloodClassifierLoRA(
        ckpt_path=ckpt_path,
        embedding_dim=embedding_dim,
        patch_size=patch_size,
        target_size=target_size,
        hidden_dim=hidden_dim,
        lr=lr,
        wd=wd,
        use_aspp=use_aspp,
        use_residual=use_residual,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        fusion_strategy=fusion_strategy,
    )
    
    # Initialize datamodule
    # Replace with your actual datamodule
    # datamodule = YourDataModule(
    #     data_dir=data_dir,
    #     batch_size=batch_size,
    #     num_workers=4,
    # )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename="lora-flood-{epoch:02d}-{val/iou:.4f}",
        monitor="val/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/iou",
        patience=15,
        mode="max",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Logger
    logger = WandbLogger(
        project=project_name,
        name=experiment_name or f"lora_r{lora_rank}_a{lora_alpha}",
        save_dir=output_dir,
    )
    
    # Log hyperparameters
    logger.experiment.config.update({
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "fusion_strategy": fusion_strategy,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
    })
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )
    
    # Train
    print("🚀 Starting training with LoRA optimization...")
    # trainer.fit(model, datamodule=datamodule)
    
    # Test
    # print("🧪 Running test...")
    # trainer.test(model, datamodule=datamodule)
    
    print("✅ Training complete!")
    print(f"📁 Checkpoints saved to: {output_dir}/checkpoints")
    
    return model, trainer


def compare_lora_configurations():
    """
    Compare different LoRA configurations to find optimal settings.
    
    This function demonstrates how to experiment with different LoRA ranks
    to balance performance and efficiency.
    """
    configs = [
        {"rank": 4, "alpha": 8.0, "name": "lora_r4"},
        {"rank": 8, "alpha": 16.0, "name": "lora_r8"},
        {"rank": 16, "alpha": 32.0, "name": "lora_r16"},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training with LoRA rank={config['rank']}, alpha={config['alpha']}")
        print(f"{'='*60}\n")
        
        train_lora_model(
            ckpt_path="path/to/clay/checkpoint.ckpt",
            data_dir="path/to/data",
            lora_rank=config["rank"],
            lora_alpha=config["alpha"],
            experiment_name=config["name"],
            max_epochs=50,
        )


def save_lora_only(model, save_path: str):
    """
    Save only LoRA parameters for efficient storage.
    
    Args:
        model: Trained LoRA model
        save_path: Path to save LoRA parameters
    """
    import torch
    
    lora_state = model.get_lora_state_dict()
    torch.save(lora_state, save_path)
    
    # Calculate size
    import os
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"💾 LoRA parameters saved to: {save_path}")
    print(f"📊 File size: {size_mb:.2f} MB")


def load_and_merge_lora(base_ckpt_path: str, lora_path: str):
    """
    Load a base model and merge LoRA weights.
    
    Args:
        base_ckpt_path: Path to base Clay checkpoint
        lora_path: Path to LoRA parameters
        
    Returns:
        Model with LoRA weights loaded
    """
    import torch
    
    # Initialize model
    model = EndToEndFloodClassifierLoRA(
        ckpt_path=base_ckpt_path,
        lora_rank=8,  # Must match saved LoRA
    )
    
    # Load LoRA weights
    lora_state = torch.load(lora_path)
    model.load_lora_state_dict(lora_state)
    
    print(f"✅ LoRA weights loaded from: {lora_path}")
    return model


if __name__ == "__main__":
    # Example 1: Train with default LoRA settings
    model, trainer = train_lora_model(
        ckpt_path="path/to/clay/checkpoint.ckpt",
        data_dir="path/to/flood/data",
        lora_rank=8,
        lora_alpha=16.0,
        max_epochs=100,
    )
    
    # Example 2: Save only LoRA parameters
    save_lora_only(model, "lora_weights.pt")
    
    # Example 3: Compare different LoRA configurations
    # compare_lora_configurations()

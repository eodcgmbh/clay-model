"""
U-Net model for flood detection using GFM datamodule.

This implementation creates a U-Net architecture that works directly with
pre/post SAR images and classifies pixels into flood and no flood classes.
Uses the same exclusion layer and water mask logic as the embedding classifier.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, ConfusionMatrix, BinaryAccuracy
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TemporalFusionUNet(nn.Module):
    """
    U-Net architecture with temporal fusion for pre/post SAR images.
    
    Args:
        in_channels: Number of input channels per image (e.g., 2 for VV/VH)
        num_classes: Number of output classes (1 for binary segmentation)
        bilinear: Whether to use bilinear upsampling
        fusion_strategy: How to fuse pre/post images ('concat', 'diff', 'concat_diff')
    """
    
    def __init__(self, 
                 in_channels: int = 2, 
                 num_classes: int = 1, 
                 bilinear: bool = False,
                 fusion_strategy: str = 'concat_diff'):
        super(TemporalFusionUNet, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Determine input channels based on fusion strategy
        if fusion_strategy == 'concat':
            model_in_channels = in_channels * 2
        elif fusion_strategy == 'diff':
            model_in_channels = in_channels
        elif fusion_strategy == 'concat_diff':
            model_in_channels = in_channels * 3
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # U-Net architecture
        self.inc = DoubleConv(model_in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, pre_image, post_image):
        """
        Forward pass for temporal fusion U-Net.
        
        Args:
            pre_image: [B, C, H, W] - Pre-event SAR image
            post_image: [B, C, H, W] - Post-event SAR image
            
        Returns:
            torch.Tensor: [B, num_classes, H, W] - Segmentation logits
        """
        # Temporal fusion
        if self.fusion_strategy == 'concat':
            x = torch.cat([pre_image, post_image], dim=1)
        elif self.fusion_strategy == 'diff':
            x = post_image - pre_image
        elif self.fusion_strategy == 'concat_diff':
            diff = post_image - pre_image
            x = torch.cat([pre_image, post_image, diff], dim=1)
        
        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetFloodClassifier(L.LightningModule):
    """
    Lightning module for U-Net flood segmentation.
    
    This model works directly with pre/post SAR images and classifies pixels
    into flood and no flood classes, using the same exclusion layer and water
    mask logic as the embedding classifier.
    """
    
    def __init__(self,
                 in_channels: int = 2,
                 num_classes: int = 1,
                 bilinear: bool = False,
                 fusion_strategy: str = 'concat_diff',
                 lr: float = 1e-4,
                 wd: float = 1e-4,
                 b1: float = 0.9,
                 b2: float = 0.95,
                 focal_alpha: float = 0.9,
                 focal_gamma: float = 2.0,
                ):
        """
        Initialize the U-Net flood classifier.
        
        Args:
            in_channels: Number of input channels per image (e.g., 2 for VV/VH)
            num_classes: Number of output classes (1 for binary segmentation)
            bilinear: Whether to use bilinear upsampling
            fusion_strategy: How to fuse pre/post images ('concat', 'diff', 'concat_diff')
            lr: Learning rate
            wd: Weight decay for optimizer
            b1: Beta1 for AdamW optimizer
            b2: Beta2 for AdamW optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.lr = lr
        self.wd = wd
        
        # Create U-Net model
        self.model = TemporalFusionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            bilinear=bilinear,
            fusion_strategy=fusion_strategy
        )
        
        # Loss and metrics (same as embedding classifier)
        self.loss_fn = smp.losses.FocalLoss(mode="binary", alpha=focal_alpha, gamma=focal_gamma, ignore_index=-1)
        self.iou = BinaryJaccardIndex(threshold=0.5, ignore_index=-1)
        self.f1 = BinaryF1Score(threshold=0.5, ignore_index=-1)
        self.OA = BinaryAccuracy(threshold=0.5, ignore_index=-1)
        
        self._confusion_matrix = ConfusionMatrix(
            task="binary",
            threshold=0.5,
            ignore_index=-1,
        )
        self.confusion_matrix = {}
        
        print(f"🏗️  UNetFloodClassifier initialized:")
        print(f"   Input: {in_channels} channels per image")
        print(f"   Output: {num_classes} classes (binary segmentation)")
        print(f"   Fusion strategy: {fusion_strategy}")
        print(f"   Bilinear upsampling: {bilinear}")
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary containing pre/post images
            
        Returns:
            torch.Tensor: Segmentation logits
        """
        return self.model(batch["pre_image"], batch["post_image"])
    
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training/validation/test.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            phase: Phase identifier ('train', 'val', or 'test')
            
        Returns:
            torch.Tensor: Loss value
        """
        labels = batch["label"].int()
        exclude_mask = batch["ignore_mask"].bool()
        labels[exclude_mask] = -1  # Set excluded pixels to ignore_index
        
        outputs = self(batch)
        
        # Binary segmentation: compute loss on logits and metrics on probabilities
        loss = self.loss_fn(outputs, labels.float())
        probs = torch.sigmoid(outputs).squeeze(1)
        iou = self.iou(probs, labels)
        f1 = self.f1(probs, labels)
        oa = self.OA(probs, labels)
        
        # Log metrics
        self.log(f"{phase}/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{phase}/iou", iou, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{phase}/f1", f1, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{phase}/overall_accuracy", oa, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)

        if phase in ["test", "val"]:
            preds = (probs > 0.5).int()
            self._confusion_matrix.update(preds, labels)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    def on_test_end(self):
        self.confusion_matrix["test"] = self._confusion_matrix.compute()
        self._confusion_matrix.reset()
    
    def on_validation_end(self):
        self.confusion_matrix["val"] = self._confusion_matrix.compute()
        self._confusion_matrix.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 0.01,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

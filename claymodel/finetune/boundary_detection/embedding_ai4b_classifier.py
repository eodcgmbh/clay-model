from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryAccuracy
import torchview
import segmentation_models_pytorch as smp
from einops import rearrange

class ResidualBlock(nn.Module):
    """Residual block for feature refinement."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.bn1(self.conv1(x))
        out = out + residual
        out = F.relu(out)
        return out
    
class SequentialBlock(nn.Module):
    """A sequence of convolutional layers with batch norm and ReLU."""
    def __init__(self, channels, num_layers=1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        self.seq = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.seq(x)


class UpsampleBlock(nn.Module):
    """Upsampling block using PixelShuffle with residual connections."""
    def __init__(self, in_channels, out_channels, upscale_factor=2, use_residual=True):
        super().__init__()
        self.conv_before = nn.Conv2d(
            in_channels, 
            out_channels * (upscale_factor ** 2),
            kernel_size=3, 
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels * (upscale_factor ** 2))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        if use_residual:
            self.refine = ResidualBlock(out_channels)
        else:
            self.refine = SequentialBlock(out_channels)
        
    def forward(self, x):
        x = self.conv_before(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_channels, out_channels, atrous_rates=[3, 6, 9]):
        super().__init__()
        
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                pooled = conv(x)
                res.append(F.interpolate(pooled, size=x.shape[2:], 
                                        mode='bilinear', align_corners=False))
            else:
                res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class TemporalFusionSegmentationHead(nn.Module):
    """
    Flexible temporal fusion segmentation head with multiple fusion strategies:
    
    Early Fusion (before decoder):
        - early_concat: Concatenate embeddings [t0, t1]
        - early_diff: Use difference [t1 - t0]
        - early_concat_diff: Concatenate with difference [t0, t1, t1-t0]
    
    Late Fusion (after decoder):
        - late_concat: Process separately, concatenate features
        - late_diff: Process separately, use difference
        - late_concat_diff: Process separately, concatenate features and difference
    
    Siamese variants (late fusion with shared weights):
        - siamese_concat: Siamese processing, concatenate
        - siamese_diff: Siamese processing, use difference
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 target_size: Tuple[int, int] = (224, 224),
                 num_classes: int = 1,
                 patch_size: int = 8,
                 hidden_dim: int = 512,
                 use_aspp: bool = True,
                 use_residual: bool = True,
                 ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.target_size = target_size
        self.num_classes = num_classes
        self.patch_size = patch_size        
        
        # Calculate input channels for initial conv based on fusion strategy
        initial_channels = embedding_dim * 6
        
        # Early fusion: single encoder path
        self.encoder = self._build_encoder(initial_channels, hidden_dim, use_residual, use_aspp)
        
        # Shared decoder (progressive upsampling)
        self.decoder = self._build_decoder(hidden_dim, use_residual)
        
        # boundary classification
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
        )

        # field classification
        self.field_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 8 + 1, hidden_dim // 8 + 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8 + 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim // 8 + 1, 1, kernel_size=1)
        )
        
    def _build_encoder(self, in_channels, hidden_dim, use_residual, use_aspp):
        """Build encoder: initial conv + optional ASPP"""
        encoder = nn.ModuleDict()
        
        encoder['conv_initial'] = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        if use_residual:
            encoder['initial_refine'] = ResidualBlock(hidden_dim)
        else:
            encoder['initial_refine'] = SequentialBlock(hidden_dim)
        
        self.use_residual = use_residual

        if use_aspp:
            encoder['aspp'] = ASPPModule(hidden_dim, hidden_dim, atrous_rates=[3, 6, 9])
        else:
            encoder['aspp'] = nn.Identity()
            
        return encoder
    
    def _apply_encoder(self, x, encoder):
        """Apply encoder module"""
        x = encoder['conv_initial'](x)
        x = encoder['initial_refine'](x)
        
        if self.use_residual:
            x = x + encoder['aspp'](x)
        else:
            x = encoder['aspp'](x)
        return x
    
    def _build_decoder(self, hidden_dim, use_residual):
        """Build progressive upsampling decoder"""
        return nn.ModuleDict({
            'up1': UpsampleBlock(hidden_dim, hidden_dim // 2, upscale_factor=2, use_residual=use_residual),
            'up2': UpsampleBlock(hidden_dim // 2, hidden_dim // 4, upscale_factor=2, use_residual=use_residual),
            'up3': UpsampleBlock(hidden_dim // 4, hidden_dim // 8, upscale_factor=2, use_residual=use_residual),
            'up4': UpsampleBlock(hidden_dim // 8, hidden_dim // 8, upscale_factor=2, use_residual=use_residual)
        })
    
    def forward(self, emb):
        """
        Forward pass for temporal fusion.
        
        Args:
            emb: [B, T, N, D] - embeddings 
            
        Returns:
            torch.Tensor: [B, num_classes, H, W] - segmentation logits
        """
        H_patches = self.target_size[0] // self.patch_size
        W_patches = self.target_size[1] // self.patch_size
        
        # Reshape to spatial format
        x = rearrange(emb, "B T (H W) D -> B (T D) H W", H=H_patches, W=W_patches)
                
        # Early fusion: fuse embeddings first
            
        # Process fused embeddings
        x = self._apply_encoder(x, self.encoder)
        
        # Decode (same for all strategies)
        x = self.decoder['up1'](x)
        x = self.decoder['up2'](x)
        x = self.decoder['up3'](x)
        if self.patch_size == 16:
            x = self.decoder['up4'](x)
        
        # Final classifications
        boundaries = self.boundary_conv(x)
        fields = self.field_conv(torch.cat([x, boundaries], dim=1))
        x = torch.cat([boundaries, fields], dim=1)
        
        return x


class EmbeddingClassifierAI4B(L.LightningModule):
    """
    Lightning module for binary flood segmentation with configurable temporal fusion
    """
    
    def __init__(self,
                 embedding_dim: int = 1024,
                 patch_size: int = 8,
                 target_size: Tuple[int, int] = (224, 224),
                 hidden_dim: int = 512,
                 lr: float = 1e-4,
                 wd: float = 1e-4,
                 b1: float = 0.9,
                 b2: float = 0.95,
                 use_aspp: bool = True,
                 use_residual: bool = True,
                 focal_alpha: float = 0.9,
                 focal_gamma: float = 2.0,
                 ):
        """Initialize the embedding classifier."""
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.target_size = target_size
        self.lr = lr
        self.wd = wd
        
        # Create segmentation head
        self.model = TemporalFusionSegmentationHead(
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            target_size=target_size,
            hidden_dim=hidden_dim,
            use_aspp=use_aspp,
            use_residual=use_residual,
        )
        
        # Loss and metrics
        self.OA = BinaryAccuracy(threshold=0.5, ignore_index=-1)        
        self.loss_fn = smp.losses.FocalLoss(mode="binary", alpha=focal_alpha, gamma=focal_gamma, ignore_index=-1)
        self.iou = BinaryJaccardIndex(threshold=0.5, ignore_index=-1)
        self.f1 = BinaryF1Score(threshold=0.5, ignore_index=-1)
        
        print(f"🏗️  EmbeddingClassifierAI4B initialized:")
        print(f"   Input: {embedding_dim}D embeddings, {patch_size}x{patch_size} patches")
        print(f"   Output: Two products of size {target_size}")
        print(f"   Fusion: early fusion")
        print(f"   ASPP: {use_aspp}, Residual: {use_residual}")
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary containing pre/post embeddings
            
        Returns:
            torch.Tensor: Segmentation logits. Beaware that output has two channels:
                          first for boundaries and second for fields. On the contrary,
                          labels have two channels: first for field labels and second for boundary labels.
        """
        return self.model(batch["embedding"])
    
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
        outputs = self(batch)

        # Split labels into boundary and field channels
        field_labels = labels[:, 0]
        boundary_labels = labels[:, 1]

        # Split outputs into boundary and field logits
        boundary_logits = outputs[:, 0]
        field_logits = outputs[:, 1]

        # Compute losses for each channel
        boundary_loss = self.loss_fn(boundary_logits.contiguous(), boundary_labels.float().contiguous())
        field_loss = self.loss_fn(field_logits.contiguous(), field_labels.float().contiguous())
        loss = boundary_loss + field_loss

        # Convert logits to probabilities
        boundary_probs = torch.sigmoid(boundary_logits)
        field_probs = torch.sigmoid(field_logits)

        # Compute metrics for boundaries
        boundary_iou = self.iou(boundary_probs, boundary_labels)
        boundary_f1 = self.f1(boundary_probs, boundary_labels)
        boundary_oa = self.OA(boundary_probs, boundary_labels)

        # Compute metrics for fields
        field_iou = self.iou(field_probs, field_labels)
        field_f1 = self.f1(field_probs, field_labels)
        field_oa = self.OA(field_probs, field_labels)

        mean_iou = (boundary_iou + field_iou) / 2.0
        mean_f1 = (boundary_f1 + field_f1) / 2.0
        mean_oa = (boundary_oa + field_oa) / 2.0

        # Log metrics for both channels
        metrics = {
            f"{phase}/loss": loss,
            f"{phase}/boundary_loss": boundary_loss,
            f"{phase}/field_loss": field_loss,
            f"{phase}/boundary_iou": boundary_iou,
            f"{phase}/boundary_f1": boundary_f1,
            f"{phase}/boundary_oa": boundary_oa,
            f"{phase}/field_iou": field_iou,
            f"{phase}/field_f1": field_f1,
            f"{phase}/field_oa": field_oa,
            f"{phase}/iou": mean_iou,
            f"{phase}/f1": mean_f1,
            f"{phase}/oa": mean_oa,
        }

        for name, value in metrics.items():
            self.log(name, value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    #def on_test_end(self):
    #    self.confusion_matrix["test"] = self._confusion_matrix.compute()
    #    self.confusion_matrix.reset()
    
#    def on_train_end(self):
#        self.confusion_matrix["train"] = self._confusion_matrix["train"].compute()

    #def on_validation_end(self):
    #    self.confusion_matrix["val"] = self._confusion_matrix.compute()
    #    self._confusion_matrix.reset()
    
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

    def draw_graph(self):
        self.model_graph = torchview.draw_graph(
            self.model, 
            input_size=(1, 6, (self.target_size[0] // self.patch_size) * (self.target_size[1] // self.patch_size), self.embedding_dim),
            expand_nested=True,
            graph_name='EmbeddingBinarySegmentationHead'
        )
        return self.model_graph


if __name__ == "__main__":
    pass
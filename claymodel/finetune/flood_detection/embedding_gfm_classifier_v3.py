from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, ConfusionMatrix, BinaryAccuracy
import torchview
import segmentation_models_pytorch as smp
from einops import rearrange

class ResidualBlock(nn.Module):
    """Residual block for feature refinement"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = out + residual
        out = F.relu(out)
        return out
    
class SequentialBlock(nn.Module):
    """A sequence of convolutional layers with batch norm and ReLU"""
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
    """Upsampling block using PixelShuffle with residual connections"""
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
    """Atrous Spatial Pyramid Pooling"""
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


class SARAdapter(nn.Module):
    """
    Multi-scale SAR feature extractor for auxiliary information.
    Extracts features at three scales matching decoder stages.
    """
    def __init__(self, in_channels=4, hidden_dim=512):
        super().__init__()
        
        # Scale 1: Match embedding resolution (e.g., 28x28 for 224x224 input with patch_size=8)
        # This will be concatenated right after encoding
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=7, stride=8, padding=3),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Scale 2: Match first upsampling stage (56x56)
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Scale 3: Match second upsampling stage (112x112)
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )

        # Full-resolution SAR adapter (scale4) to fuse just before classification
        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, sar):
        """
        Args:
            sar: [B, 3, H, W] - Original SAR images
        Returns:
            Dictionary with features at three scales
        """
        return {
            'scale1': self.scale1(sar),  # Coarsest - matches embedding resolution
            'scale2': self.scale2(sar),  # Medium
            'scale3': self.scale3(sar),  # Finest
            'scale4': self.scale4(sar)   # Full resolution
        }


class TemporalFusionSegmentationHead(nn.Module):
    """
    Flexible temporal fusion segmentation head with SAR auxiliary input.
    
    NEW: Accepts SAR images and fuses multi-scale SAR features throughout decoder.
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 target_size: Tuple[int, int] = (224, 224),
                 num_classes: int = 1,
                 patch_size: int = 8,
                 hidden_dim: int = 512,
                 use_aspp: bool = True,
                 use_residual: bool = True,
                 sar_channels: int = 2,
                 fusion_strategy: Literal[
                     "early_concat", "early_diff", "early_concat_diff",
                     "late_concat", "late_diff", "late_concat_diff",
                     "siamese_concat", "siamese_diff"
                 ] = "early_concat_diff"):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.target_size = target_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.fusion_strategy = fusion_strategy

    
        self.sar_adapter = SARAdapter(in_channels=sar_channels*2, hidden_dim=hidden_dim)
        
        # Determine if we need siamese/dual processing
        self.is_late_fusion = fusion_strategy.startswith("late_") or fusion_strategy.startswith("siamese_")
        self.is_siamese = fusion_strategy.startswith("siamese_")
        
        # Calculate input channels for initial conv based on fusion strategy
        if fusion_strategy in ["early_concat", "siamese_concat", "late_concat"]:
            initial_channels = embedding_dim * 2
        elif fusion_strategy in ["early_diff", "siamese_diff", "late_diff"]:
            initial_channels = embedding_dim
        elif fusion_strategy in ["early_concat_diff", "late_concat_diff"]:
            initial_channels = embedding_dim * 3
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # For late fusion, we process each timestamp separately first
        if self.is_late_fusion:
            self.encoder_t0 = self._build_encoder(embedding_dim, hidden_dim, use_residual, use_aspp)
            if not self.is_siamese:
                self.encoder_t1 = self._build_encoder(embedding_dim, hidden_dim, use_residual, use_aspp)
            
            if "concat" in fusion_strategy:
                decoder_input_channels = hidden_dim * 2
            elif "diff" in fusion_strategy and "concat_diff" not in fusion_strategy:
                decoder_input_channels = hidden_dim
            else:  # concat_diff
                decoder_input_channels = hidden_dim * 3
            
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(decoder_input_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            # Early fusion: single encoder path
            self.encoder = self._build_encoder(initial_channels, hidden_dim, use_residual, use_aspp)
        
        # Adjust decoder input channels if using SAR fusion at scale1
        decoder_start_channels = hidden_dim + hidden_dim  # Embedding + SAR features
        
        # Shared decoder with SAR fusion (concat mode)
        self.decoder = self._build_decoder(
            decoder_start_channels, 
            hidden_dim, 
            use_residual
        )
        
        # Final classification
        # final_channels = hidden_dim // 8 + hidden_dim // 4  # Decoder + SAR scale3
        final_channels = hidden_dim // 8 * 2 # Decoder + SAR scale4
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_channels, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim // 8, num_classes, kernel_size=1)
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
    
    def _build_decoder(self, in_channels, hidden_dim, use_residual):
        """Build progressive upsampling decoder with SAR fusion points"""
        decoder = nn.ModuleDict()
        
        # Fusion convolutions to handle concatenated features
        # After scale1 SAR fusion
        decoder['fusion1'] = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Up1: hidden_dim -> hidden_dim // 2
        decoder['up1'] = UpsampleBlock(hidden_dim, hidden_dim // 2, upscale_factor=2, use_residual=use_residual)
        
        # After up1, concat with scale2: (hidden_dim // 2 + hidden_dim // 2)
        decoder['fusion2'] = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Up2: hidden_dim // 2 -> hidden_dim // 4
        decoder['up2'] = UpsampleBlock(hidden_dim // 2, hidden_dim // 4, upscale_factor=2, use_residual=use_residual)
        
        # After up2, concat with scale3: (hidden_dim // 4 + hidden_dim // 4)
        decoder['fusion3'] = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # Up3: hidden_dim // 4 -> hidden_dim // 8
        decoder['up3'] = UpsampleBlock(hidden_dim // 4, hidden_dim // 8, upscale_factor=2, use_residual=use_residual)
        
        return decoder
    
    def forward(self, emb_t0, emb_t1, sar_t0=None, sar_t1=None):
        """
        Args:
            emb_t0: [B, N, D] - embeddings from time 0
            emb_t1: [B, N, D] - embeddings from time 1
            sar_t0: [B, C, H, W] - SAR image at time 0 (optional)
            sar_t1: [B, C, H, W] - SAR image at time 1 (optional)
        Returns:
            logits: [B, num_classes, H, W]
        """
        H_patches = self.target_size[0] // self.patch_size
        W_patches = self.target_size[1] // self.patch_size
        
        # Reshape to spatial format
        x0 = rearrange(emb_t0, "B (H W) D -> B D H W", H=H_patches, W=W_patches)
        x1 = rearrange(emb_t1, "B (H W) D -> B D H W", H=H_patches, W=W_patches)
        
        # Concatenate temporal SAR features
        sar_combined = torch.cat([sar_t0, sar_t1], dim=1)  # [B, 2C, H, W]
        sar_features = self.sar_adapter(sar_combined)
    
        # Temporal fusion (same as before)
        if self.is_late_fusion:
            feat_t0 = self._apply_encoder(x0, self.encoder_t0)
            if self.is_siamese:
                feat_t1 = self._apply_encoder(x1, self.encoder_t0)
            else:
                feat_t1 = self._apply_encoder(x1, self.encoder_t1)
            
            if "concat_diff" in self.fusion_strategy:
                x = torch.cat([feat_t0, feat_t1, feat_t1 - feat_t0], dim=1)
            elif "concat" in self.fusion_strategy:
                x = torch.cat([feat_t0, feat_t1], dim=1)
            else:  # diff only
                x = feat_t1 - feat_t0
            
            x = self.fusion_conv(x)
        else:
            if self.fusion_strategy == "early_concat":
                x = torch.cat([x0, x1], dim=1)
            elif self.fusion_strategy == "early_diff":
                x = x1 - x0
            elif self.fusion_strategy == "early_concat_diff":
                x = torch.cat([x0, x1, x1 - x0], dim=1)
            
            x = self._apply_encoder(x, self.encoder)
            
            # Fuse scale1 SAR features
        x = torch.cat([x, sar_features['scale1']], dim=1)
        x = self.decoder['fusion1'](x)
        
        # Up1 + fuse scale2
        x = self.decoder['up1'](x)
        x = torch.cat([x, sar_features['scale2']], dim=1)
        x = self.decoder['fusion2'](x)
        
        # Up2 + fuse scale3
        x = self.decoder['up2'](x)
        x = torch.cat([x, sar_features['scale3']], dim=1)
        x = self.decoder['fusion3'](x)
        
        # Up3
        x = self.decoder['up3'](x)        

        # Full-resolution SAR fusion (scale4) before classification
        x = torch.cat([x, sar_features['scale4']], dim=1)
        
        # Final classification
        x = self.final_conv(x)
        
        return x


class EmbeddingClassifierGFM(L.LightningModule):
    """
    Lightning module for binary flood segmentation with SAR auxiliary input
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
                 sar_channels: int = 2,
                 fusion_strategy: Literal[
                     "early_concat", "early_diff", "early_concat_diff",
                     "late_concat", "late_diff", "late_concat_diff",
                     "siamese_concat", "siamese_diff"
                 ] = "early_concat_diff"):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            patch_size: Patch size used in embeddings
            target_size: Target output size (h_out, w_out)
            hidden_dim: Hidden dimensions for decoder
            lr: Learning rate
            wd: Weight decay for optimizer
            b1, b2: Adam betas
            use_aspp: Whether to use ASPP module
            use_residual: Whether to use residual connections
            use_sar_fusion: Whether to use SAR auxiliary input (NEW)
            sar_channels: Number of SAR channels (NEW)
            sar_fusion_mode: How to fuse SAR features - 'concat' or 'add' (NEW)
            fusion_strategy: How to fuse temporal embeddings
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.target_size = target_size
        self.num_classes = 1
        self.lr = lr
        self.wd = wd
        self.fusion_strategy = fusion_strategy
        self.sar_channels = sar_channels
        
        # Create segmentation head
        self.model = TemporalFusionSegmentationHead(
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            target_size=target_size,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            use_aspp=use_aspp,
            use_residual=use_residual,
            sar_channels=sar_channels,
            fusion_strategy=fusion_strategy
        )
        
        # Loss and metrics (aligned with v2)
        self.OA = BinaryAccuracy(threshold=0.5, ignore_index=-1)
        self.loss_fn = smp.losses.FocalLoss(mode="binary", ignore_index=-1)
        self.iou = BinaryJaccardIndex(threshold=0.5, ignore_index=-1)
        self.f1 = BinaryF1Score(threshold=0.5, ignore_index=-1)
        self._confusion_matrix = ConfusionMatrix(
            task="binary",
            threshold=0.5,
            ignore_index=-1,
        )
        self.confusion_matrix = {}
        
        print(f"🏗️  EmbeddingClassifierGFM initialized:")
        print(f"   Input: {embedding_dim}D embeddings, {patch_size}x{patch_size} patches")
        print(f"   Output: Binary segmentation, {target_size} spatial")
        print(f"   Fusion: {fusion_strategy}")
        print(f"   SAR Fusion: True (mode: concat)")
        print(f"   ASPP: {use_aspp}, Residual: {use_residual}")
    
    def forward(self, batch):
        """Forward pass"""
        
        return self.model(
            batch["pre_embedding"], 
            batch["post_embedding"],
            batch["pre_image"],
            batch["post_image"]
        )
    
    def shared_step(self, batch, batch_idx, phase):
        """Shared step for training/validation/test"""
        labels = batch["label"].int()
        exclude_mask = batch["ignore_mask"].bool()
        labels[exclude_mask] = -1
        
        outputs = self(batch)
        
        loss = self.loss_fn(outputs, labels.float())
        probs = torch.sigmoid(outputs).squeeze(1)
        iou = self.iou(probs, labels)
        f1 = self.f1(probs, labels)
        oa = self.OA(probs, labels)
        
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
        self.confusion_matrix = self._confusion_matrix.compute()
        self._confusion_matrix.reset()
    
    def on_validation_end(self):
        self.confusion_matrix = self._confusion_matrix.compute()
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

    def draw_graph(self):
        self.model_graph = torchview.draw_graph(
            self.model, 
            input_size=(
                (1, (self.target_size[0] // self.patch_size) * (self.target_size[1] // self.patch_size), self.embedding_dim),
                (1, (self.target_size[0] // self.patch_size) * (self.target_size[1] // self.patch_size), self.embedding_dim),
                (1, self.sar_channels, self.target_size[0], self.target_size[1]),  # SAR t0
                (1, self.sar_channels, self.target_size[0], self.target_size[1])   # SAR t1
            ),
            expand_nested=True,
            graph_name='EmbeddingBinarySegmentationHead'
        )
        return self.model_graph
"""
End-to-end flood detection model with LoRA optimization applied to the frozen Clay encoder.

This model combines:
1. Clay encoder (frozen with LoRA adapters) - extracts embeddings with parameter-efficient fine-tuning
2. Decoder - upsamples embeddings back to image resolution
3. Binary classifier - predicts flood/no-flood

LoRA (Low-Rank Adaptation) allows efficient fine-tuning of the frozen encoder by adding
trainable low-rank matrices to the attention and MLP layers.
"""

from typing import Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryAccuracy
import segmentation_models_pytorch as smp
from einops import rearrange

from claymodel.module import ClayMAEModule


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for parameter-efficient fine-tuning.
    
    Adds trainable low-rank matrices A and B to a frozen linear layer:
    h = W_0 x + (B @ A) x
    
    where W_0 is frozen and B @ A is the low-rank adaptation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.
        
        Args:
            x: Input tensor [*, in_features]
            
        Returns:
            LoRA output [*, out_features]
        """
        # x @ A @ B with scaling
        result = self.dropout(x) @ self.lora_A @ self.lora_B
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Wrapper that combines a frozen linear layer with LoRA adaptation.
    """
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen linear + LoRA adaptation"""
        return self.linear(x) + self.lora(x)


def apply_lora_to_linear(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list = None
) -> None:
    """
    Recursively apply LoRA to linear layers in a module.
    
    Args:
        module: Module to apply LoRA to
        rank: Rank of LoRA matrices
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA
        target_modules: List of module name patterns to target (e.g., ['q', 'v', 'mlp'])
    """
    if target_modules is None:
        target_modules = ['q', 'v', 'k', 'proj', 'fc1', 'fc2']
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if this linear layer should be adapted
            should_adapt = any(target in name for target in target_modules)
            if should_adapt:
                # Replace with LoRA-enhanced version
                setattr(
                    module,
                    name,
                    LinearWithLoRA(child, rank=rank, alpha=alpha, dropout=dropout)
                )
        else:
            # Recursively apply to child modules
            apply_lora_to_linear(child, rank, alpha, dropout, target_modules)


class ResidualBlock(nn.Module):
    """Residual block for feature refinement."""
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
    Temporal fusion segmentation head with multiple fusion strategies.
    Processes pre/post embeddings and outputs binary segmentation.
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 target_size: Tuple[int, int] = (224, 224),
                 num_classes: int = 1,
                 patch_size: int = 8,
                 hidden_dim: int = 512,
                 use_aspp: bool = True,
                 use_residual: bool = True,
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
        
        # Shared decoder (progressive upsampling)
        self.decoder = self._build_decoder(hidden_dim, use_residual)
        
        # Final classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
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
    
    def _build_decoder(self, hidden_dim, use_residual):
        """Build progressive upsampling decoder"""
        return nn.ModuleDict({
            'up1': UpsampleBlock(hidden_dim, hidden_dim // 2, upscale_factor=2, use_residual=use_residual),
            'up2': UpsampleBlock(hidden_dim // 2, hidden_dim // 4, upscale_factor=2, use_residual=use_residual),
            'up3': UpsampleBlock(hidden_dim // 4, hidden_dim // 8, upscale_factor=2, use_residual=use_residual)
        })
    
    def forward(self, emb_t0, emb_t1):
        """
        Forward pass for temporal fusion.
        
        Args:
            emb_t0: [B, N, D] - embeddings from time 0
            emb_t1: [B, N, D] - embeddings from time 1
            
        Returns:
            torch.Tensor: [B, num_classes, H, W] - segmentation logits
        """
        H_patches = self.target_size[0] // self.patch_size
        W_patches = self.target_size[1] // self.patch_size
        
        # Reshape to spatial format
        x0 = rearrange(emb_t0, "B (H W) D -> B D H W", H=H_patches, W=W_patches)
        x1 = rearrange(emb_t1, "B (H W) D -> B D H W", H=H_patches, W=W_patches)
        
        if self.is_late_fusion:
            # Process each timestamp through encoder (shared or separate)
            feat_t0 = self._apply_encoder(x0, self.encoder_t0)
            if self.is_siamese:
                feat_t1 = self._apply_encoder(x1, self.encoder_t0)  # Shared weights
            else:
                feat_t1 = self._apply_encoder(x1, self.encoder_t1)  # Separate weights
            
            # Fuse encoded features
            if "concat_diff" in self.fusion_strategy:
                x = torch.cat([feat_t0, feat_t1, feat_t1 - feat_t0], dim=1)
            elif "concat" in self.fusion_strategy:
                x = torch.cat([feat_t0, feat_t1], dim=1)
            else:  # diff only
                x = feat_t1 - feat_t0
            
            x = self.fusion_conv(x)
        else:
            # Early fusion: fuse embeddings first
            if self.fusion_strategy == "early_concat":
                x = torch.cat([x0, x1], dim=1)
            elif self.fusion_strategy == "early_diff":
                x = x1 - x0
            elif self.fusion_strategy == "early_concat_diff":
                x = torch.cat([x0, x1, x1 - x0], dim=1)
            
            # Process fused embeddings
            x = self._apply_encoder(x, self.encoder)
        
        # Decode (same for all strategies)
        x = self.decoder['up1'](x)
        x = self.decoder['up2'](x)
        x = self.decoder['up3'](x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x


class EndToEndFloodClassifierLoRA(L.LightningModule):
    """
    End-to-end Lightning module for flood detection with LoRA optimization.
    
    This model:
    1. Encodes raw images using Clay encoder (frozen with LoRA adapters)
    2. Decodes embeddings back to spatial resolution
    3. Classifies flood/no-flood pixels
    
    LoRA allows parameter-efficient fine-tuning of the frozen encoder by adding
    trainable low-rank matrices to attention and MLP layers.
    """
    
    def __init__(self,
                 ckpt_path: str,
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
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_dropout: float = 0.0,
                 lora_target_modules: list = None,
                 fusion_strategy: Literal[
                     "early_concat", "early_diff", "early_concat_diff",
                     "late_concat", "late_diff", "late_concat_diff",
                     "siamese_concat", "siamese_diff"
                 ] = "early_concat_diff"):
        """
        Initialize the end-to-end classifier with LoRA.
        
        Args:
            ckpt_path: Path to Clay encoder checkpoint
            embedding_dim: Dimension of Clay embeddings
            patch_size: Size of patches
            target_size: Target image size
            hidden_dim: Hidden dimension for decoder
            lr: Learning rate
            wd: Weight decay
            b1: Adam beta1
            b2: Adam beta2
            use_aspp: Use ASPP module
            use_residual: Use residual connections
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
            lora_rank: Rank of LoRA matrices (lower = fewer parameters)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA layers
            lora_target_modules: List of module names to apply LoRA to
            fusion_strategy: Strategy for temporal fusion
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.target_size = target_size
        self.num_classes = 1  # Binary segmentation
        self.lr = lr
        self.wd = wd
        self.fusion_strategy = fusion_strategy
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Load Clay encoder
        print(f"🔄 Loading Clay encoder from: {ckpt_path}")
        self.clay_encoder = ClayMAEModule.load_from_checkpoint(
            ckpt_path,
            strict=False,
            model_size="large",
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1]*7,
            mask_ratio=0.0,
            shuffle=False
        ).model.encoder
        
        # Freeze all encoder parameters first
        for param in self.clay_encoder.parameters():
            param.requires_grad = False
        
        # Apply LoRA to encoder
        if lora_target_modules is None:
            # Default: apply to query, value projections and MLP layers
            lora_target_modules = ['to_q', 'to_v', 'to_out', 'net']
        
        print(f"🔧 Applying LoRA to Clay encoder:")
        print(f"   Rank: {lora_rank}")
        print(f"   Alpha: {lora_alpha}")
        print(f"   Dropout: {lora_dropout}")
        print(f"   Target modules: {lora_target_modules}")
        
        apply_lora_to_linear(
            self.clay_encoder,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules
        )
        
        # Count trainable parameters
        encoder_params = sum(p.numel() for p in self.clay_encoder.parameters() if p.requires_grad)
        total_encoder_params = sum(p.numel() for p in self.clay_encoder.parameters())
        
        print(f"📊 Encoder parameters:")
        print(f"   Trainable (LoRA): {encoder_params:,}")
        print(f"   Frozen: {total_encoder_params - encoder_params:,}")
        print(f"   Total: {total_encoder_params:,}")
        print(f"   LoRA efficiency: {100 * encoder_params / total_encoder_params:.2f}%")
        
        # Keep encoder in eval mode (BatchNorm/Dropout frozen)
        self.clay_encoder.eval()
        
        # Create segmentation head (decoder + classifier)
        self.segmentation_head = TemporalFusionSegmentationHead(
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            target_size=target_size,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            use_aspp=use_aspp,
            use_residual=use_residual,
            fusion_strategy=fusion_strategy
        )
        
        # Loss and metrics
        self.OA = BinaryAccuracy(threshold=0.5, ignore_index=-1)        
        self.loss_fn = smp.losses.FocalLoss(mode="binary", alpha=focal_alpha, gamma=focal_gamma, ignore_index=-1)
        self.iou = BinaryJaccardIndex(threshold=0.5, ignore_index=-1)
        self.f1 = BinaryF1Score(threshold=0.5, ignore_index=-1)
        
        print(f"🏗️  EndToEndFloodClassifierLoRA initialized:")
        print(f"   Input: Raw images {target_size}")
        print(f"   Encoder: Clay (frozen with LoRA adapters)")
        print(f"   Decoder: {hidden_dim}D hidden, ASPP={use_aspp}, Residual={use_residual}")
        print(f"   Fusion: {fusion_strategy}")
        print(f"   Output: Binary flood segmentation")
    
    def encode_image(self, image_batch):
        """
        Encode raw images using Clay encoder with LoRA.
        
        Args:
            image_batch: Dict with 'pixels', 'time', 'latlon', 'waves', 'gsd'
            
        Returns:
            torch.Tensor: [B, N, D] embeddings (excluding CLS token)
        """
        # Even though encoder has LoRA, we keep it in eval mode
        # to freeze BatchNorm/Dropout, but LoRA layers will still be trained
        unmsk_patch, *_ = self.clay_encoder(image_batch)
        
        # Return all patch embeddings (excluding CLS token at position 0)
        embeddings = unmsk_patch[:, 1:, :]
        return embeddings
    
    def forward(self, batch):
        """
        Forward pass: encode -> decode -> classify.
        
        Args:
            batch: Dictionary containing pre/post image batches
            
        Returns:
            torch.Tensor: Segmentation logits [B, 1, H, W]
        """
        # Encode pre and post images
        pre_embeddings = self.encode_image(batch["pre_image"])
        post_embeddings = self.encode_image(batch["post_image"])
        
        # Decode and classify
        logits = self.segmentation_head(pre_embeddings, post_embeddings)
        
        return logits
    
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
        labels[exclude_mask] = -1
        
        outputs = self(batch)
        
        # Binary segmentation: compute loss on logits, metrics on probabilities
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
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        # Optimize LoRA parameters + decoder + classifier
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        print(f"🎯 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
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
    
    def on_train_start(self):
        """Ensure encoder stays in eval mode (but LoRA layers are trainable)."""
        self.clay_encoder.eval()
    
    def on_train_epoch_start(self):
        """Ensure encoder stays in eval mode at the start of each epoch."""
        self.clay_encoder.eval()
    
    def get_lora_state_dict(self):
        """
        Extract only LoRA parameters for efficient checkpoint saving.
        
        Returns:
            dict: State dict containing only LoRA parameters
        """
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_state_dict[name] = param
        return lora_state_dict
    
    def load_lora_state_dict(self, state_dict):
        """
        Load LoRA parameters from a state dict.
        
        Args:
            state_dict: State dict containing LoRA parameters
        """
        self.load_state_dict(state_dict, strict=False)

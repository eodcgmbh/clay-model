import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple, Optional
import segmentation_models_pytorch as smp
from torchmetrics.classification import F1Score, MulticlassJaccardIndex, ConfusionMatrix
# from torchsummary import summary
from torchview import draw_graph
from einops import rearrange

class EmbeddingSegmentationHead(nn.Module):
    """
    Segmentation head that takes Clay embeddings as input and outputs segmentation masks
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 target_size: Tuple[int, int],
                 num_classes: int,
                 patch_size: int=8,
                 hidden_dim: int=512,
                 C_out: int=64):
        """
        Args:
            embedding_dim: Dimension of Clay embeddings (e.g., 1024)
            embedding_size: Spatial size of embeddings (e.g., (8, 8))
            target_size: Target output size (e.g., (224, 224))
            num_classes: Number of segmentation classes
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        # self.embedding_size = embedding_size
        self.target_size = target_size
        self.num_classes = num_classes       
        self.patch_size = patch_size 
        
        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv_ps = nn.Conv2d(hidden_dim, C_out * patch_size * patch_size, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size) # consider diving this in several different steps and or  adding ASPP
        self.conv_out = nn.Conv2d(C_out, num_classes, kernel_size=3, padding=1)
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim, h_emb, w_emb)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes, target_h, target_w)
        """
        # x = self.fpn(embeddings) #aca tengo que llegar a [16,768,1024]

         # Reshape embeddings to [B, D, H', W']
        H_patches = self.target_size[0] // self.patch_size
        W_patches = self.target_size[1] // self.patch_size
        x = rearrange(embeddings, "B (H W) D -> B D H W", H=H_patches, W=W_patches)

        
        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_ps(x)  # [B, C_out * r^2, H', W']

        # Upsample using PixelShuffle
        x = self.pixel_shuffle(x)  # [B, C_out, H_in, W_in]

        # Final convolution to get desired output channels
        x = self.conv_out(x)  # [B, num_outputs, H_in, W_in]
            
        return x

class EmbeddingClassifier(L.LightningModule):
    """
    Lightning module for training a classifier on Clay embeddings
    """
    
    def __init__(self,
                 embedding_dim: int=1024,
                 patch_size: int=8,
                 target_size: Tuple[int, int] = (224, 224),
                 hidden_dim: int=512,
                 num_classes: int = 7,
                 lr: float = 1e-4,
                 wd: float = 1e-4,
                 b1: float=0.9,
                 b2: float=0.95,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            patch_size: Spatial size of embeddings
            target_size: Target output size (h_out, w_out)
            num_classes: Number of classes for segmentation
            hidden_dims: Hidden dimensions for decoder
            lr: Learning rate
            wd: Weight decay for optimizer
            class_weights: Optional class weights for loss function
        """
        super().__init__()
        self.save_hyperparameters()  
        
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.lr = lr
        self.wd = wd
        
        # Create segmentation head
        self.model = EmbeddingSegmentationHead(
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            target_size=target_size,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        self.loss_fn = smp.losses.FocalLoss(mode="multiclass", ignore_index=-1)
        self.iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="weighted",
            ignore_index=-1,
        )
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            ignore_index=-1,
        )

        self._confusion_matrix = ConfusionMatrix(
            num_classes=num_classes,
            task="multiclass",
            ignore_index=-1,
        )
        
        # summary(self.model, ((target_size[0] // patch_size) * (target_size[1] // patch_size), embedding_dim))
        self.model_graph = draw_graph(
            self.model, 
            input_size=(1, (target_size[0] // patch_size) * (target_size[1] // patch_size), embedding_dim),
            expand_nested=True,
            graph_name='EmbeddingSegmentationHead'
        )
        
        print(f"🏗️  EmbeddingClassifier initialized:")
        print(f"   Input: {embedding_dim} channels, {patch_size} spatial")
        print(f"   Output: {num_classes} classes, {target_size} spatial")
    
    def forward(self, batch):
        """
        Forward pass
        
        Args:
            batch: dictionary with labels and embeddings
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes, target_h, target_w)
        """
        return self.model(batch["embedding"])
    
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        labels = batch["label"].long()
        outputs = self(batch)
        # print(outputs.shape, labels.shape)
        # outputs = F.interpolate(
        #     outputs,
        #     size=self.target_size,
        #     mode="bilinear",
        #     align_corners=False,
        # )  # Resize to match labels size
        # print(outputs.shape, labels.shape)


        loss = self.loss_fn(outputs, labels)
        iou = self.iou(outputs, labels)
        f1 = self.f1(outputs, labels)

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if phase=="test":
            self._confusion_matrix.update(torch.argmax(outputs, dim=1), labels)

        return loss
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    def on_test_epoch_end(self):
        # Compute and log confusion matrix
        self.confusion_matrix = self._confusion_matrix.compute()
        
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = torch.optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
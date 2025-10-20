# finetune/segment/embedding_extractor.py
import torch
import lightning as L
import numpy as np
from pathlib import Path
from claymodel.module import ClayMAEModule

class ClayEmbeddingExtractor(L.LightningModule):
    """
    Lightning module to extract embeddings from Clay encoder
    """
    
    def __init__(self, ckpt_path, freeze_encoder=True, cls=True):
        """
        Args:
            ckpt_path: Path to the Clay model checkpoint
            freeze_encoder: Whether to freeze the encoder (True for inference)
            cls: Wether to return the cls embedding or all the other
        """
        super().__init__()

        self.cls = cls
        # if cls:
        #     mask_ratio = 0.0
        # else:
        #     mask_ratio = 0.0
        
        # Load the Clay model
        self.clay_encoder = ClayMAEModule.load_from_checkpoint(
            ckpt_path,
            strict=False,
            model_size="large",
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1]*7,
            mask_ratio=0.0,
            shuffle=False
        ).model.encoder
        
        if freeze_encoder:
            # Freeze all parameters for inference
            for param in self.clay_encoder.parameters():
                param.requires_grad = False
            self.clay_encoder.eval()
        
        print(f"✅ Clay model loaded from: {ckpt_path}")
        print(f"🔒 Encoder frozen: {freeze_encoder}")
    
    def forward(self, x):
        """
        Forward pass through Clay encoder to get embeddings
        
        Args:
            x: Input tensor of shape (batch_size, t, channels, height, width)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim, height', width')
        """
        # Extract embeddings from Clay encoder
        # The Clay model returns embeddings from the encoder
        embeddings = []
        for t in range(x["pixels"].shape[1]):

            x_t = x.copy()
            x_t["pixels"] = x["pixels"][:,t,...]
            x_t["time"] = x["times"][:,t,...]

            with torch.no_grad():
                unmsk_patch, *_ = self.clay_encoder(x_t) 

            if self.cls:
                embeddings_t = unmsk_patch[:, 0, :]
            else:
                embeddings_t = unmsk_patch[:, 1:, :]

            embeddings_t = embeddings_t.cpu().numpy()
            embeddings.append(embeddings_t)

        embeddings = np.stack(embeddings, axis=1)  # Shape: (batch_size, time, embedding_dim, height', width')
        
        return embeddings
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step that extracts embeddings
        """
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch
            y = None
        
        # Get embeddings
        embeddings = self(x)
        
        return {
            'embeddings': embeddings,
            'batch_idx': batch_idx,
            'input_shape': x["pixels"].shape,
            'embedding_shape': embeddings.shape,
            'chip_names': batch["chip_name"]
        }
    
    def configure_optimizers(self):
        """Not needed for inference, but Lightning requires this method"""
        return torch.optim.Adam(self.parameters(), lr=1e-3)
# finetune/segment/embedding_extractor.py
import torch
import lightning as L
import numpy as np
from pathlib import Path
from terratorch.registry import BACKBONE_REGISTRY
import torchview

class PrithviEmbeddingExtractor(L.LightningModule):
    """
    Lightning module to extract embeddings from Clay encoder
    """
    
    def __init__(self, ckpt_path, freeze_encoder=True, cls=True, bands=["BLUE", "GREEN", "RED", "NIR_NARROW"], indices=[5, 11, 17, 23]):
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

        # Minimal backbone args for Prithvi

        self.bands = bands
        self.num_frames = 6
        self.indices = indices

        model = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300_tl",
            pretrained=True,
            bands=self.bands,
            coords_encoding=["time", "location"],
            num_frames=self.num_frames,
            img_size=(256,256),
            )

        # Extract the encoder (backbone)
        self.encoder = model

        self.model_graph = torchview.draw_graph(
                self.encoder, 
                input_size=(1, 4, 6, 256, 256), #[batch, channels, time, height, width]
                expand_nested=True,
                graph_name='Prithvi_encoder'
            )
        
        if freeze_encoder:
            # Freeze all parameters for inference
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
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
        # for t in range(x["pixels"].shape[1]):

        #     x_t = x.copy()
        #     x_t["pixels"] = x["pixels"][:,t,...]
        #     x_t["time"] = x["times"][:,t,...]

        with torch.no_grad():
            # unmsk_patch, *_ = self.encoder(x_t) 
            output = self.encoder(
                x["pixels"].swapdims(1, 2),
                temporal_coords=x["times"],
                location_coords=x["latlon"]
                )  # Shape: (batch_size, embedding_dim, height', width')
            
        patch = torch.tensor(output[-1])  # Select specified indices

        if self.cls:
            embeddings = patch[:, 0, :]
        else:
            embeddings = patch[:, 1:, :]

        # embeddings_t = embeddings_t.cpu().numpy()
        # embeddings.append(embeddings_t)

        # embeddings = np.stack(embeddings, axis=1)  # Shape: (batch_size, time, embedding_dim, height', width')
        
        
        return embeddings.cpu().numpy().reshape(embeddings.shape[0], self.num_frames, -1, embeddings.shape[-1])  # Shape: (time, embedding_patches, embedding_dim)
    
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
# finetune/segment/embedding_callback.py
import torch
import lightning as L
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class EmbeddingSaveCallback(L.Callback):
    """
    Callback to save embeddings as .npy files
    """
    
    def __init__(self, 
                 save_metadata=True,
                 compression=True,
                 compressed_quantization=True,
                 model_path=""):
        """
        Args:
            output_dir: Directory to save embeddings
            save_metadata: Whether to save metadata about embeddings
            compression: Whether to use compression when saving .npy files
            compressed_quantization: Whether to use float16 instead of float32
        """
        self.output_dir = None  # Will be set in predict
        self.save_metadata = save_metadata
        self.compression = compression
        self.compressed_quantization = compressed_quantization
        self.model_path = model_path
        self.batch_count = 0
        self.total_samples = 0
        
        # Metadata storage
        self.metadata = {
            'start_time': None,
            'end_time': None,
            'total_batches': 0,
            'total_samples': 0,
            'embedding_shapes': [],
            'input_shapes': [],
            'files_created': []
        }
        
        if compression:
            print("🗜️  Using compressed .npz format")
        if compressed_quantization:
            print("🗜️  Saving in float16")
    
    def on_predict_start(self, trainer, pl_module):
        """Called when prediction starts"""
        self.metadata['start_time'] = datetime.now().isoformat()
        print("🚀 Starting embedding extraction...")
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Save embeddings after each batch"""
        
        embeddings = outputs['embeddings']  # Shape: (batch_size, embed_dim, h, w)
        batch_size = embeddings.shape[0]

        if self.output_dir is None:
            self.output_dir = Path(batch["chip_dir"][0]) / Path("embeddings_" + self.model_path)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving embeddings for batch {batch_idx} (shape: {embeddings.shape})")
        
        # Convert to numpy
        if self.compressed_quantization:
            # embeddings_np = embeddings.half().cpu().numpy()
            embeddings_np = embeddings.astype(np.float16)
        else:
            # embeddings_np = embeddings.cpu().numpy()
            embeddings_np = embeddings
        
        # Save embeddings
        if self.compression:
            # Save as compressed .npz file
            filename = f"embeddings_batch_{batch_idx:04d}.npz"
            filepath = self.output_dir / filename
            np.savez_compressed(filepath, embeddings=embeddings_np)

            self.metadata['files_created'].append(str(filename))
            self.metadata['embedding_shapes'].append(list(embeddings.shape))
            self.metadata['input_shapes'].append(list(outputs['input_shape']))
        else:
            # Save as different .npy files
            for i in range(embeddings.shape[0]):            
                filename = batch["chip_name"][i].replace(".npy", "_emb.npy").replace(".tif", "")
                filepath = self.output_dir / filename
                np.save(filepath, embeddings_np[i])
        
                self.metadata['files_created'].append(str(filename))
                self.metadata['embedding_shapes'].append(list(embeddings_np[i].shape))
                self.metadata['input_shapes'].append(list(outputs['input_shape'][1:]))

        self.batch_count += 1
        self.total_samples += batch_size
                    
    def on_predict_end(self, trainer, pl_module):
        """Called when prediction ends"""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_batches'] = self.batch_count
        self.metadata['total_samples'] = self.total_samples
        self.metadata["compressed_quantization_f16"] = self.compressed_quantization
        
        if self.save_metadata:
            metadata_file = self.output_dir / "embedding_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            summary_file = self.output_dir / "embedding_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Embedding Extraction Summary\n")
                f.write(f"===========================\n\n")
                f.write(f"Start time: {self.metadata['start_time']}\n")
                f.write(f"End time: {self.metadata['end_time']}\n")
                f.write(f"Total batches processed: {self.metadata['total_batches']}\n")
                f.write(f"Total samples processed: {self.metadata['total_samples']}\n")
                f.write(f"Files created: {len(self.metadata['files_created'])}\n")
                f.write(f"Output directory: {self.output_dir}\n")
                f.write(f"Compressed quantization used: {self.compressed_quantization}\n\n")
                f.write(f"Compression used: {self.compression}\n\n")

                
                if self.metadata['embedding_shapes']:
                    f.write(f"Embedding shapes:\n")
                    unique_shapes = list(set(map(tuple, self.metadata['embedding_shapes'])))
                    for shape in unique_shapes:
                        count = sum(1 for s in self.metadata['embedding_shapes'] if tuple(s) == shape)
                        f.write(f"  {shape}: {count} batches\n")
        
        print(f"✅ Embedding extraction completed!")
        print(f"📊 Total batches: {self.batch_count}")
        print(f"📊 Total samples: {self.total_samples}")
        print(f"📁 Results saved to: {self.output_dir}")

        self.output_dir = None

class EmbeddingLoaderCallback(L.Callback):
    """
    Optional callback to demonstrate how to load saved embeddings
    """
    
    @staticmethod
    def load_embeddings(embeddings_dir, batch_idx=None, compressed=True):
        """
        Load embeddings from saved files
        
        Args:
            embeddings_dir: Directory containing saved embeddings
            batch_idx: Specific batch to load (None for all)
            compressed: Whether files are compressed (.npz) or not (.npy)
            
        Returns:
            embeddings: numpy array or list of arrays
        """
        embeddings_dir = Path(embeddings_dir)
        
        if batch_idx is not None:
            # Load specific batch
            if compressed:
                filepath = embeddings_dir / f"embeddings_batch_{batch_idx:04d}.npz"
                data = np.load(filepath)
                return data['embeddings']
            else:
                filepath = embeddings_dir / f"embeddings_batch_{batch_idx:04d}.npy"
                return np.load(filepath)
        else:
            # Load all batches
            embeddings_list = []
            if compressed:
                pattern = "embeddings_batch_*.npz"
                files = sorted(embeddings_dir.glob(pattern))
                for file in files:
                    data = np.load(file)
                    embeddings_list.append(data['embeddings'])
            else:
                pattern = "embeddings_batch_*.npy"
                files = sorted(embeddings_dir.glob(pattern))
                for file in files:
                    embeddings_list.append(np.load(file))
            
            return embeddings_list
    
    @staticmethod
    def load_metadata(embeddings_dir):
        """Load metadata about the embeddings"""
        metadata_file = Path(embeddings_dir) / "embedding_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
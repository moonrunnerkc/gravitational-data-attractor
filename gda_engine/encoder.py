"""
Multimodal encoder for converting text, image, and audio inputs to mass vectors.
Implements lightweight, efficient encoding strategies for each modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ModalityEncoder(nn.Module):
    """Base encoder class for all modalities."""
    
    def __init__(self, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        
    def encode_to_masses(self, x):
        """Convert encoded features to mass vectors."""
        # Normalize to unit sphere
        x_norm = F.normalize(x, p=2, dim=-1)
        
        # Add mass dimension (learned from magnitude)
        masses = torch.norm(x, dim=-1, keepdim=True) / 10.0
        masses = torch.clamp(masses, min=0.01, max=1.0)
        
        return x_norm * self.output_dim, masses


class TextEncoder(ModalityEncoder):
    """Lightweight text encoder using character-level convolutions."""
    
    def __init__(self, vocab_size=256, embed_dim=64, output_dim=128):
        super().__init__(output_dim)
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Multi-scale convolutions for feature extraction
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=7, padding=3)
        
        # Compression layers
        self.compress = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Position encoding
        self.register_buffer('pos_encoding', self._create_position_encoding(1000, embed_dim))
        
    def _create_position_encoding(self, max_len, d_model):
        """Create sinusoidal position encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, text_input):
        """
        Encode text to mass vectors.
        
        Args:
            text_input: Tensor of character indices (B, L) or raw text strings
            
        Returns:
            Mass vectors (B, N, D) and masses (B, N, 1)
        """
        if isinstance(text_input, list):
            # Convert strings to tensor
            text_input = self._text_to_tensor(text_input)
        
        batch_size, seq_len = text_input.shape
        
        # Embed characters
        x = self.char_embed(text_input)  # (B, L, E)
        
        # Add position encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply convolutions
        x_t = x.transpose(1, 2)  # (B, E, L)
        
        c1 = F.relu(self.conv1(x_t))
        c2 = F.relu(self.conv2(x_t))
        c3 = F.relu(self.conv3(x_t))
        
        # Concatenate multi-scale features
        features = torch.cat([c1, c2, c3], dim=1)  # (B, 384, L)
        
        # Global pooling
        features_mean = features.mean(dim=2)
        features_max = features.max(dim=2)[0]
        
        # Compress to output dimension
        global_features = self.compress(features_mean)
        
        # Generate multiple mass points from local features
        # Sample at regular intervals
        num_masses = min(10, seq_len)
        indices = torch.linspace(0, seq_len-1, num_masses).long()
        
        local_features = features[:, :, indices].transpose(1, 2)  # (B, N, 384)
        local_masses = self.compress(local_features)  # (B, N, D)
        
        # Combine global and local representations
        all_masses = torch.cat([global_features.unsqueeze(1), local_masses], dim=1)
        
        return self.encode_to_masses(all_masses)
    
    def _text_to_tensor(self, texts):
        """Convert text strings to character index tensors."""
        max_len = max(len(t) for t in texts)
        
        # Create padded tensor
        tensor = torch.zeros(len(texts), max_len, dtype=torch.long)
        
        for i, text in enumerate(texts):
            chars = [ord(c) % 256 for c in text]  # Simple ASCII encoding
            tensor[i, :len(chars)] = torch.tensor(chars)
        
        return tensor


class ImageEncoder(ModalityEncoder):
    """Lightweight image encoder using strided convolutions."""
    
    def __init__(self, input_channels=3, output_dim=128):
        super().__init__(output_dim)
        
        # Efficient downsampling with strided convolutions
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(input_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Spatial attention for selecting important regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Project to output dimension
        self.projection = nn.Linear(512, output_dim)
        
    def forward(self, image_input):
        """
        Encode images to mass vectors.
        
        Args:
            image_input: Tensor of images (B, C, H, W)
            
        Returns:
            Mass vectors (B, N, D) and masses (B, N, 1)
        """
        batch_size = image_input.shape[0]
        
        # Extract features
        features = self.encoder(image_input)  # (B, 512, 7, 7)
        
        # Compute spatial attention
        attention = self.spatial_attention(features)  # (B, 1, 7, 7)
        
        # Apply attention
        attended_features = features * attention
        
        # Extract mass points from spatial locations
        # Flatten spatial dimensions
        features_flat = attended_features.view(batch_size, 512, -1).transpose(1, 2)  # (B, 49, 512)
        
        # Select top-k attended regions
        attention_flat = attention.view(batch_size, -1)  # (B, 49)
        k = min(10, attention_flat.shape[1])
        _, top_indices = torch.topk(attention_flat, k, dim=1)
        
        # Gather top features
        top_features = torch.gather(
            features_flat, 
            1, 
            top_indices.unsqueeze(-1).expand(-1, -1, 512)
        )  # (B, k, 512)
        
        # Project to output dimension
        mass_vectors = self.projection(top_features)  # (B, k, D)
        
        # Add global representation
        global_features = features.mean(dim=[2, 3])  # (B, 512)
        global_mass = self.projection(global_features).unsqueeze(1)  # (B, 1, D)
        
        all_masses = torch.cat([global_mass, mass_vectors], dim=1)
        
        return self.encode_to_masses(all_masses)


class AudioEncoder(ModalityEncoder):
    """Lightweight audio encoder using 1D convolutions."""
    
    def __init__(self, sample_rate=16000, output_dim=128):
        super().__init__(output_dim)
        self.sample_rate = sample_rate
        
        # Waveform encoder
        self.waveform_encoder = nn.Sequential(
            # Initial convolution
            nn.Conv1d(1, 64, 80, stride=16),  # ~1000Hz features
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Downsample
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(512, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Softmax(dim=-1)
        )
        
        # Project to output
        self.projection = nn.Linear(512, output_dim)
        
    def forward(self, audio_input):
        """
        Encode audio to mass vectors.
        
        Args:
            audio_input: Tensor of audio waveforms (B, L) or (B, 1, L)
            
        Returns:
            Mass vectors (B, N, D) and masses (B, N, 1)
        """
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(1)
        
        batch_size = audio_input.shape[0]
        
        # Extract features
        features = self.waveform_encoder(audio_input)  # (B, 512, T)
        
        # Compute temporal attention
        attention = self.temporal_attention(features)  # (B, 1, T)
        
        # Apply attention
        attended_features = features * attention
        
        # Extract mass points from time steps
        features_t = features.transpose(1, 2)  # (B, T, 512)
        
        # Sample at regular intervals
        num_masses = min(10, features_t.shape[1])
        indices = torch.linspace(0, features_t.shape[1]-1, num_masses).long()
        
        sampled_features = features_t[:, indices, :]  # (B, num_masses, 512)
        mass_vectors = self.projection(sampled_features)
        
        # Add global representation
        global_features = (features * attention).sum(dim=2) / attention.sum(dim=2)  # (B, 512)
        global_mass = self.projection(global_features).unsqueeze(1)  # (B, 1, D)
        
        all_masses = torch.cat([global_mass, mass_vectors], dim=1)
        
        return self.encode_to_masses(all_masses)


class MultiModalEncoder(nn.Module):
    """Unified encoder for all modalities."""
    
    def __init__(self, output_dim=128):
        super().__init__()
        
        self.text_encoder = TextEncoder(output_dim=output_dim)
        self.image_encoder = ImageEncoder(output_dim=output_dim)
        self.audio_encoder = AudioEncoder(output_dim=output_dim)
        
        # Modality embeddings for identification
        self.modality_embeddings = nn.Embedding(3, output_dim)
        
    def forward(self, inputs, modality='auto'):
        """
        Encode inputs based on modality.
        
        Args:
            inputs: Input data
            modality: 'text', 'image', 'audio', or 'auto'
            
        Returns:
            Mass vectors and masses
        """
        if modality == 'auto':
            modality = self._detect_modality(inputs)
        
        if modality == 'text':
            masses, weights = self.text_encoder(inputs)
            mod_id = 0
        elif modality == 'image':
            masses, weights = self.image_encoder(inputs)
            mod_id = 1
        elif modality == 'audio':
            masses, weights = self.audio_encoder(inputs)
            mod_id = 2
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Add modality embedding
        mod_embedding = self.modality_embeddings(
            torch.tensor([mod_id], device=masses.device)
        )
        masses = masses + mod_embedding.unsqueeze(1) * 0.1
        
        return masses, weights
    
    def _detect_modality(self, inputs):
        """Auto-detect input modality based on shape."""
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            return 'text'
        elif isinstance(inputs, torch.Tensor):
            if inputs.dim() == 2:
                return 'text' if inputs.dtype == torch.long else 'audio'
            elif inputs.dim() == 3:
                return 'audio'
            elif inputs.dim() == 4:
                return 'image'
        
        raise ValueError("Could not auto-detect modality")
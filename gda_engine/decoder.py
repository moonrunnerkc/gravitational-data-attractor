"""
Decoder for reconstructing inputs from latent attractor representations.
Implements inverse mapping from compressed attractor fields back to original modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ModalityDecoder(nn.Module):
    """Base decoder class for all modalities."""
    
    def __init__(self, input_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
    def expand_from_attractors(self, attractor_field, target_shape):
        """Expand compressed attractor representation to target shape."""
        # Use learned expansion or interpolation
        raise NotImplementedError


class TextDecoder(ModalityDecoder):
    """Decode attractor fields back to text representations."""
    
    def __init__(self, input_dim=128, vocab_size=256, max_length=1000):
        super().__init__(input_dim)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Expansion network
        self.expand = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        
        # Sequence generation with LSTM
        self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(512, vocab_size)
        
        # Learned start token
        self.start_embedding = nn.Parameter(torch.randn(1, 512))
        
    def forward(self, attractor_field, target_length=None):
        """
        Decode attractor field to text.
        
        Args:
            attractor_field: Compressed representation (B, D)
            target_length: Desired output length (optional)
            
        Returns:
            Logits for character prediction (B, L, V)
        """
        batch_size = attractor_field.shape[0]
        
        # Expand attractor field
        expanded = self.expand(attractor_field)  # (B, 512)
        
        # Initialize hidden state with expanded field
        h_0 = expanded.unsqueeze(0).repeat(2, 1, 1)  # (2, B, 512)
        c_0 = torch.zeros_like(h_0)
        
        # Determine sequence length
        if target_length is None:
            # Predict length from attractor field
            length_features = self.expand(attractor_field)
            predicted_lengths = torch.clamp(
                torch.norm(length_features, dim=-1) * 10,
                min=10, max=self.max_length
            ).long()
            target_length = predicted_lengths.max().item()
        
        # Generate sequence
        outputs = []
        input_t = self.start_embedding.expand(batch_size, 1, -1)
        
        hidden = (h_0, c_0)
        
        for t in range(target_length):
            # LSTM step
            output_t, hidden = self.lstm(input_t, hidden)
            
            # Project to vocabulary
            logits_t = self.output_proj(output_t)
            outputs.append(logits_t)
            
            # Teacher forcing during training, argmax during inference
            if self.training:
                # Use expanded field as input (simulating teacher forcing)
                input_t = expanded.unsqueeze(1) + torch.randn_like(expanded.unsqueeze(1)) * 0.1
            else:
                # Greedy decoding
                pred_t = logits_t.argmax(dim=-1)
                input_t = self.output_proj.weight[pred_t]  # (B, 1, 512)
        
        # Stack outputs
        output_logits = torch.cat(outputs, dim=1)  # (B, L, V)
        
        return output_logits
    
    def decode_to_text(self, attractor_field):
        """Decode attractor field to actual text strings."""
        with torch.no_grad():
            logits = self.forward(attractor_field)
            predictions = logits.argmax(dim=-1)
            
            texts = []
            for pred in predictions:
                chars = [chr(c.item()) for c in pred if c < 256]
                texts.append(''.join(chars))
            
            return texts


class ImageDecoder(ModalityDecoder):
    """Decode attractor fields back to images."""
    
    def __init__(self, input_dim=128, output_channels=3, image_size=224):
        super().__init__(input_dim)
        
        self.output_channels = output_channels
        self.image_size = image_size
        
        # Initial expansion
        self.initial_size = 7
        self.expand = nn.Sequential(
            nn.Linear(input_dim, 512 * self.initial_size * self.initial_size),
            nn.ReLU(),
        )
        
        # Transposed convolutions for upsampling
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
    def forward(self, attractor_field):
        """
        Decode attractor field to image.
        
        Args:
            attractor_field: Compressed representation (B, D)
            
        Returns:
            Reconstructed images (B, C, H, W)
        """
        batch_size = attractor_field.shape[0]
        
        # Expand to spatial representation
        expanded = self.expand(attractor_field)
        spatial = expanded.view(batch_size, 512, self.initial_size, self.initial_size)
        
        # Decode through transposed convolutions
        reconstructed = self.decoder(spatial)
        
        # Ensure correct output size
        if reconstructed.shape[-1] != self.image_size:
            reconstructed = F.interpolate(
                reconstructed, 
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        return reconstructed


class AudioDecoder(ModalityDecoder):
    """Decode attractor fields back to audio waveforms."""
    
    def __init__(self, input_dim=128, sample_rate=16000, max_duration=5.0):
        super().__init__(input_dim)
        
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        
        # Initial expansion
        self.expand = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )
        
        # Waveform generation network
        self.decoder = nn.Sequential(
            # Upsample progressively
            nn.ConvTranspose1d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, 16, stride=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 1, 16, stride=8, padding=4),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
    def forward(self, attractor_field, target_length=None):
        """
        Decode attractor field to audio waveform.
        
        Args:
            attractor_field: Compressed representation (B, D)
            target_length: Desired output length in samples
            
        Returns:
            Reconstructed waveforms (B, 1, L)
        """
        batch_size = attractor_field.shape[0]
        
        # Expand attractor field
        expanded = self.expand(attractor_field)  # (B, 1024)
        
        # Reshape for convolutions
        expanded = expanded.unsqueeze(2)  # (B, 1024, 1)
        
        # Generate waveform
        waveform = self.decoder(expanded)
        
        # Adjust length if needed
        if target_length is not None:
            current_length = waveform.shape[-1]
            if current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                waveform = F.pad(waveform, (0, padding))
            elif current_length > target_length:
                # Truncate
                waveform = waveform[:, :, :target_length]
        
        return waveform


class MultiModalDecoder(nn.Module):
    """Unified decoder for all modalities."""
    
    def __init__(self, input_dim=128):
        super().__init__()
        
        self.text_decoder = TextDecoder(input_dim=input_dim)
        self.image_decoder = ImageDecoder(input_dim=input_dim)
        self.audio_decoder = AudioDecoder(input_dim=input_dim)
        
        # Modality classifier to determine output type
        self.modality_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 modalities
            nn.Softmax(dim=-1)
        )
        
    def forward(self, attractor_field, target_modality='auto', **kwargs):
        """
        Decode attractor field based on target modality.
        
        Args:
            attractor_field: Compressed representation (B, D)
            target_modality: 'text', 'image', 'audio', or 'auto'
            **kwargs: Additional arguments for specific decoders
            
        Returns:
            Reconstructed output in specified modality
        """
        if target_modality == 'auto':
            # Predict modality from attractor field
            modality_probs = self.modality_classifier(attractor_field)
            target_modality_idx = modality_probs.argmax(dim=-1)
            
            # Handle batch with potentially different modalities
            outputs = []
            for i, mod_idx in enumerate(target_modality_idx):
                field_i = attractor_field[i:i+1]
                
                if mod_idx == 0:
                    output = self.text_decoder(field_i, **kwargs)
                elif mod_idx == 1:
                    output = self.image_decoder(field_i)
                else:
                    output = self.audio_decoder(field_i, **kwargs)
                
                outputs.append(output)
            
            return outputs, target_modality_idx
        
        else:
            # Decode to specific modality
            if target_modality == 'text':
                return self.text_decoder(attractor_field, **kwargs)
            elif target_modality == 'image':
                return self.image_decoder(attractor_field)
            elif target_modality == 'audio':
                return self.audio_decoder(attractor_field, **kwargs)
            else:
                raise ValueError(f"Unknown modality: {target_modality}")
    
    def get_modality_probabilities(self, attractor_field):
        """Get probability distribution over modalities for attractor field."""
        return self.modality_classifier(attractor_field)
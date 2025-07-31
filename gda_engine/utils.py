"""
Shared utility functions for GDA engine.
Includes math operations, normalization, masking, and helper functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def normalize_vectors(vectors, dim=-1, eps=1e-8):
    """
    Normalize vectors to unit length.
    
    Args:
        vectors: Input tensors
        dim: Dimension along which to normalize
        eps: Small value for numerical stability
        
    Returns:
        Normalized vectors
    """
    norm = torch.norm(vectors, p=2, dim=dim, keepdim=True)
    return vectors / (norm + eps)


def create_padding_mask(lengths, max_length):
    """
    Create padding mask for variable length sequences.
    
    Args:
        lengths: Tensor of actual lengths (B,)
        max_length: Maximum sequence length
        
    Returns:
        Boolean mask (B, max_length) where True indicates valid positions
    """
    batch_size = lengths.shape[0]
    mask = torch.arange(max_length, device=lengths.device).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return mask


def apply_rotary_embedding(x, dim, max_length=5000):
    """
    Apply rotary position embeddings.
    
    Args:
        x: Input tensor (B, L, D)
        dim: Embedding dimension
        max_length: Maximum sequence length
        
    Returns:
        Tensor with rotary embeddings applied
    """
    device = x.device
    seq_len = x.shape[1]
    
    # Create position indices
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    
    # Create frequency bands
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * -(np.log(10000.0) / dim)
    )
    
    # Apply sin/cos
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Apply rotation
    x_rot = x * pe.unsqueeze(0)
    
    return x_rot


def compute_pairwise_distances(x, y=None, squared=False):
    """
    Compute pairwise distances between sets of vectors.
    
    Args:
        x: First set of vectors (B, N, D)
        y: Second set of vectors (B, M, D), if None uses x
        squared: Return squared distances
        
    Returns:
        Distance matrix (B, N, M)
    """
    if y is None:
        y = x
    
    # Compute squared distances
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)
    
    distances_squared = x_norm + y_norm - 2 * torch.matmul(x, y.transpose(-2, -1))
    distances_squared = torch.clamp(distances_squared, min=0.0)
    
    if squared:
        return distances_squared
    else:
        return torch.sqrt(distances_squared)


def sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax sampling.
    
    Args:
        logits: Unnormalized log probabilities
        temperature: Temperature parameter
        hard: Return one-hot encoded samples
        
    Returns:
        Soft or hard samples
    """
    gumbels = sample_gumbel(logits.shape, logits.device)
    y_soft = F.softmax((logits + gumbels) / temperature, dim=-1)
    
    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def masked_mean(tensor, mask, dim=None, keepdim=False):
    """
    Compute mean over masked values.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask
        dim: Dimension(s) to reduce
        keepdim: Keep reduced dimensions
        
    Returns:
        Masked mean
    """
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    
    masked_tensor = tensor * mask.float()
    sum_tensor = masked_tensor.sum(dim=dim, keepdim=keepdim)
    count = mask.float().sum(dim=dim, keepdim=keepdim)
    
    return sum_tensor / (count + 1e-8)


def stable_softmax(x, dim=-1, temperature=1.0):
    """Numerically stable softmax with temperature."""
    x_scaled = x / temperature
    x_max = x_scaled.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x_scaled - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: Quaternions (B, 4) in (w, x, y, z) format
        
    Returns:
        Rotation matrices (B, 3, 3)
    """
    w, x, y, z = quaternions.unbind(-1)
    
    matrix = torch.stack([
        torch.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)
        ], dim=-1),
        torch.stack([
            2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)
        ], dim=-1),
        torch.stack([
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)
        ], dim=-1)
    ], dim=-2)
    
    return matrix


def apply_3d_rotation(points, rotation_matrix):
    """
    Apply 3D rotation to points.
    
    Args:
        points: 3D points (B, N, 3)
        rotation_matrix: Rotation matrices (B, 3, 3)
        
    Returns:
        Rotated points (B, N, 3)
    """
    return torch.matmul(points, rotation_matrix.transpose(-2, -1))


def compute_entropy(probs, dim=-1, eps=1e-8):
    """Compute entropy of probability distributions."""
    return -(probs * torch.log(probs + eps)).sum(dim=dim)


def reparameterize(mu, log_var):
    """
    Reparameterization trick for VAE.
    
    Args:
        mu: Mean
        log_var: Log variance
        
    Returns:
        Sampled values
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, log_var):
    """Compute KL divergence from N(0, 1)."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)


def cosine_similarity_matrix(x, y=None):
    """Compute cosine similarity matrix between sets of vectors."""
    x_norm = F.normalize(x, p=2, dim=-1)
    if y is None:
        y_norm = x_norm
    else:
        y_norm = F.normalize(y, p=2, dim=-1)
    
    return torch.matmul(x_norm, y_norm.transpose(-2, -1))


def top_k_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: Logits distribution
        top_k: Keep only top k tokens
        top_p: Keep tokens with cumulative probability >= top_p
        filter_value: Value to assign filtered positions
        
    Returns:
        Filtered logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class EarlyStopping:
    """Early stopping handler for training."""
    
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
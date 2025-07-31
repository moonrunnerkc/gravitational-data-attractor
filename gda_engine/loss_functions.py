"""
Enhanced loss functions with meta-learning and physics-informed regularization.
Combines gravitational physics constraints with adaptive multi-objective optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MetaLearningWeights(nn.Module):
    """Meta-learning module for adaptive loss weighting."""
    
    def __init__(self, num_losses=4, hidden_dim=64):
        super().__init__()
        
        # Meta network for weight prediction
        self.meta_net = nn.Sequential(
            nn.Linear(num_losses * 3, hidden_dim),  # loss value, gradient, variance
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_losses),
            nn.Softmax(dim=-1)
        )
        
        # Running statistics
        self.register_buffer('loss_history', torch.zeros(100, num_losses))
        self.register_buffer('gradient_history', torch.zeros(100, num_losses))
        self.history_idx = 0
        
    def forward(self, current_losses, gradients=None):
        """
        Predict adaptive weights based on loss dynamics.
        
        Args:
            current_losses: Dict of current loss values
            gradients: Optional dict of loss gradients
            
        Returns:
            Adaptive weights for each loss component
        """
        # Extract loss values
        loss_values = torch.stack([
            current_losses.get('reconstruction', torch.tensor(0.0)),
            current_losses.get('gravitational', torch.tensor(0.0)),
            current_losses.get('contrastive', torch.tensor(0.0)),
            current_losses.get('compression', torch.tensor(0.0))
        ])
        
        # Compute statistics
        loss_mean = self.loss_history.mean(dim=0)
        loss_std = self.loss_history.std(dim=0) + 1e-8
        
        # Normalize current losses
        normalized_losses = (loss_values - loss_mean) / loss_std
        
        # Compute gradient magnitudes if available
        if gradients is not None:
            grad_values = torch.stack([
                gradients.get(k, torch.tensor(0.0)) for k in 
                ['reconstruction', 'gravitational', 'contrastive', 'compression']
            ])
        else:
            grad_values = torch.zeros_like(loss_values)
        
        # Create input features
        features = torch.cat([
            normalized_losses,
            grad_values,
            loss_std
        ])
        
        # Predict weights
        weights = self.meta_net(features)
        
        # Update history
        self.loss_history[self.history_idx] = loss_values.detach()
        self.history_idx = (self.history_idx + 1) % 100
        
        return weights


class PhysicsInformedRegularization(nn.Module):
    """Physics-informed regularization terms."""
    
    def __init__(self, energy_conservation_weight=0.1, 
                 momentum_conservation_weight=0.05):
        super().__init__()
        self.energy_weight = energy_conservation_weight
        self.momentum_weight = momentum_conservation_weight
        
    def forward(self, trajectory_data):
        """
        Compute physics-based regularization.
        
        Args:
            trajectory_data: Dict with positions, velocities, masses, energies
            
        Returns:
            Physics regularization loss
        """
        losses = {}
        
        # Energy conservation
        if 'energies' in trajectory_data and trajectory_data['energies']:
            energies = torch.tensor(trajectory_data['energies'])
            if len(energies) > 1:
                # Energy should decrease (second law of thermodynamics)
                energy_increases = F.relu(energies[1:] - energies[:-1])
                losses['energy_conservation'] = energy_increases.mean()
        
        # Momentum conservation
        if all(k in trajectory_data for k in ['positions', 'velocities', 'masses']):
            positions = trajectory_data['positions']
            velocities = trajectory_data.get('velocities', [])
            masses = trajectory_data['masses']
            
            if isinstance(positions, list) and len(positions) > 1 and len(velocities) > 1:
                # Compute total momentum at each step
                momenta = []
                for pos, vel in zip(positions, velocities):
                    if isinstance(vel, torch.Tensor) and isinstance(masses, torch.Tensor):
                        momentum = (masses.unsqueeze(-1) * vel).sum(dim=0)
                        momenta.append(momentum)
                
                if len(momenta) > 1:
                    momenta = torch.stack(momenta)
                    # Momentum change should be minimal
                    momentum_changes = torch.norm(momenta[1:] - momenta[:-1], dim=-1)
                    losses['momentum_conservation'] = momentum_changes.mean()
        
        # Combine physics losses
        total_loss = torch.tensor(0.0)
        if 'energy_conservation' in losses:
            total_loss += self.energy_weight * losses['energy_conservation']
        if 'momentum_conservation' in losses:
            total_loss += self.momentum_weight * losses['momentum_conservation']
        
        return total_loss, losses


class GravitationalLoss(nn.Module):
    """Enhanced gravitational loss with stability constraints."""
    
    def __init__(self, energy_weight=1.0, stability_weight=0.1, 
                 compactness_weight=0.1, physics_weight=0.05):
        super().__init__()
        self.energy_weight = energy_weight
        self.stability_weight = stability_weight
        self.compactness_weight = compactness_weight
        self.physics_weight = physics_weight
        
        # Physics regularization
        self.physics_reg = PhysicsInformedRegularization()
        
    def forward(self, attractor_states, convergence_info, trajectory_data=None):
        """
        Compute gravitational loss with physics constraints.
        
        Args:
            attractor_states: Final attractor field states
            convergence_info: Convergence metrics
            trajectory_data: Optional trajectory information
            
        Returns:
            Total gravitational loss and components
        """
        losses = {}
        
        # Handle multiple modalities
        if isinstance(convergence_info, dict):
            # Multi-modal case
            total_energy = 0
            total_iterations = 0
            all_converged = True
            
            for modality, info in convergence_info.items():
                energy_history = info.get('energy_history', [])
                if energy_history:
                    total_energy += energy_history[-1]
                
                total_iterations += info.get('iterations', 50)
                all_converged = all_converged and info.get('converged', False)
            
            losses['final_energy'] = torch.tensor(total_energy / len(convergence_info))
            converged = all_converged
            num_iterations = total_iterations / len(convergence_info)
        else:
            # Single modality case
            energy_history = convergence_info.get('energy_history', [])
            if energy_history:
                losses['final_energy'] = torch.tensor(energy_history[-1])
                
                # Energy should decrease monotonically
                if len(energy_history) > 1:
                    energy_smoothness = sum(
                        max(0, energy_history[i] - energy_history[i-1])
                        for i in range(1, len(energy_history))
                    ) / len(energy_history)
                    losses['energy_smoothness'] = torch.tensor(energy_smoothness)
            
            converged = convergence_info.get('converged', True)
            num_iterations = convergence_info.get('iterations', 50)
        
        # Stability loss
        stability_loss = torch.tensor(0.0)
        if not converged:
            stability_loss += 1.0
        
        # Penalize excessive iterations
        stability_loss += max(0, (num_iterations - 30) / 50)
        losses['stability'] = stability_loss
        
        # Compactness loss
        if isinstance(attractor_states, torch.Tensor):
            # Encourage tight clusters
            variance = torch.var(attractor_states, dim=-1).mean()
            
            # Also encourage minimum separation (avoid collapse)
            if attractor_states.dim() > 1 and attractor_states.shape[0] > 1:
                pairwise_distances = torch.pdist(attractor_states.view(-1, attractor_states.shape[-1]))
                if len(pairwise_distances) > 0:
                    min_distance = pairwise_distances.min()
                    
                    # Penalize if particles are too close
                    collapse_penalty = F.relu(0.01 - min_distance)
                    
                    losses['variance'] = variance
                    losses['collapse_penalty'] = collapse_penalty
                    compactness_loss = variance + collapse_penalty
                else:
                    losses['variance'] = variance
                    compactness_loss = variance
            else:
                losses['variance'] = variance
                compactness_loss = variance
        else:
            compactness_loss = torch.tensor(0.0)
        
        # Physics regularization
        physics_loss = torch.tensor(0.0)
        if trajectory_data is not None:
            physics_loss, physics_components = self.physics_reg(trajectory_data)
            losses.update({f'physics_{k}': v for k, v in physics_components.items()})
        
        # Combine losses
        total_loss = (
            self.energy_weight * losses.get('final_energy', torch.tensor(0.0)) +
            self.stability_weight * stability_loss +
            self.compactness_weight * compactness_loss +
            self.physics_weight * physics_loss
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


class TestTimeOptimization(nn.Module):
    """Test-time optimization for ARC-AGI style tasks."""
    
    def __init__(self, base_loss_fn, num_steps=10, lr=0.01):
        super().__init__()
        self.base_loss = base_loss_fn
        self.num_steps = num_steps
        self.lr = lr
        
    def forward(self, predictions, targets, model_params):
        """
        Optimize predictions at test time.
        
        Args:
            predictions: Initial predictions
            targets: Ground truth (may be partially known)
            model_params: Parameters to optimize
            
        Returns:
            Optimized predictions
        """
        # Create temporary parameters
        temp_params = [p.clone().detach().requires_grad_(True) 
                       for p in model_params if p.requires_grad]
        
        if not temp_params:
            return predictions
        
        optimizer = torch.optim.Adam(temp_params, lr=self.lr)
        
        best_loss = float('inf')
        best_predictions = predictions.clone()
        
        for step in range(self.num_steps):
            try:
                # Compute loss with temporary parameters
                loss = self.base_loss(predictions, targets)
                
                # Backward and update
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
                # Track best
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_predictions = predictions.clone()
            except:
                # If optimization fails, return original predictions
                break
        
        return best_predictions


class ReconstructionLoss(nn.Module):
    """Multi-modal reconstruction loss with perceptual components."""
    
    def __init__(self):
        super().__init__()
        
        # Modality-specific losses
        self.text_loss = nn.MSELoss(reduction='mean')  # Simplified for compatibility
        self.image_loss = nn.MSELoss()
        self.audio_loss = nn.L1Loss()
        
        # Perceptual loss for images (simplified)
        self.perceptual_weight = 0.1
        
    def forward(self, predictions, targets, modality):
        """
        Compute reconstruction loss with perceptual components.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            modality: Data modality
            
        Returns:
            Reconstruction loss
        """
        if modality == 'text':
            # Handle text reconstruction (simplified to MSE for compatibility)
            loss = self.text_loss(predictions, targets)
                
        elif modality == 'image':
            # Normalize targets if needed
            if targets.max() > 1.0:
                targets = (targets - 127.5) / 127.5
            
            # Pixel-wise loss
            pixel_loss = self.image_loss(predictions, targets)
            
            # Simple perceptual loss using gradients
            if predictions.shape[-1] > 1 and predictions.shape[-2] > 1:
                # Compute image gradients
                pred_dx = predictions[..., :, 1:] - predictions[..., :, :-1]
                pred_dy = predictions[..., 1:, :] - predictions[..., :-1, :]
                
                target_dx = targets[..., :, 1:] - targets[..., :, :-1]
                target_dy = targets[..., 1:, :] - targets[..., :-1, :]
                
                # Gradient loss
                grad_loss = (
                    F.l1_loss(pred_dx, target_dx) + 
                    F.l1_loss(pred_dy, target_dy)
                )
                
                loss = pixel_loss + self.perceptual_weight * grad_loss
            else:
                loss = pixel_loss
                
        elif modality == 'audio':
            # Time-domain loss
            time_loss = self.audio_loss(predictions, targets)
            
            # Simple frequency-domain loss
            if predictions.shape[-1] > 16:
                try:
                    # FFT
                    pred_fft = torch.fft.rfft(predictions, dim=-1)
                    target_fft = torch.fft.rfft(targets, dim=-1)
                    
                    # Magnitude spectrum loss
                    freq_loss = F.l1_loss(
                        pred_fft.abs(), 
                        target_fft.abs()
                    )
                    
                    loss = time_loss + 0.1 * freq_loss
                except:
                    loss = time_loss
            else:
                loss = time_loss
        else:
            # Default to MSE for unknown modalities
            loss = F.mse_loss(predictions, targets)
        
        return loss


class ContrastiveLoss(nn.Module):
    """Enhanced contrastive loss with multiple negatives and margin."""
    
    def __init__(self, temperature=0.07, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, embeddings_dict):
        """
        Compute multi-modal contrastive loss.
        
        Args:
            embeddings_dict: Dict of embeddings by modality
            
        Returns:
            Contrastive loss
        """
        modalities = list(embeddings_dict.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0)
        
        total_loss = torch.tensor(0.0)
        num_pairs = 0
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i < j:
                    embeddings1 = embeddings_dict[mod1]
                    embeddings2 = embeddings_dict[mod2]
                    
                    # Normalize
                    embeddings1 = F.normalize(embeddings1, p=2, dim=-1)
                    embeddings2 = F.normalize(embeddings2, p=2, dim=-1)
                    
                    # Compute similarity matrix
                    similarity = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
                    
                    batch_size = embeddings1.shape[0]
                    labels = torch.arange(batch_size, device=embeddings1.device)
                    
                    # InfoNCE loss
                    loss_12 = F.cross_entropy(similarity, labels)
                    loss_21 = F.cross_entropy(similarity.t(), labels)
                    
                    # Triplet margin loss for additional constraint
                    pos_similarity = similarity[torch.arange(batch_size), labels]
                    
                    # Hard negative mining
                    mask = torch.eye(batch_size, device=similarity.device).bool()
                    neg_similarity = similarity.masked_fill(mask, -float('inf'))
                    hard_negatives, _ = neg_similarity.max(dim=1)
                    
                    triplet_loss = F.relu(
                        hard_negatives - pos_similarity + self.margin
                    ).mean()
                    
                    # Combine losses
                    pair_loss = (loss_12 + loss_21) / 2 + 0.1 * triplet_loss
                    
                    total_loss += pair_loss
                    num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else total_loss


class CompressionLoss(nn.Module):
    """Loss for encouraging efficient compression with information theory."""
    
    def __init__(self, sparsity_weight=0.01, quantization_weight=0.001,
                 entropy_weight=0.001):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.quantization_weight = quantization_weight
        self.entropy_weight = entropy_weight
        
    def forward(self, attractor_field):
        """
        Compute compression-related losses.
        
        Args:
            attractor_field: Compressed representation
            
        Returns:
            Compression loss
        """
        losses = {}
        
        # Sparsity loss (L1)
        sparsity = torch.abs(attractor_field).mean()
        losses['sparsity'] = sparsity
        
        # Quantization loss
        # Encourage values near discrete levels (-1, -0.5, 0, 0.5, 1)
        levels = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], 
                            device=attractor_field.device)
        
        # Distance to nearest level
        expanded_field = attractor_field.unsqueeze(-1)
        if attractor_field.dim() == 1:
            expanded_levels = levels.view(-1)
        elif attractor_field.dim() == 2:
            expanded_levels = levels.view(1, -1)
        else:
            expanded_levels = levels.view(1, 1, -1)
            
        distances = torch.abs(expanded_field - expanded_levels)
        min_distances, _ = distances.min(dim=-1)
        
        quantization = min_distances.mean()
        losses['quantization'] = quantization
        
        # Entropy regularization
        if attractor_field.shape[-1] > 1:
            # Compute probability distribution over dimensions
            field_abs = torch.abs(attractor_field)
            field_probs = field_abs / (field_abs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Entropy
            entropy = -(field_probs * torch.log(field_probs + 1e-8)).sum(dim=-1).mean()
            losses['entropy'] = entropy
        else:
            losses['entropy'] = torch.tensor(0.0)
        
        # Total compression loss
        total_loss = (
            self.sparsity_weight * losses['sparsity'] +
            self.quantization_weight * losses['quantization'] +
            self.entropy_weight * losses['entropy']
        )
        
        return total_loss


class GDALoss(nn.Module):
    """Enhanced GDA loss with meta-learning and test-time optimization."""
    
    def __init__(self, 
                 reconstruction_weight=1.0,
                 gravitational_weight=0.1,
                 contrastive_weight=0.1,
                 compression_weight=0.01,
                 use_meta_weights=True,
                 use_test_time_opt=False):
        super().__init__()
        
        # Component losses
        self.reconstruction_loss = ReconstructionLoss()
        self.gravitational_loss = GravitationalLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.compression_loss = CompressionLoss()
        
        # Base weights
        self.base_weights = {
            'reconstruction': reconstruction_weight,
            'gravitational': gravitational_weight,
            'contrastive': contrastive_weight,
            'compression': compression_weight
        }
        
        # Meta-learning for adaptive weights
        self.use_meta_weights = use_meta_weights
        if use_meta_weights:
            self.meta_weighter = MetaLearningWeights(num_losses=4)
        
        # Test-time optimization
        self.use_test_time_opt = use_test_time_opt
        if use_test_time_opt:
            self.test_time_opt = TestTimeOptimization(
                self.reconstruction_loss
            )
        
        # Multi-stage training schedule
        self.register_buffer('training_stage', torch.tensor(0))
        self.stage_transitions = [1000, 5000, 10000]  # Steps
        
    def forward(self, outputs, targets, modality='auto', global_step=0):
        """
        Compute total GDA loss with adaptive weighting.
        
        Args:
            outputs: Model outputs dict
            targets: Ground truth targets
            modality: Target modality
            global_step: Current training step
            
        Returns:
            Total loss and detailed components
        """
        losses = {}
        raw_losses = {}
        
        # Update training stage
        self._update_training_stage(global_step)
        
        # Compute individual losses
        
        # 1. Reconstruction loss
        if 'reconstructions' in outputs and targets is not None:
            try:
                recon_loss = self.reconstruction_loss(
                    outputs['reconstructions'], targets, modality
                )
                raw_losses['reconstruction'] = recon_loss
                
                # Test-time optimization if enabled
                if self.use_test_time_opt and not self.training:
                    try:
                        outputs['reconstructions'] = self.test_time_opt(
                            outputs['reconstructions'], targets,
                            [p for p in self.parameters() if p.requires_grad]
                        )
                        # Recompute loss
                        recon_loss = self.reconstruction_loss(
                            outputs['reconstructions'], targets, modality
                        )
                    except:
                        pass  # Fall back to original loss
                
                losses['reconstruction'] = recon_loss
            except Exception as e:
                losses['reconstruction'] = torch.tensor(0.0)
        
        # 2. Gravitational loss
        if 'convergence_info' in outputs:
            try:
                trajectory_data = outputs.get('trajectory_data', None)
                grav_loss, grav_components = self.gravitational_loss(
                    outputs.get('attractor_field'),
                    outputs['convergence_info'],
                    trajectory_data
                )
                raw_losses['gravitational'] = grav_loss
                losses['gravitational'] = grav_loss
                losses.update({f'grav_{k}': v for k, v in grav_components.items()})
            except Exception as e:
                losses['gravitational'] = torch.tensor(0.0)
        
        # 3. Contrastive loss
        if 'contrastive_features' in outputs:
            try:
                features = outputs['contrastive_features']
                if isinstance(features, dict) and len(features) >= 2:
                    contrast_loss = self.contrastive_loss(features)
                    raw_losses['contrastive'] = contrast_loss
                    losses['contrastive'] = contrast_loss
            except Exception as e:
                losses['contrastive'] = torch.tensor(0.0)
        
        # 4. Compression loss
        if 'attractor_field' in outputs:
            try:
                comp_loss = self.compression_loss(outputs['attractor_field'])
                raw_losses['compression'] = comp_loss
                losses['compression'] = comp_loss
            except Exception as e:
                losses['compression'] = torch.tensor(0.0)
        
        # Get adaptive weights
        if self.use_meta_weights and len(raw_losses) > 0:
            try:
                # Compute gradients for meta-learning
                grad_dict = {}
                for name, loss in raw_losses.items():
                    if loss.requires_grad:
                        try:
                            grad = torch.autograd.grad(
                                loss, self.parameters(), 
                                retain_graph=True, allow_unused=True
                            )
                            grad_norm = sum(g.norm().item() for g in grad if g is not None)
                            grad_dict[name] = torch.tensor(grad_norm)
                        except:
                            grad_dict[name] = torch.tensor(0.0)
                
                # Get meta weights
                meta_weights = self.meta_weighter(raw_losses, grad_dict)
                
                # Apply stage-based adjustments
                stage_weights = self._get_stage_weights()
                
                # Combine base, meta, and stage weights
                final_weights = {}
                weight_names = ['reconstruction', 'gravitational', 'contrastive', 'compression']
                
                for i, name in enumerate(weight_names):
                    base = self.base_weights[name]
                    meta = meta_weights[i].item() if i < len(meta_weights) else 1.0
                    stage = stage_weights.get(name, 1.0)
                    
                    final_weights[name] = base * meta * stage
            except:
                # Fall back to base weights with stage adjustments
                stage_weights = self._get_stage_weights()
                final_weights = {
                    k: v * stage_weights.get(k, 1.0) 
                    for k, v in self.base_weights.items()
                }
        else:
            # Use base weights with stage adjustments
            stage_weights = self._get_stage_weights()
            final_weights = {
                k: v * stage_weights.get(k, 1.0) 
                for k, v in self.base_weights.items()
            }
        
        # Compute weighted total loss
        device = next(iter(outputs.values())).device if outputs else torch.device('cpu')
        total_loss = torch.tensor(0.0, device=device)
        
        for loss_name, weight in final_weights.items():
            if loss_name in losses and isinstance(losses[loss_name], torch.Tensor):
                total_loss = total_loss + weight * losses[loss_name]
        
        # Add total to losses dict
        losses['total'] = total_loss
        losses['weights'] = final_weights
        
        return total_loss, losses
    
    def _update_training_stage(self, global_step):
        """Update training stage based on step count."""
        for i, transition in enumerate(self.stage_transitions):
            if global_step >= transition:
                self.training_stage = torch.tensor(i + 1)
    
    def _get_stage_weights(self):
        """Get stage-specific weight adjustments."""
        stage = self.training_stage.item()
        
        if stage == 0:
            # Early stage: focus on reconstruction
            return {
                'reconstruction': 2.0,
                'gravitational': 0.5,
                'contrastive': 0.1,
                'compression': 0.01
            }
        elif stage == 1:
            # Mid stage: balance all objectives
            return {
                'reconstruction': 1.0,
                'gravitational': 1.0,
                'contrastive': 0.5,
                'compression': 0.1
            }
        elif stage == 2:
            # Late stage: refine with physics
            return {
                'reconstruction': 0.8,
                'gravitational': 1.5,
                'contrastive': 1.0,
                'compression': 0.5
            }
        else:
            # Final stage: focus on generalization
            return {
                'reconstruction': 0.5,
                'gravitational': 2.0,
                'contrastive': 1.5,
                'compression': 1.0
            }
"""
Enhanced training script for GDA models with multi-stage optimization.
Implements hybrid architectures, meta-learning, and distributed training support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import json
import os
import time
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List, Tuple
import logging
from contextlib import nullcontext
from collections import defaultdict

# Import GDA modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gda_engine.encoder import MultiModalEncoder
from gda_engine.decoder import MultiModalDecoder
from gda_engine.attractor_core import AttractorCore
from gda_engine.fusion import MultiModalFusion
from gda_engine.loss_functions import GDALoss
from gda_engine.utils import get_device, EarlyStopping
from train.datasets_loader import MultiModalDataset, create_dataloaders


class GDAModel(nn.Module):
    """Enhanced GDA model with hybrid architectures."""
    
    def __init__(self, config):
        super().__init__()
        
        # Core components with enhancements
        self.encoder = MultiModalEncoder(output_dim=config['latent_dim'])
        
        self.attractor_core = AttractorCore(
            latent_dim=config['latent_dim'],
            num_attractors=config['num_attractors'],
            gravity_strength=config['gravity_strength'],
            damping=config['damping'],
            use_hamiltonian=config.get('use_hamiltonian', True),
            use_boundary_pinn=config.get('use_boundary_pinn', True)
        )
        
        self.decoder = MultiModalDecoder(input_dim=config['latent_dim'])
        
        self.fusion = MultiModalFusion(
            dim=config['latent_dim'],
            gravity_strength=config['gravity_strength']
        )
        
        # Variational components for hybrid architecture
        if config.get('use_variational', False):
            self.vae_encoder = nn.Sequential(
                nn.Linear(config['latent_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, config['latent_dim'] * 2)  # mean and log_var
            )
            self.vae_decoder = nn.Sequential(
                nn.Linear(config['latent_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, config['latent_dim'])
            )
        
        self.config = config
        
    def forward(self, inputs, target_modality='auto', return_all=False):
        """
        Enhanced forward pass with optional VAE regularization.
        
        Args:
            inputs: Input data (dict or tensor)
            target_modality: Output modality
            return_all: Return all intermediate results
            
        Returns:
            Model outputs dict
        """
        outputs = {}
        
        # Multi-modal processing
        if isinstance(inputs, dict):
            embeddings = {}
            masses = {}
            
            # Encode each modality
            for modality, data in inputs.items():
                if modality in ['text', 'image', 'audio']:
                    emb, mass = self.encoder(data, modality=modality)
                    embeddings[modality] = emb
                    masses[modality] = mass
            
            # Attractor dynamics per modality
            attractor_fields = {}
            convergence_infos = {}
            trajectory_data = {}
            
            for modality, emb in embeddings.items():
                field, conv_info = self.attractor_core(
                    emb, 
                    num_iterations=self.config.get('num_iterations', 50),
                    return_trajectory=return_all,
                    adaptive_steps=True
                )
                
                attractor_fields[modality] = field
                convergence_infos[modality] = conv_info
                
                # Store trajectory for physics loss
                if return_all and conv_info.get('trajectory'):
                    trajectory_data[modality] = {
                        'positions': conv_info['trajectory'],
                        'masses': masses[modality],
                        'energies': conv_info.get('energy_history', [])
                    }
            
            # Fusion
            if len(attractor_fields) > 1:
                fusion_output = self.fusion(embeddings, attractor_fields)
                unified_field = fusion_output['unified_field']
                outputs['fusion_info'] = fusion_output['fusion_info']
                outputs['contrastive_features'] = fusion_output['contrastive_features']
            else:
                # Single modality
                unified_field = list(attractor_fields.values())[0]
            
            outputs['convergence_info'] = convergence_infos
            outputs['trajectory_data'] = trajectory_data if trajectory_data else None
            
        else:
            # Single input processing
            embeddings, masses = self.encoder(inputs)
            unified_field, conv_info = self.attractor_core(
                embeddings, 
                num_iterations=self.config.get('num_iterations', 50),
                return_trajectory=return_all
            )
            outputs['convergence_info'] = {'auto': conv_info}
            
            if return_all and conv_info.get('trajectory'):
                outputs['trajectory_data'] = {
                    'auto': {
                        'positions': conv_info['trajectory'],
                        'masses': masses,
                        'energies': conv_info.get('energy_history', [])
                    }
                }
        
        # VAE regularization if enabled
        if hasattr(self, 'vae_encoder'):
            vae_params = self.vae_encoder(unified_field)
            mu = vae_params[:, :self.config['latent_dim']]
            log_var = vae_params[:, self.config['latent_dim']:]
            
            # Reparameterization
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # VAE reconstruction
            vae_field = self.vae_decoder(z)
            
            # Combine with attractor field
            unified_field = 0.7 * unified_field + 0.3 * vae_field
            
            outputs['vae_mu'] = mu
            outputs['vae_log_var'] = log_var
        
        # Decode
        outputs['attractor_field'] = unified_field
        outputs['reconstructions'] = self.decoder(unified_field, target_modality=target_modality)
        
        return outputs


class DistributedTrainer:
    """Enhanced trainer with distributed training support."""
    
    def __init__(self, model, config, rank=0, world_size=1):
        self.model = model
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        # Setup device
        if self.is_distributed:
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = get_device()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Distributed wrapper
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )
        
        # Setup optimizer with LARS for large batch training
        self.optimizer = self._setup_optimizer()
        
        # Enhanced loss with meta-learning
        self.criterion = GDALoss(
            reconstruction_weight=config.get('reconstruction_weight', 1.0),
            gravitational_weight=config.get('gravitational_weight', 0.1),
            contrastive_weight=config.get('contrastive_weight', 0.1),
            compression_weight=config.get('compression_weight', 0.01),
            use_meta_weights=config.get('use_meta_weights', True),
            use_test_time_opt=config.get('use_test_time_opt', False)
        )
        
        # Learning rate scheduling
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Gradient accumulation
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Setup logging
        self.setup_logging()
        
    def _setup_optimizer(self):
        """Setup optimizer with LARS and layer-wise learning rates."""
        # Get parameter groups
        encoder_params = []
        attractor_params = []
        decoder_params = []
        fusion_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'attractor' in name:
                attractor_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            elif 'fusion' in name:
                fusion_params.append(param)
            else:
                other_params.append(param)
        
        # Layer-wise learning rates
        base_lr = self.config['learning_rate']
        param_groups = [
            {'params': encoder_params, 'lr': base_lr, 'name': 'encoder'},
            {'params': attractor_params, 'lr': base_lr * 0.1, 'name': 'attractor'},
            {'params': decoder_params, 'lr': base_lr, 'name': 'decoder'},
            {'params': fusion_params, 'lr': base_lr * 0.5, 'name': 'fusion'},
            {'params': other_params, 'lr': base_lr, 'name': 'other'}
        ]
        
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        # Choose optimizer
        opt_name = self.config.get('optimizer', 'adamw')
        
        if opt_name == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.999)
            )
        elif opt_name == 'adam':
            optimizer = optim.Adam(
                param_groups,
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif opt_name == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif opt_name == 'lars':
            # LARS optimizer for large batch training
            try:
                from torch.optim import LARS
                optimizer = LARS(
                    param_groups,
                    weight_decay=self.config.get('weight_decay', 0),
                    momentum=0.9
                )
            except ImportError:
                print("LARS optimizer not available, using AdamW instead")
                optimizer = optim.AdamW(
                    param_groups,
                    weight_decay=self.config.get('weight_decay', 0.01),
                    betas=(0.9, 0.999)
                )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        num_epochs = self.config['num_epochs']
        warmup_epochs = self.config.get('warmup_epochs', 5)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        return scheduler
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        if self.rank == 0:  # Only main process logs
            # Create log directory
            log_dir = os.path.join(self.config.get('save_dir', 'logs'), 'training')
            os.makedirs(log_dir, exist_ok=True)
            
            # Setup logger
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(log_dir, 'training.log')),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            
            # Weights & Biases
            if self.config.get('use_wandb', False):
                wandb.init(
                    project="gda-training-enhanced",
                    config=self.config,
                    name=self.config.get('experiment_name', 'gda_run')
                )
                wandb.watch(self.model, log_freq=100)
        else:
            self.logger = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        # Progress bar only on main process
        if self.rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = train_loader
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Mixed precision context
            amp_context = torch.cuda.amp.autocast() if self.use_amp else nullcontext()
            
            with amp_context:
                # Forward pass with all intermediate results for physics loss
                outputs = self.model(
                    batch['inputs'], 
                    target_modality=batch.get('modality', 'auto'),
                    return_all=True
                )
                
                # Compute loss
                loss, loss_components = self.criterion(
                    outputs,
                    batch.get('targets', batch['inputs']),
                    modality=batch.get('modality', 'auto'),
                    global_step=self.global_step
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip', 1.0)
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Stabilize attractors periodically
                if hasattr(self.model, 'module'):
                    if hasattr(self.model.module.attractor_core, 'stabilize_attractors'):
                        self.model.module.attractor_core.stabilize_attractors()
                else:
                    if hasattr(self.model.attractor_core, 'stabilize_attractors'):
                        self.model.attractor_core.stabilize_attractors()
            
            # Track metrics
            epoch_losses.append(loss.item() * self.accumulation_steps)
            for key, value in loss_components.items():
                if key != 'total' and isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
            
            # Update progress bar
            if self.rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                    'grad_norm': f"{grad_norm:.3f}" if 'grad_norm' in locals() else 'N/A'
                })
            
            # Logging
            if self.rank == 0 and self.global_step % 100 == 0:
                self._log_step(loss, loss_components, grad_norm if 'grad_norm' in locals() else 0)
            
            self.global_step += 1
        
        # Compute epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader):
        """Validate model with detailed metrics."""
        self.model.eval()
        
        val_losses = []
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=self.rank != 0):
                batch = self._batch_to_device(batch)
                
                # Forward pass
                with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
                    outputs = self.model(
                        batch['inputs'],
                        target_modality=batch.get('modality', 'auto')
                    )
                    
                    loss, loss_components = self.criterion(
                        outputs,
                        batch.get('targets', batch['inputs']),
                        modality=batch.get('modality', 'auto')
                    )
                
                val_losses.append(loss.item())
                for key, value in loss_components.items():
                    if key != 'total' and isinstance(value, torch.Tensor):
                        val_metrics[key].append(value.item())
        
        avg_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in val_metrics.items()}
        
        # Gather metrics across processes if distributed
        if self.is_distributed:
            avg_loss = self._gather_metric(avg_loss)
            avg_metrics = {k: self._gather_metric(v) for k, v in avg_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader, val_loader=None, num_epochs=None):
        """Main training loop with enhancements."""
        num_epochs = num_epochs or self.config['num_epochs']
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 10),
            mode='min'
        )
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader)
            else:
                val_loss, val_metrics = train_loss, train_metrics
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            if self.rank == 0:
                self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics)
                
                # Save checkpoint
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                
                # Periodic checkpoint
                if epoch % self.config.get('checkpoint_interval', 10) == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Early stopping
            if early_stopping(val_loss):
                if self.rank == 0:
                    self.logger.info("Early stopping triggered")
                break
        
        # Final checkpoint
        if self.rank == 0:
            self.save_checkpoint('final_model.pt')
    
    def save_checkpoint(self, filename):
        """Save enhanced checkpoint with all components."""
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'criterion_state': self.criterion.state_dict() if hasattr(self.criterion, 'state_dict') else None
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = os.path.join(self.config.get('save_dir', 'checkpoints'), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(checkpoint, save_path)
        
        if self.logger:
            self.logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filepath):
        """Load checkpoint with proper state restoration."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if checkpoint.get('criterion_state') and hasattr(self.criterion, 'load_state_dict'):
            self.criterion.load_state_dict(checkpoint['criterion_state'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.logger:
            self.logger.info(f"Checkpoint loaded from {filepath} (epoch {self.epoch})")
    
    def _batch_to_device(self, batch):
        """Move batch to device with proper handling."""
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        else:
            return self._to_device(batch)
    
    def _to_device(self, data):
        """Move data to device recursively."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(v) for v in data]
        else:
            return data
    
    def _gather_metric(self, metric):
        """Gather metric across distributed processes."""
        if not self.is_distributed:
            return metric
        
        metric_tensor = torch.tensor(metric, device=self.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item() / self.world_size
    
    def _log_step(self, loss, loss_components, grad_norm):
        """Log training step metrics."""
        if self.config.get('use_wandb', False):
            log_dict = {
                'train/loss': loss.item() * self.accumulation_steps,
                'train/grad_norm': grad_norm,
                'train/step': self.global_step,
                'train/lr': self.scheduler.get_last_lr()[0]
            }
            
            for k, v in loss_components.items():
                if isinstance(v, torch.Tensor):
                    log_dict[f'train/{k}'] = v.item()
            
            wandb.log(log_dict)
    
    def _log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        """Log epoch metrics."""
        # Console logging
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Train Loss: {train_loss:.4f}")
        self.logger.info(f"  Val Loss: {val_loss:.4f}")
        self.logger.info(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
        
        # Detailed metrics
        self.logger.info("  Train Metrics:")
        for k, v in train_metrics.items():
            self.logger.info(f"    {k}: {v:.4f}")
        
        self.logger.info("  Val Metrics:")
        for k, v in val_metrics.items():
            self.logger.info(f"    {k}: {v:.4f}")
        
        # Weights & Biases
        if self.config.get('use_wandb', False):
            log_dict = {
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
            for k, v in train_metrics.items():
                log_dict[f'train/{k}'] = v
            for k, v in val_metrics.items():
                log_dict[f'val/{k}'] = v
            
            # Log attractor statistics if available
            try:
                if hasattr(self.model, 'module'):
                    attractor_stats = self.model.module.attractor_core.get_attractor_stats()
                else:
                    attractor_stats = self.model.attractor_core.get_attractor_stats()
                
                for k, v in attractor_stats.items():
                    log_dict[f'attractor/{k}'] = v
            except:
                pass
            
            wandb.log(log_dict)


def setup_distributed(rank, world_size):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_distributed(rank, world_size, config):
    """Training function for distributed training."""
    # Setup distributed
    setup_distributed(rank, world_size)
    
    # Create model
    model = GDAModel(config)
    
    # Create data loaders with distributed sampler
    train_dataset = MultiModalDataset(
        data_path=config['data']['dataset_path'],
        split='train',
        config=config['data']
    )
    val_dataset = MultiModalDataset(
        data_path=config['data']['dataset_path'],
        split='val',
        config=config['data']
    )
    
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        sampler=train_sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Create trainer
    trainer = DistributedTrainer(model, config['training'], rank, world_size)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main training function with enhanced features."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'train_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Merge model and training configs
    full_config = {**config['model'], **config['training']}
    full_config['data'] = config['data']
    
    # Set random seeds
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Check for distributed training
    world_size = config['hardware'].get('world_size', 1)
    distributed = config['hardware'].get('distributed', False) and world_size > 1
    
    if distributed and torch.cuda.device_count() >= world_size:
        # Distributed training
        print(f"Starting distributed training on {world_size} GPUs")
        mp.spawn(
            train_distributed,
            args=(world_size, full_config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU/CPU training
        print("Starting single device training")
        
        # Create model
        model = GDAModel(config['model'])
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(config)
        
        # Create trainer
        trainer = DistributedTrainer(model, full_config, rank=0, world_size=1)
        
        # Resume from checkpoint if specified
        if config['experiment'].get('resume_from'):
            trainer.load_checkpoint(config['experiment']['resume_from'])
        
        # Train
        trainer.train(train_loader, val_loader)
        
        # Export final model
        export_path = os.path.join(
            config['experiment'].get('save_dir', 'models'),
            'gda_enhanced.pt'
        )
        
        # Save in optimized format
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, export_path)
        
        print(f"Training complete! Model exported to {export_path}")
        
        # Log final statistics
        if trainer.logger:
            trainer.logger.info("Training Summary:")
            trainer.logger.info(f"  Total epochs: {trainer.epoch + 1}")
            trainer.logger.info(f"  Total steps: {trainer.global_step}")
            trainer.logger.info(f"  Best validation loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
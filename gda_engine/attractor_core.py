"""
Core attractor dynamics with stability enhancements and hybrid architectures.
Manages evolution of attractor fields with physics-informed constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from .gravity_kernel import GravityKernel
from typing import Dict, Optional, Tuple, List


class HamiltonianLayer(nn.Module):
    """Hamiltonian-inspired layer for energy conservation."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.potential_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, 1)
        )
        self.kinetic_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, 1)
        )
        
    def forward(self, positions, velocities):
        """Compute Hamiltonian dynamics."""
        # H = T + V
        potential = self.potential_net(positions)
        kinetic = self.kinetic_net(velocities)
        
        # Symplectic update
        dp_dt = -torch.autograd.grad(potential.sum(), positions, 
                                    create_graph=True)[0]
        dq_dt = torch.autograd.grad(kinetic.sum(), velocities,
                                   create_graph=True)[0]
        
        return dp_dt, dq_dt


class PhysicsInformedBoundary(nn.Module):
    """Physics-informed neural network for boundary conditions."""
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, positions, velocities):
        """Apply learned boundary conditions."""
        combined = torch.cat([positions, velocities], dim=-1)
        correction = self.net(combined)
        
        # Ensure stability through soft constraints
        positions_corrected = positions + 0.1 * torch.tanh(correction)
        
        return positions_corrected, velocities


class AttractorCore(nn.Module):
    """Enhanced attractor field dynamics with stability guarantees."""
    
    def __init__(self, latent_dim=128, num_attractors=10, 
                 gravity_strength=1e-3, damping=0.95, 
                 stability_threshold=1e-6, use_hamiltonian=True,
                 use_boundary_pinn=True):
        """
        Initialize attractor core with stability enhancements.
        
        Args:
            latent_dim: Dimension of latent space
            num_attractors: Number of attractor points
            gravity_strength: Strength of gravitational attraction
            damping: Velocity damping factor
            stability_threshold: Convergence threshold
            use_hamiltonian: Use Hamiltonian dynamics for stability
            use_boundary_pinn: Use PINN for boundary conditions
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_attractors = num_attractors
        self.gravity_strength = gravity_strength
        self.damping = damping
        self.stability_threshold = stability_threshold
        
        # Enhanced gravity kernel with GPU support
        self.gravity_kernel = GravityKernel(
            G=gravity_strength,
            use_cuda=True  # Auto-detect GPU
        )
        
        # Learnable attractor parameters with better initialization
        self.attractor_positions = nn.Parameter(
            self._initialize_attractors()
        )
        self.attractor_masses = nn.Parameter(
            torch.ones(num_attractors) * 0.1
        )
        
        # State tracking buffers
        self.register_buffer('velocities', torch.zeros(num_attractors, latent_dim))
        self.register_buffer('convergence_history', torch.zeros(100))
        self.register_buffer('energy_history', torch.zeros(100))
        self.history_idx = 0
        
        # Stability modules
        self.use_hamiltonian = use_hamiltonian
        if use_hamiltonian:
            self.hamiltonian_layer = HamiltonianLayer(latent_dim)
        
        self.use_boundary_pinn = use_boundary_pinn
        if use_boundary_pinn:
            self.boundary_net = PhysicsInformedBoundary(latent_dim)
        
        # Adaptive parameters
        self.register_buffer('adaptive_damping', torch.tensor(damping))
        self.register_buffer('adaptive_dt', torch.tensor(0.1))
        
    def _initialize_attractors(self):
        """Initialize attractors with stable configuration."""
        # Use structured initialization (e.g., on hypersphere)
        angles = torch.linspace(0, 2 * np.pi, self.num_attractors + 1)[:-1]
        
        if self.latent_dim == 2:
            positions = torch.stack([
                torch.cos(angles),
                torch.sin(angles)
            ], dim=1) * 2.0
        else:
            # Higher dimensions: use random orthogonal vectors
            positions = torch.randn(self.num_attractors, self.latent_dim)
            positions = torch.nn.functional.normalize(positions, p=2, dim=1) * 2.0
            
        return positions
    
    def forward(self, input_embeddings, num_iterations=50, 
                return_trajectory=False, adaptive_steps=True):
        """
        Process embeddings through enhanced attractor dynamics.
        
        Args:
            input_embeddings: Input mass vectors (B, N, D)
            num_iterations: Maximum number of iterations
            return_trajectory: Return full trajectory history
            adaptive_steps: Use adaptive time stepping
            
        Returns:
            Final state and convergence info
        """
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Initialize particles from input
        particles = input_embeddings.clone()
        particle_masses = torch.ones(
            batch_size, particles.shape[1], device=device
        ) * 0.1
        particle_velocities = torch.zeros_like(particles)
        
        # Combine with learned attractors
        all_positions = torch.cat([
            particles.reshape(-1, self.latent_dim),
            self.attractor_positions.repeat(batch_size, 1)
        ], dim=0)
        
        all_masses = torch.cat([
            particle_masses.reshape(-1),
            self.attractor_masses.repeat(batch_size)
        ], dim=0)
        
        all_velocities = torch.cat([
            particle_velocities.reshape(-1, self.latent_dim),
            self.velocities.repeat(batch_size, 1)
        ], dim=0)
        
        # Tracking
        position_history = [] if return_trajectory else None
        energy_history = []
        convergence_history = []
        
        # Adaptive time stepping
        dt = self.adaptive_dt.item() if adaptive_steps else 0.1
        
        # Main dynamics loop
        for i in range(num_iterations):
            prev_positions = all_positions.clone()
            
            # Apply gravitational dynamics with boundary conditions
            boundary_fn = lambda p, v: self.boundary_net(p, v) if self.use_boundary_pinn else (p, v)
            
            all_positions, all_velocities = self.gravity_kernel.apply_gravitational_dynamics(
                all_masses, all_positions, all_velocities, 
                dt=dt,
                use_barnes_hut=True,  # Automatic optimization
                boundary_conditions=boundary_fn
            )
            
            # Apply Hamiltonian constraints if enabled
            if self.use_hamiltonian and i % 5 == 0:
                with torch.enable_grad():
                    all_positions.requires_grad_(True)
                    all_velocities.requires_grad_(True)
                    
                    dp_dt, dq_dt = self.hamiltonian_layer(
                        all_positions, all_velocities
                    )
                    
                    # Symplectic correction
                    all_positions = all_positions + 0.01 * dp_dt
                    all_velocities = all_velocities + 0.01 * dq_dt
                    
                    all_positions = all_positions.detach()
                    all_velocities = all_velocities.detach()
            
            # Adaptive damping based on convergence
            current_damping = self._compute_adaptive_damping(i, convergence_history)
            all_velocities = all_velocities * current_damping
            
            # Apply stability constraints
            all_positions = self._apply_stability_constraints(all_positions)
            
            # Track convergence
            position_change = torch.norm(all_positions - prev_positions)
            convergence_history.append(position_change.item())
            
            if return_trajectory:
                position_history.append(all_positions.clone())
            
            # Compute and track energy
            energy = self.gravity_kernel.compute_gravitational_energy(
                all_masses, all_positions
            )
            energy_history.append(energy.item())
            
            # Adaptive time stepping
            if adaptive_steps:
                dt = self._compute_adaptive_timestep(
                    position_change, energy_history
                )
            
            # Check convergence
            if position_change < self.stability_threshold:
                break
        
        # Extract final states
        num_particles = particles.shape[1]
        final_particles = all_positions[:batch_size * num_particles].reshape(
            batch_size, num_particles, self.latent_dim
        )
        
        final_attractors = all_positions[batch_size * num_particles:].reshape(
            batch_size, self.num_attractors, self.latent_dim
        )
        
        # Update learned velocities
        self.velocities.data = all_velocities[batch_size * num_particles:].reshape(
            batch_size, self.num_attractors, self.latent_dim
        ).mean(dim=0)
        
        # Compute final field representation
        attractor_field = self.compute_attractor_field(
            final_particles, final_attractors
        )
        
        # Update history buffers
        self._update_history_buffers(convergence_history, energy_history)
        
        convergence_info = {
            'iterations': i + 1,
            'final_change': convergence_history[-1] if convergence_history else 0,
            'energy_history': energy_history,
            'converged': position_change < self.stability_threshold,
            'trajectory': position_history if return_trajectory else None
        }
        
        return attractor_field, convergence_info
    
    def compute_attractor_field(self, particles, attractors):
        """
        Enhanced field computation with multi-scale representation.
        
        Args:
            particles: Particle positions (B, N, D)
            attractors: Attractor positions (B, K, D)
            
        Returns:
            Multi-scale field representation
        """
        # Masses for weighting
        particle_masses = torch.ones(particles.shape[:2], device=particles.device)
        attractor_masses = self.attractor_masses.unsqueeze(0).repeat(
            particles.shape[0], 1
        )
        
        # Combine all masses and positions
        all_masses = torch.cat([particle_masses, attractor_masses], dim=1)
        all_positions = torch.cat([particles, attractors], dim=1)
        
        # Multi-scale features
        features = []
        
        # 1. Center of mass (global)
        total_mass = all_masses.sum(dim=1, keepdim=True)
        weighted_positions = all_masses.unsqueeze(-1) * all_positions
        center_of_mass = weighted_positions.sum(dim=1) / total_mass
        features.append(center_of_mass)
        
        # 2. Variance (spread)
        variance = ((all_positions - center_of_mass.unsqueeze(1)) ** 2).mean(dim=1)
        features.append(variance)
        
        # 3. Higher moments (shape)
        centered = all_positions - center_of_mass.unsqueeze(1)
        skewness = (centered ** 3).mean(dim=1)
        features.append(skewness)
        
        # 4. Pairwise statistics
        if all_positions.shape[1] > 1:
            distances = torch.pdist(all_positions[0])
            dist_stats = torch.stack([
                distances.mean(),
                distances.std(),
                distances.min(),
                distances.max()
            ]).unsqueeze(0).repeat(particles.shape[0], 1)
            features.append(dist_stats)
        
        # Combine features
        field_representation = torch.cat(features, dim=-1)
        
        # Project to standard dimension
        if field_representation.shape[-1] != self.latent_dim:
            if not hasattr(self, 'field_projection'):
                self.field_projection = nn.Linear(
                    field_representation.shape[-1], 
                    self.latent_dim
                ).to(field_representation.device)
            field_representation = self.field_projection(field_representation)
        
        return field_representation
    
    def _apply_stability_constraints(self, positions):
        """Apply constraints to ensure numerical stability."""
        # Soft clamping with smooth function
        max_radius = 10.0
        positions = torch.tanh(positions / max_radius) * max_radius
        
        # Remove NaN/Inf
        positions = torch.nan_to_num(
            positions, nan=0.0, posinf=max_radius, neginf=-max_radius
        )
        
        return positions
    
    def _compute_adaptive_damping(self, iteration, convergence_history):
        """Compute adaptive damping based on convergence behavior."""
        if len(convergence_history) < 2:
            return self.damping
        
        # Check if oscillating
        recent_changes = convergence_history[-10:] if len(convergence_history) > 10 else convergence_history
        
        if len(recent_changes) > 2:
            # Compute variance of changes
            changes_tensor = torch.tensor(recent_changes)
            variance = changes_tensor.std()
            
            # High variance suggests oscillation
            if variance > 0.1:
                # Increase damping
                return min(0.99, self.damping + 0.02)
            elif variance < 0.01:
                # Decrease damping for faster convergence
                return max(0.8, self.damping - 0.02)
        
        return self.damping
    
    def _compute_adaptive_timestep(self, position_change, energy_history):
        """Compute adaptive time step for stability."""
        base_dt = 0.1
        
        # Reduce dt if change is too large
        if position_change > 1.0:
            return base_dt * 0.5
        
        # Check energy conservation
        if len(energy_history) > 1:
            energy_change = abs(energy_history[-1] - energy_history[-2])
            if energy_change > 0.1:
                return base_dt * 0.7
        
        # Increase dt if converging slowly
        if position_change < 0.01:
            return min(base_dt * 1.2, 0.5)
        
        return base_dt
    
    def _update_history_buffers(self, convergence_history, energy_history):
        """Update circular history buffers."""
        # Store recent convergence values
        n_conv = min(len(convergence_history), len(self.convergence_history))
        if n_conv > 0:
            self.convergence_history[:n_conv] = torch.tensor(
                convergence_history[-n_conv:]
            )
        
        # Store recent energy values
        n_energy = min(len(energy_history), len(self.energy_history))
        if n_energy > 0:
            self.energy_history[:n_energy] = torch.tensor(
                energy_history[-n_energy:]
            )
    
    def stabilize_attractors(self):
        """Apply stability constraints to learned parameters."""
        with torch.no_grad():
            # Normalize positions to prevent drift
            self.attractor_positions.data = self._apply_stability_constraints(
                self.attractor_positions.data
            )
            
            # Ensure positive masses with bounds
            self.attractor_masses.data = torch.clamp(
                self.attractor_masses.data, min=1e-3, max=1.0
            )
            
            # Reset velocities if too high
            velocity_norm = torch.norm(self.velocities, dim=-1)
            high_velocity_mask = velocity_norm > 5.0
            self.velocities[high_velocity_mask] *= 0.1
    
    def get_attractor_stats(self):
        """Get comprehensive statistics about attractor state."""
        with torch.no_grad():
            stats = {
                'position_mean': self.attractor_positions.mean().item(),
                'position_std': self.attractor_positions.std().item(),
                'position_range': (
                    self.attractor_positions.min().item(),
                    self.attractor_positions.max().item()
                ),
                'mass_mean': self.attractor_masses.mean().item(),
                'mass_std': self.attractor_masses.std().item(),
                'velocity_norm': torch.norm(self.velocities, dim=-1).mean().item(),
                'total_energy': self.gravity_kernel.compute_gravitational_energy(
                    self.attractor_masses, self.attractor_positions
                ).item(),
                'recent_convergence': self.convergence_history[:10].mean().item(),
                'stability_score': self._compute_stability_score()
            }
        return stats
    
    def _compute_stability_score(self):
        """Compute overall stability score (0-1)."""
        scores = []
        
        # Position bounded-ness
        pos_range = self.attractor_positions.abs().max().item()
        scores.append(1.0 / (1.0 + pos_range / 10.0))
        
        # Velocity bounded-ness
        vel_norm = torch.norm(self.velocities, dim=-1).max().item()
        scores.append(1.0 / (1.0 + vel_norm / 5.0))
        
        # Energy stability
        energy_std = self.energy_history[:10].std().item()
        scores.append(1.0 / (1.0 + energy_std))
        
        return np.mean(scores)
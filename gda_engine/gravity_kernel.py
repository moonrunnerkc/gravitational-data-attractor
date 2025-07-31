"""
Gravitational force simulation kernel for GDA with optimized algorithms.
Implements physics-inspired attraction with Barnes-Hut approximation and GPU support.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class BarnesHutNode:
    """Node for Barnes-Hut quadtree/octree acceleration."""
    
    def __init__(self, center, size, dimension=2):
        self.center = center
        self.size = size
        self.dimension = dimension
        self.mass = 0.0
        self.center_of_mass = torch.zeros_like(center)
        self.children = None
        self.is_leaf = True
        self.particles = []
        
    def insert(self, position, mass, idx):
        """Insert particle into tree."""
        if self.is_leaf and len(self.particles) == 0:
            self.particles.append((position, mass, idx))
            self.mass = mass
            self.center_of_mass = position
        elif self.is_leaf and len(self.particles) == 1:
            # Split node
            self._subdivide()
            # Reinsert existing particle
            old_pos, old_mass, old_idx = self.particles[0]
            self._insert_to_child(old_pos, old_mass, old_idx)
            self.particles = []
            # Insert new particle
            self._insert_to_child(position, mass, idx)
        else:
            # Internal node
            self._insert_to_child(position, mass, idx)
        
        # Update mass and center of mass
        total_mass = self.mass + mass
        if total_mass > 0:
            self.center_of_mass = (self.mass * self.center_of_mass + mass * position) / total_mass
        self.mass = total_mass
    
    def _subdivide(self):
        """Create child nodes."""
        self.is_leaf = False
        self.children = []
        
        # Create 2^dimension children
        for i in range(2**self.dimension):
            offset = torch.zeros_like(self.center)
            for d in range(self.dimension):
                if i & (1 << d):
                    offset[d] = self.size / 4
                else:
                    offset[d] = -self.size / 4
            
            child_center = self.center + offset
            child = BarnesHutNode(child_center, self.size / 2, self.dimension)
            self.children.append(child)
    
    def _insert_to_child(self, position, mass, idx):
        """Insert particle to appropriate child."""
        child_idx = 0
        for d in range(self.dimension):
            if position[d] > self.center[d]:
                child_idx |= (1 << d)
        self.children[child_idx].insert(position, mass, idx)


class GravityKernel:
    """Simulates gravitational forces with optimized algorithms."""
    
    def __init__(self, G=6.674e-11, epsilon=1e-8, max_force=1e3, 
                 use_cuda=None, theta=0.5):
        """
        Initialize gravity kernel with optimization options.
        
        Args:
            G: Gravitational constant
            epsilon: Small value to prevent division by zero
            max_force: Maximum force magnitude
            use_cuda: Force CUDA usage (None=auto-detect)
            theta: Barnes-Hut accuracy parameter (smaller=more accurate)
        """
        self.G = G
        self.epsilon = epsilon
        self.max_force = max_force
        self.theta = theta
        
        # Auto-detect CUDA
        if use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
    def compute_distance_matrix(self, masses):
        """
        Compute pairwise distances using efficient GPU operations.
        
        Args:
            masses: Tensor of shape (N, D)
            
        Returns:
            Distance matrix of shape (N, N)
        """
        # Move to GPU if available
        masses = masses.to(self.device)
        
        # Efficient pairwise distance computation
        masses_norm = (masses ** 2).sum(dim=1, keepdim=True)
        distances = masses_norm + masses_norm.t() - 2 * torch.mm(masses, masses.t())
        distances = torch.sqrt(torch.clamp(distances, min=self.epsilon))
        
        return distances
    
    def build_barnes_hut_tree(self, positions, masses):
        """Build Barnes-Hut tree for efficient force computation."""
        N, D = positions.shape
        
        # Find bounding box
        min_pos = positions.min(dim=0)[0]
        max_pos = positions.max(dim=0)[0]
        center = (min_pos + max_pos) / 2
        size = (max_pos - min_pos).max().item() * 1.1
        
        # Build tree
        root = BarnesHutNode(center, size, dimension=D)
        for i in range(N):
            root.insert(positions[i], masses[i].item(), i)
        
        return root
    
    def compute_force_barnes_hut(self, position, mass, node, theta=0.5):
        """Compute force on particle using Barnes-Hut approximation."""
        if node.mass == 0:
            return torch.zeros_like(position)
        
        displacement = node.center_of_mass - position
        distance = torch.norm(displacement)
        
        # Check if we can use node as single mass
        if node.is_leaf or (node.size / distance) < theta:
            if distance < self.epsilon:
                return torch.zeros_like(position)
            
            force_magnitude = self.G * mass * node.mass / (distance ** 2)
            force_magnitude = torch.clamp(force_magnitude, max=self.max_force)
            force_direction = displacement / distance
            
            return force_magnitude * force_direction
        else:
            # Recurse to children
            force = torch.zeros_like(position)
            for child in node.children:
                force += self.compute_force_barnes_hut(position, mass, child, theta)
            return force
    
    def compute_gravitational_force(self, m1, m2, positions1, positions2):
        """
        Compute gravitational force with GPU acceleration.
        
        F = G * (m1 * m2) / r^2
        """
        # Move to GPU
        positions1 = positions1.to(self.device)
        positions2 = positions2.to(self.device)
        m1 = m1.to(self.device) if isinstance(m1, torch.Tensor) else torch.tensor(m1, device=self.device)
        m2 = m2.to(self.device) if isinstance(m2, torch.Tensor) else torch.tensor(m2, device=self.device)
        
        # Compute displacement vectors
        displacement = positions2 - positions1
        
        # Compute distances with numerical stability
        distances = torch.norm(displacement, dim=-1, keepdim=True)
        distances = torch.clamp(distances, min=self.epsilon)
        
        # Compute force magnitudes with clamping
        force_magnitude = self.G * (m1 * m2) / (distances ** 2)
        force_magnitude = torch.clamp(force_magnitude, max=self.max_force)
        
        # Compute force directions
        force_direction = displacement / distances
        
        # Handle NaN/Inf
        force_direction = torch.nan_to_num(force_direction, nan=0.0, posinf=0.0, neginf=0.0)
        
        return force_magnitude * force_direction
    
    def apply_gravitational_dynamics(self, masses, positions, velocities, dt=0.01, 
                                   use_barnes_hut=True, boundary_conditions=None):
        """
        Update positions and velocities with optimized algorithms.
        
        Args:
            masses: Tensor of masses (N,)
            positions: Tensor of positions (N, D)
            velocities: Tensor of velocities (N, D)
            dt: Time step
            use_barnes_hut: Use Barnes-Hut approximation for large N
            boundary_conditions: Optional boundary enforcement function
            
        Returns:
            Updated positions and velocities
        """
        N = masses.shape[0]
        D = positions.shape[1]
        
        # Move to GPU
        masses = masses.to(self.device)
        positions = positions.to(self.device)
        velocities = velocities.to(self.device)
        
        # Initialize forces
        forces = torch.zeros_like(positions)
        
        # Choose algorithm based on problem size
        if use_barnes_hut and N > 100:
            # Use Barnes-Hut for large systems
            tree = self.build_barnes_hut_tree(positions.cpu(), masses.cpu())
            
            for i in range(N):
                force = self.compute_force_barnes_hut(
                    positions[i].cpu(), masses[i].cpu(), tree, self.theta
                )
                forces[i] = force.to(self.device)
        else:
            # Direct N-body for small systems (GPU optimized)
            # Compute all pairwise forces efficiently
            pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, D)
            distances = torch.norm(pos_diff, dim=2)  # (N, N)
            distances = torch.clamp(distances, min=self.epsilon)
            
            # Mask diagonal
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            distances = distances + (~mask) * 1e10
            
            # Force magnitudes
            mass_products = masses.unsqueeze(0) * masses.unsqueeze(1)  # (N, N)
            force_mags = self.G * mass_products / (distances ** 2)  # (N, N)
            force_mags = torch.clamp(force_mags * mask, max=self.max_force)
            
            # Force directions
            force_dirs = pos_diff / distances.unsqueeze(2)  # (N, N, D)
            force_dirs = torch.nan_to_num(force_dirs, nan=0.0)
            
            # Sum forces
            forces = (force_mags.unsqueeze(2) * force_dirs).sum(dim=1)
        
        # Update velocities (F = ma)
        accelerations = forces / masses.unsqueeze(1)
        velocities = velocities + accelerations * dt
        
        # Update positions
        positions = positions + velocities * dt
        
        # Apply boundary conditions if specified
        if boundary_conditions is not None:
            positions, velocities = boundary_conditions(positions, velocities)
        
        return positions, velocities
    
    def compute_field_potential(self, masses, positions):
        """
        Compute gravitational potential field with GPU optimization.
        
        V = -G * m / r
        """
        positions = positions.to(self.device)
        masses = masses.to(self.device)
        
        distances = self.compute_distance_matrix(positions)
        
        # Mask diagonal
        mask = 1 - torch.eye(distances.shape[0], device=self.device)
        distances = distances + (1 - mask) * 1e10
        
        # Compute potentials
        potentials = -self.G * masses.unsqueeze(1) / distances
        potentials = potentials * mask
        
        # Sum contributions
        total_potential = potentials.sum(dim=0)
        
        return total_potential
    
    def compute_center_of_mass(self, masses, positions):
        """Compute center of mass for system."""
        masses = masses.to(self.device)
        positions = positions.to(self.device)
        
        total_mass = masses.sum()
        weighted_positions = masses.unsqueeze(1) * positions
        center_of_mass = weighted_positions.sum(dim=0) / total_mass
        
        return center_of_mass
    
    def compute_gravitational_energy(self, masses, positions):
        """
        Compute total gravitational potential energy.
        
        U = -G * Σ(m_i * m_j / r_ij) for all i < j
        """
        masses = masses.to(self.device)
        positions = positions.to(self.device)
        
        distances = self.compute_distance_matrix(positions)
        
        # Upper triangular mask
        mask = torch.triu(torch.ones_like(distances), diagonal=1)
        
        # Compute pairwise energies
        mass_products = masses.unsqueeze(1) * masses.unsqueeze(0)
        energies = -self.G * mass_products / (distances + self.epsilon)
        
        # Sum upper triangular part
        total_energy = (energies * mask).sum()
        
        return total_energy
    
    def apply_boundary_conditions(self, positions, velocities, 
                                boundary_type='reflective', bounds=(-10, 10)):
        """
        Apply boundary conditions to prevent divergence.
        
        Args:
            positions: Current positions
            velocities: Current velocities
            boundary_type: 'reflective', 'periodic', or 'damping'
            bounds: Tuple of (min, max) bounds
            
        Returns:
            Updated positions and velocities
        """
        min_bound, max_bound = bounds
        
        if boundary_type == 'reflective':
            # Reflect particles that exceed bounds
            mask_low = positions < min_bound
            mask_high = positions > max_bound
            
            positions[mask_low] = 2 * min_bound - positions[mask_low]
            positions[mask_high] = 2 * max_bound - positions[mask_high]
            
            velocities[mask_low] *= -1
            velocities[mask_high] *= -1
            
        elif boundary_type == 'periodic':
            # Wrap around boundaries
            positions = ((positions - min_bound) % (max_bound - min_bound)) + min_bound
            
        elif boundary_type == 'damping':
            # Apply damping near boundaries
            distance_to_bounds = torch.minimum(
                positions - min_bound,
                max_bound - positions
            )
            damping_factor = torch.sigmoid(distance_to_bounds)
            velocities *= damping_factor
            
            # Soft clamping
            positions = torch.tanh(positions / max_bound) * max_bound
        
        return positions, velocities
"""
Enhanced ARC-AGI benchmark runner with hybrid reasoning and test-time compute.
Implements chain-of-thought, ensemble methods, and gravitational abstraction.
"""
import torch.nn.functional as F
import torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
import itertools

# Import GDA modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gda_engine.encoder import MultiModalEncoder
from gda_engine.decoder import MultiModalDecoder
from gda_engine.attractor_core import AttractorCore
from gda_engine.utils import get_device


@dataclass
class Pattern:
    """Represents a discovered pattern in ARC puzzles."""
    pattern_type: str  # 'color_map', 'transformation', 'symmetry', etc.
    confidence: float
    params: Dict[str, Any]
    examples: List[Tuple[np.ndarray, np.ndarray]]


class SymbolicReasoner:
    """Symbolic reasoning module for pattern discovery."""
    
    def __init__(self):
        self.pattern_library = {
            'color_mapping': self._check_color_mapping,
            'geometric_transform': self._check_geometric_transform,
            'counting': self._check_counting_pattern,
            'filling': self._check_filling_pattern,
            'symmetry': self._check_symmetry_pattern,
            'repetition': self._check_repetition_pattern,
            'logical_op': self._check_logical_operation,
        }
        
    def analyze_examples(self, examples):
        """Analyze training examples to discover patterns."""
        patterns = []
        
        for pattern_type, checker in self.pattern_library.items():
            result = checker(examples)
            if result is not None:
                patterns.append(result)
        
        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return patterns
    
    def _check_color_mapping(self, examples):
        """Check if output is a color mapping of input."""
        mappings = []
        
        for inp, out in examples:
            if inp.shape != out.shape:
                return None
            
            # Find color mapping
            color_map = {}
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_color = inp[i, j]
                    out_color = out[i, j]
                    
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            # Inconsistent mapping
                            return None
                    else:
                        color_map[in_color] = out_color
            
            mappings.append(color_map)
        
        # Check consistency across examples
        if all(m == mappings[0] for m in mappings):
            return Pattern(
                pattern_type='color_mapping',
                confidence=0.95,
                params={'mapping': mappings[0]},
                examples=examples
            )
        
        return None
    
    def _check_geometric_transform(self, examples):
        """Check for geometric transformations."""
        transforms = []
        
        for inp, out in examples:
            # Check rotation
            for k in range(4):
                rotated = np.rot90(inp, k)
                if np.array_equal(rotated, out):
                    transforms.append(('rotate', k))
                    break
            
            # Check flip
            if np.array_equal(np.flip(inp, axis=0), out):
                transforms.append(('flip_v', None))
            elif np.array_equal(np.flip(inp, axis=1), out):
                transforms.append(('flip_h', None))
            
            # Check transpose
            if inp.shape[0] == inp.shape[1] and np.array_equal(inp.T, out):
                transforms.append(('transpose', None))
        
        # Check consistency
        if transforms and all(t == transforms[0] for t in transforms):
            return Pattern(
                pattern_type='geometric_transform',
                confidence=0.9,
                params={'transform': transforms[0]},
                examples=examples
            )
        
        return None
    
    def _check_counting_pattern(self, examples):
        """Check if output encodes count information."""
        for inp, out in examples:
            # Check if output size relates to object count
            unique_colors = np.unique(inp)
            non_zero_colors = unique_colors[unique_colors != 0]
            
            for color in non_zero_colors:
                count = np.sum(inp == color)
                
                # Check various counting patterns
                if out.shape == (count, count):
                    return Pattern(
                        pattern_type='counting',
                        confidence=0.7,
                        params={'count_color': int(color), 'output_type': 'square'},
                        examples=examples
                    )
        
        return None
    
    def _check_filling_pattern(self, examples):
        """Check for filling patterns."""
        for inp, out in examples:
            if inp.shape != out.shape:
                continue
            
            # Check if enclosed regions are filled
            # Simple check: border vs interior
            if self._is_border_interior_pattern(inp, out):
                return Pattern(
                    pattern_type='filling',
                    confidence=0.8,
                    params={'fill_type': 'enclosed'},
                    examples=examples
                )
        
        return None
    
    def _check_symmetry_pattern(self, examples):
        """Check for symmetry-based patterns."""
        symmetries = []
        
        for inp, out in examples:
            if inp.shape != out.shape:
                continue
            
            # Check if output completes symmetry
            h, w = inp.shape
            
            # Vertical symmetry
            if np.array_equal(inp[:h//2], out[h//2:][::-1]):
                symmetries.append('vertical')
            
            # Horizontal symmetry
            if np.array_equal(inp[:, :w//2], out[:, w//2:][:, ::-1]):
                symmetries.append('horizontal')
        
        if symmetries:
            return Pattern(
                pattern_type='symmetry',
                confidence=0.75,
                params={'symmetry_type': symmetries[0]},
                examples=examples
            )
        
        return None
    
    def _check_repetition_pattern(self, examples):
        """Check for repetition/tiling patterns."""
        for inp, out in examples:
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Check if output is tiled input
            if h_out % h_in == 0 and w_out % w_in == 0:
                tiles_h = h_out // h_in
                tiles_w = w_out // w_in
                
                # Verify tiling
                is_tiled = True
                for i in range(tiles_h):
                    for j in range(tiles_w):
                        tile = out[i*h_in:(i+1)*h_in, j*w_in:(j+1)*w_in]
                        if not np.array_equal(tile, inp):
                            is_tiled = False
                            break
                
                if is_tiled:
                    return Pattern(
                        pattern_type='repetition',
                        confidence=0.85,
                        params={'tiles': (tiles_h, tiles_w)},
                        examples=examples
                    )
        
        return None
    
    def _check_logical_operation(self, examples):
        """Check for logical operations between colors."""
        # Simplified check for OR/AND/XOR patterns
        for inp, out in examples:
            if inp.shape != out.shape:
                continue
            
            # Check if output is logical OR of specific colors
            mask1 = inp == 1
            mask2 = inp == 2
            
            if np.array_equal(out > 0, mask1 | mask2):
                return Pattern(
                    pattern_type='logical_op',
                    confidence=0.7,
                    params={'operation': 'OR', 'colors': [1, 2]},
                    examples=examples
                )
        
        return None
    
    def _is_border_interior_pattern(self, inp, out):
        """Check if pattern involves border/interior distinction."""
        h, w = inp.shape
        
        # Check if border is one color and interior is filled
        border_mask = np.zeros_like(inp, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        
        interior_mask = ~border_mask
        
        # Check if all border cells have same color in input
        if len(np.unique(inp[border_mask])) == 1:
            # Check if interior is filled with different color in output
            if len(np.unique(out[interior_mask])) == 1:
                return True
        
        return False


class ARCEncoder:
    """Enhanced encoder for ARC grid inputs with pattern awareness."""
    
    def __init__(self, max_grid_size=30, embedding_dim=128):
        self.max_grid_size = max_grid_size
        self.embedding_dim = embedding_dim
        
        # Color embeddings
        self.color_embeddings = torch.nn.Embedding(10, embedding_dim // 4)
        
        # Position encodings
        self.pos_encoding = self._create_position_encoding()
        
        # Pattern embeddings for discovered patterns
        self.pattern_embeddings = torch.nn.Embedding(20, embedding_dim // 4)
        
    def _create_position_encoding(self):
        """Create 2D sinusoidal position encodings."""
        pos_x = torch.arange(self.max_grid_size).float()
        pos_y = torch.arange(self.max_grid_size).float()
        
        pos_x = pos_x.unsqueeze(1).repeat(1, self.max_grid_size)
        pos_y = pos_y.unsqueeze(0).repeat(self.max_grid_size, 1)
        
        div_term = torch.exp(torch.arange(0, self.embedding_dim // 4, 2).float() * 
                           -(np.log(10000.0) / (self.embedding_dim // 4)))
        
        pos_encoding = torch.zeros(self.max_grid_size, self.max_grid_size, self.embedding_dim // 2)
        
        pos_encoding[:, :, 0::4] = torch.sin(pos_x.unsqueeze(2) * div_term)
        pos_encoding[:, :, 1::4] = torch.cos(pos_x.unsqueeze(2) * div_term)
        pos_encoding[:, :, 2::4] = torch.sin(pos_y.unsqueeze(2) * div_term)
        pos_encoding[:, :, 3::4] = torch.cos(pos_y.unsqueeze(2) * div_term)
        
        return pos_encoding
    
    def encode_grid(self, grid, patterns=None):
        """
        Encode ARC grid with pattern information.
        
        Args:
            grid: numpy array (H, W)
            patterns: Optional discovered patterns
            
        Returns:
            Mass vectors and masses
        """
        H, W = grid.shape
        device = get_device()
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).long().to(device)
        
        # Color embeddings
        color_emb = self.color_embeddings(grid_tensor)
        
        # Position embeddings
        pos_emb = self.pos_encoding[:H, :W, :].to(device)
        
        # Structural features
        structure_features = self._compute_structure_features(grid_tensor)
        
        # Pattern features if available
        if patterns:
            pattern_features = self._encode_patterns(patterns, H, W, device)
        else:
            pattern_features = torch.zeros(H, W, self.embedding_dim // 4).to(device)
        
        # Combine all features
        combined_features = torch.cat([
            color_emb, pos_emb, structure_features, pattern_features
        ], dim=-1)
        
        # Create mass vectors
        mass_vectors = combined_features.reshape(-1, self.embedding_dim)
        
        # Compute masses with pattern weighting
        base_masses = (grid_tensor > 0).float().reshape(-1) * 0.1 + 0.01
        
        if patterns:
            # Higher mass for cells involved in patterns
            pattern_weight = 1.0 + 0.5 * len(patterns)
            masses = base_masses * pattern_weight
        else:
            masses = base_masses
        
        # Add global context
        global_features = self._compute_global_features(
            grid_tensor, color_emb, structure_features
        )
        
        mass_vectors = torch.cat([global_features.unsqueeze(0), mass_vectors], dim=0)
        masses = torch.cat([torch.tensor([0.3], device=device), masses], dim=0)
        
        return mass_vectors.unsqueeze(0), masses.unsqueeze(0)
    
    def _compute_structure_features(self, grid_tensor):
        """Compute structural features of the grid."""
        H, W = grid_tensor.shape
        device = grid_tensor.device
        features = torch.zeros(H, W, self.embedding_dim // 4).to(device)
        
        # Edge detection
        padded = F.pad(grid_tensor.float().unsqueeze(0).unsqueeze(0), 
                      (1, 1, 1, 1), mode='constant', value=0)
        
        # Sobel filters
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().to(device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().to(device)
        
        edge_x = F.conv2d(padded, sobel_x).squeeze()
        edge_y = F.conv2d(padded, sobel_y).squeeze()
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        features[:, :, 0] = edge_magnitude
        
        # Local pattern detection
        for i in range(H):
            for j in range(W):
                # 3x3 neighborhood
                i_start, i_end = max(0, i-1), min(H, i+2)
                j_start, j_end = max(0, j-1), min(W, j+2)
                
                neighborhood = grid_tensor[i_start:i_end, j_start:j_end]
                
                # Features
                features[i, j, 1] = neighborhood.float().mean()  # Local density
                features[i, j, 2] = len(torch.unique(neighborhood))  # Local variety
                features[i, j, 3] = (neighborhood == grid_tensor[i, j]).float().mean()  # Similarity
        
        return features
    
    def _encode_patterns(self, patterns, H, W, device):
        """Encode discovered patterns into features."""
        pattern_features = torch.zeros(H, W, self.embedding_dim // 4).to(device)
        
        for idx, pattern in enumerate(patterns[:5]):  # Top 5 patterns
            pattern_id = hash(pattern.pattern_type) % 20
            pattern_emb = self.pattern_embeddings(
                torch.tensor([pattern_id], device=device)
            )
            
            # Weight by confidence
            weight = pattern.confidence
            pattern_features += weight * pattern_emb.view(1, 1, -1)
        
        return pattern_features
    
    def _compute_global_features(self, grid_tensor, color_emb, structure_features):
        """Compute global features of the grid."""
        device = grid_tensor.device
        
        # Global statistics
        unique_colors = len(torch.unique(grid_tensor))
        density = (grid_tensor > 0).float().mean()
        
        # Color distribution
        color_hist = torch.zeros(10, device=device)
        for c in range(10):
            color_hist[c] = (grid_tensor == c).float().mean()
        
        # Aggregate embeddings
        mean_color = color_emb.mean(dim=(0, 1))
        mean_structure = structure_features.mean(dim=(0, 1))
        
        # Shape information
        H, W = grid_tensor.shape
        shape_features = torch.tensor([
            H / 30.0, W / 30.0, H/W, min(H, W) / max(H, W)
        ], device=device)
        
        # Combine
        global_features = torch.cat([
            mean_color,
            mean_structure,
            color_hist,
            shape_features,
            torch.tensor([unique_colors / 10.0, density], device=device)
        ])
        
        # Project to correct dimension
        if global_features.shape[0] != self.embedding_dim:
            projection = torch.nn.Linear(
                global_features.shape[0], 
                self.embedding_dim
            ).to(device)
            global_features = projection(global_features)
        
        return global_features


class ARCDecoder:
    """Enhanced decoder with pattern-guided generation."""
    
    def __init__(self, embedding_dim=128, max_grid_size=30):
        self.embedding_dim = embedding_dim
        self.max_grid_size = max_grid_size
        
        # Main decoder network
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 10)
        ).to(get_device())
        
        # Pattern-specific decoders
        self.pattern_decoders = torch.nn.ModuleDict({
            'color_mapping': self._make_simple_decoder(),
            'geometric_transform': self._make_simple_decoder(),
            'counting': self._make_counting_decoder(),
            'filling': self._make_spatial_decoder(),
            'symmetry': self._make_spatial_decoder(),
        })
    
    def _make_simple_decoder(self):
        """Create simple pattern decoder."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        ).to(get_device())
    
    def _make_counting_decoder(self):
        """Create decoder for counting patterns."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim + 1, 64),  # +1 for count
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        ).to(get_device())
    
    def _make_spatial_decoder(self):
        """Create decoder with spatial awareness."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim + 4, 128),  # +4 for position
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        ).to(get_device())
    
    def decode_to_grid(self, attractor_field, target_shape, pattern=None):
        """
        Decode attractor field to grid with pattern guidance.
        
        Args:
            attractor_field: Compressed representation (1, D)
            target_shape: Target grid shape (H, W)
            pattern: Optional discovered pattern
            
        Returns:
            Decoded grid
        """
        H, W = target_shape
        device = attractor_field.device
        
        if pattern and pattern.pattern_type in self.pattern_decoders:
            # Use pattern-specific decoder
            grid = self._decode_with_pattern(
                attractor_field, target_shape, pattern
            )
        else:
            # General decoding
            grid = self._decode_general(attractor_field, target_shape)
        
        return grid
    
    def _decode_general(self, attractor_field, target_shape):
        """General grid decoding."""
        H, W = target_shape
        device = attractor_field.device
        
        # Expand attractor field
        expanded = attractor_field.repeat(H * W, 1)
        
        # Add position information
        pos_x = torch.arange(W, device=device).repeat(H).float() / W
        pos_y = torch.arange(H, device=device).repeat_interleave(W).float() / H
        
        # Create position features
        pos_features = torch.stack([
            pos_x, pos_y,
            (pos_x - 0.5).abs(),
            (pos_y - 0.5).abs()
        ], dim=1)
        
        # Project positions
        pos_proj = torch.nn.Linear(4, self.embedding_dim).to(device)
        pos_emb = pos_proj(pos_features)
        
        # Combine with attractor field
        combined = expanded + 0.2 * pos_emb
        
        # Decode
        color_logits = self.decoder(combined)
        grid = color_logits.argmax(dim=-1).reshape(H, W)
        
        return grid.cpu().numpy()
    
    def _decode_with_pattern(self, attractor_field, target_shape, pattern):
        """Decode using discovered pattern."""
        if pattern.pattern_type == 'color_mapping':
            return self._apply_color_mapping(target_shape, pattern)
        
        elif pattern.pattern_type == 'geometric_transform':
            return self._apply_geometric_transform(target_shape, pattern)
        
        elif pattern.pattern_type == 'counting':
            return self._apply_counting_pattern(attractor_field, pattern)
        
        elif pattern.pattern_type == 'filling':
            return self._apply_filling_pattern(attractor_field, target_shape, pattern)
        
        else:
            # Fallback to general decoding
            return self._decode_general(attractor_field, target_shape)
    
    def _apply_color_mapping(self, target_shape, pattern):
        """Apply color mapping pattern."""
        # Get input from pattern examples
        if pattern.examples:
            inp, _ = pattern.examples[0]
            if inp.shape == target_shape:
                # Apply mapping
                mapping = pattern.params['mapping']
                output = np.zeros_like(inp)
                
                for in_color, out_color in mapping.items():
                    output[inp == in_color] = out_color
                
                return output
        
        return np.zeros(target_shape, dtype=int)
    
    def _apply_geometric_transform(self, target_shape, pattern):
        """Apply geometric transformation."""
        if pattern.examples:
            inp, _ = pattern.examples[0]
            transform_type, param = pattern.params['transform']
            
            if transform_type == 'rotate':
                return np.rot90(inp, param)
            elif transform_type == 'flip_v':
                return np.flip(inp, axis=0)
            elif transform_type == 'flip_h':
                return np.flip(inp, axis=1)
            elif transform_type == 'transpose':
                return inp.T
        
        return np.zeros(target_shape, dtype=int)
    
    def _apply_counting_pattern(self, attractor_field, pattern):
        """Apply counting pattern."""
        count_color = pattern.params.get('count_color', 1)
        output_type = pattern.params.get('output_type', 'square')
        
        # Get count from examples
        if pattern.examples:
            inp, _ = pattern.examples[0]
            count = np.sum(inp == count_color)
            
            if output_type == 'square':
                size = int(np.sqrt(count))
                if size * size == count:
                    return np.ones((size, size), dtype=int) * count_color
        
        return np.array([[1]])
    
    def _apply_filling_pattern(self, attractor_field, target_shape, pattern):
        """Apply filling pattern."""
        if pattern.examples:
            inp, out = pattern.examples[0]
            
            if inp.shape == target_shape:
                # Simple border filling
                output = inp.copy()
                H, W = output.shape
                
                # Fill interior
                if H > 2 and W > 2:
                    # Detect fill color from examples
                    interior_color = out[1:-1, 1:-1].flat[0]
                    output[1:-1, 1:-1] = interior_color
                
                return output
        
        return np.zeros(target_shape, dtype=int)


class ChainOfThoughtReasoner:
    """Implements chain-of-thought reasoning for ARC puzzles."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.reasoning_steps = []
        
    def reason(self, examples, test_input):
        """
        Apply chain-of-thought reasoning.
        
        Args:
            examples: Training examples
            test_input: Test input grid
            
        Returns:
            Reasoning chain and predictions
        """
        self.reasoning_steps = []
        
        # Step 1: Analyze grid properties
        self._analyze_grids(examples, test_input)
        
        # Step 2: Identify patterns
        patterns = self._identify_patterns(examples)
        
        # Step 3: Generate hypotheses
        hypotheses = self._generate_hypotheses(patterns, examples)
        
        # Step 4: Test hypotheses
        valid_hypotheses = self._test_hypotheses(hypotheses, examples)
        
        # Step 5: Apply to test
        predictions = self._apply_hypotheses(valid_hypotheses, test_input)
        
        return {
            'reasoning_steps': self.reasoning_steps,
            'patterns': patterns,
            'hypotheses': valid_hypotheses,
            'predictions': predictions
        }
    
    def _analyze_grids(self, examples, test_input):
        """Analyze basic properties of grids."""
        step = "Analyzing grid properties..."
        
        # Input sizes
        input_sizes = [inp.shape for inp, _ in examples]
        output_sizes = [out.shape for _, out in examples]
        
        # Color usage
        input_colors = [set(np.unique(inp)) for inp, _ in examples]
        output_colors = [set(np.unique(out)) for _, out in examples]
        
        analysis = {
            'input_sizes': input_sizes,
            'output_sizes': output_sizes,
            'size_change': any(i != o for i, o in zip(input_sizes, output_sizes)),
            'input_colors': input_colors,
            'output_colors': output_colors,
            'new_colors': any(o - i for i, o in zip(input_colors, output_colors)),
            'test_size': test_input.shape,
            'test_colors': set(np.unique(test_input))
        }
        
        self.reasoning_steps.append({
            'step': step,
            'analysis': analysis
        })
        
        return analysis
    
    def _identify_patterns(self, examples):
        """Identify potential patterns."""
        step = "Identifying patterns..."
        
        patterns = []
        
        # Check transformations
        for idx, (inp, out) in enumerate(examples):
            # Size relationships
            if inp.shape != out.shape:
                h_ratio = out.shape[0] / inp.shape[0]
                w_ratio = out.shape[1] / inp.shape[1]
                
                if h_ratio == w_ratio and h_ratio == int(h_ratio):
                    patterns.append({
                        'type': 'scaling',
                        'factor': int(h_ratio),
                        'example': idx
                    })
            
            # Object detection
            objects_in = self._detect_objects(inp)
            objects_out = self._detect_objects(out)
            
            if len(objects_in) == len(objects_out):
                patterns.append({
                    'type': 'object_transform',
                    'num_objects': len(objects_in),
                    'example': idx
                })
        
        self.reasoning_steps.append({
            'step': step,
            'patterns': patterns
        })
        
        return patterns
    
    def _generate_hypotheses(self, patterns, examples):
        """Generate testable hypotheses."""
        step = "Generating hypotheses..."
        
        hypotheses = []
        
        # Based on patterns
        for pattern in patterns:
            if pattern['type'] == 'scaling':
                hypotheses.append({
                    'type': 'uniform_scaling',
                    'params': {'factor': pattern['factor']},
                    'confidence': 0.8
                })
            
            elif pattern['type'] == 'object_transform':
                hypotheses.append({
                    'type': 'per_object_transform',
                    'params': {'preserve_count': True},
                    'confidence': 0.7
                })
        
        # Always add common hypotheses
        hypotheses.extend([
            {
                'type': 'color_replacement',
                'params': {},
                'confidence': 0.6
            },
            {
                'type': 'pattern_completion',
                'params': {},
                'confidence': 0.5
            }
        ])
        
        self.reasoning_steps.append({
            'step': step,
            'num_hypotheses': len(hypotheses)
        })
        
        return hypotheses
    
    def _test_hypotheses(self, hypotheses, examples):
        """Test hypotheses against examples."""
        step = "Testing hypotheses..."
        
        valid_hypotheses = []
        
        for hypothesis in hypotheses:
            success_count = 0
            
            for inp, expected_out in examples:
                predicted = self._apply_hypothesis(hypothesis, inp)
                
                if predicted is not None and np.array_equal(predicted, expected_out):
                    success_count += 1
            
            accuracy = success_count / len(examples)
            
            if accuracy > 0.5:  # At least works on half
                hypothesis['accuracy'] = accuracy
                valid_hypotheses.append(hypothesis)
        
        # Sort by accuracy
        valid_hypotheses.sort(key=lambda h: h['accuracy'], reverse=True)
        
        self.reasoning_steps.append({
            'step': step,
            'valid_count': len(valid_hypotheses),
            'best_accuracy': valid_hypotheses[0]['accuracy'] if valid_hypotheses else 0
        })
        
        return valid_hypotheses
    
    def _apply_hypotheses(self, hypotheses, test_input):
        """Apply valid hypotheses to test input."""
        predictions = []
        
        for hypothesis in hypotheses:
            prediction = self._apply_hypothesis(hypothesis, test_input)
            
            if prediction is not None:
                predictions.append({
                    'hypothesis': hypothesis,
                    'prediction': prediction
                })
        
        return predictions
    
    def _apply_hypothesis(self, hypothesis, grid):
        """Apply a single hypothesis to a grid."""
        h_type = hypothesis['type']
        
        if h_type == 'uniform_scaling':
            factor = hypothesis['params']['factor']
            return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
        
        elif h_type == 'color_replacement':
            # Simple example: increment all non-zero colors
            output = grid.copy()
            output[output > 0] = (output[output > 0] % 9) + 1
            return output
        
        # Add more hypothesis applications as needed
        
        return None
    
    def _detect_objects(self, grid):
        """Detect distinct objects in grid."""
        from scipy import ndimage
        
        objects = []
        
        for color in range(1, 10):
            mask = (grid == color)
            if mask.any():
                labeled, num = ndimage.label(mask)
                
                for i in range(1, num + 1):
                    obj_mask = (labeled == i)
                    objects.append({
                        'color': color,
                        'mask': obj_mask,
                        'size': obj_mask.sum()
                    })
        
        return objects


class ARCGDASolver:
    """Enhanced ARC solver with hybrid reasoning and ensembles."""
    
    def __init__(self, model_path=None, use_symbolic=True, use_cot=True, ensemble_size=3):
        self.device = get_device()
        
        # Core components
        self.encoder = ARCEncoder()
        self.decoder = ARCDecoder()
        self.attractor_core = AttractorCore(
            latent_dim=128,
            num_attractors=5,
            gravity_strength=0.01,
            damping=0.9,
            use_hamiltonian=True,
            use_boundary_pinn=True
        ).to(self.device)
        
        # Reasoning components
        self.use_symbolic = use_symbolic
        if use_symbolic:
            self.symbolic_reasoner = SymbolicReasoner()
        
        self.use_cot = use_cot
        if use_cot:
            self.cot_reasoner = ChainOfThoughtReasoner()
        
        # Ensemble settings
        self.ensemble_size = ensemble_size
        
        # Memory for few-shot learning
        self.memory_bank = []
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def solve_puzzle(self, train_examples, test_input):
        """
        Solve ARC puzzle using hybrid approach.
        
        Args:
            train_examples: List of (input, output) grids
            test_input: Test input grid
            
        Returns:
            Predicted output grid
        """
        # Step 1: Symbolic reasoning
        patterns = []
        if self.use_symbolic:
            patterns = self.symbolic_reasoner.analyze_examples(train_examples)
        
        # Step 2: Chain-of-thought reasoning
        cot_predictions = []
        if self.use_cot:
            cot_result = self.cot_reasoner.reason(train_examples, test_input)
            cot_predictions = cot_result['predictions']
        
        # Step 3: Gravitational reasoning (ensemble)
        gda_predictions = []
        for _ in range(self.ensemble_size):
            pred = self._solve_with_attractors(
                train_examples, test_input, patterns
            )
            gda_predictions.append(pred)
        
        # Step 4: Combine predictions
        all_predictions = []
        
        # Add symbolic predictions
        if patterns and patterns[0].confidence > 0.8:
            all_predictions.append({
                'source': 'symbolic',
                'prediction': self._apply_best_pattern(patterns[0], test_input),
                'confidence': patterns[0].confidence
            })
        
        # Add CoT predictions
        for cot_pred in cot_predictions[:2]:  # Top 2
            all_predictions.append({
                'source': 'cot',
                'prediction': cot_pred['prediction'],
                'confidence': cot_pred['hypothesis']['accuracy']
            })
        
        # Add GDA predictions
        for i, gda_pred in enumerate(gda_predictions):
            all_predictions.append({
                'source': f'gda_{i}',
                'prediction': gda_pred,
                'confidence': 0.7  # Default confidence
            })
        
        # Select best prediction
        if all_predictions:
            # Sort by confidence
            all_predictions.sort(key=lambda p: p['confidence'], reverse=True)
            
            # Voting for high-confidence predictions
            high_conf_preds = [p for p in all_predictions if p['confidence'] > 0.7]
            
            if len(high_conf_preds) >= 2:
                # Vote among high confidence predictions
                return self._vote_predictions([p['prediction'] for p in high_conf_preds])
            else:
                # Return highest confidence
                return all_predictions[0]['prediction']
        
        # Fallback: return zeros
        return np.zeros_like(test_input)
    
    def _solve_with_attractors(self, train_examples, test_input, patterns):
        """Solve using gravitational attractor dynamics."""
        # Encode training examples with patterns
        train_attractors = []
        transformations = []
        
        for inp, out in train_examples:
            # Encode with pattern information
            inp_masses, inp_weights = self.encoder.encode_grid(inp, patterns)
            out_masses, out_weights = self.encoder.encode_grid(out, patterns)
            
            # Run through attractor dynamics with enhanced stability
            inp_field, _ = self.attractor_core(
                inp_masses, 
                num_iterations=50,
                adaptive_steps=True
            )
            
            out_field, _ = self.attractor_core(
                out_masses, 
                num_iterations=50,
                adaptive_steps=True
            )
            
            train_attractors.append({
                'input': inp_field,
                'output': out_field,
                'output_shape': out.shape,
                'input_grid': inp,
                'output_grid': out
            })
            
            # Learn transformation
            transformation = out_field - inp_field
            transformations.append(transformation)
        
        # Encode test input
        test_masses, test_weights = self.encoder.encode_grid(test_input, patterns)
        test_field, convergence_info = self.attractor_core(
            test_masses, 
            num_iterations=100,  # More iterations for test
            adaptive_steps=True,
            return_trajectory=True
        )
        
        # Find best matching example using gravitational similarity
        similarities = []
        
        for train_data in train_attractors:
            # Multi-scale similarity
            sim_global = -torch.norm(test_field - train_data['input'])
            
            # Local similarity (if shapes match)
            sim_local = 0
            if test_input.shape == train_data['input_grid'].shape:
                test_local = test_masses.reshape(-1, self.encoder.embedding_dim)
                train_local = self.encoder.encode_grid(
                    train_data['input_grid'], patterns
                )[0].reshape(-1, self.encoder.embedding_dim)
                
                sim_local = F.cosine_similarity(
                    test_local, train_local, dim=1
                ).mean()
            
            # Combined similarity
            similarity = 0.7 * sim_global + 0.3 * sim_local
            similarities.append(similarity)
        
        # Select best match
        best_idx = torch.tensor(similarities).argmax().item()
        best_match = train_attractors[best_idx]
        best_transformation = transformations[best_idx]
        
        # Determine output shape
        if all(t['output_shape'] == train_attractors[0]['output_shape'] 
               for t in train_attractors):
            target_shape = train_attractors[0]['output_shape']
        else:
            # Shape varies - try to infer
            target_shape = self._infer_output_shape(
                train_examples, test_input
            )
        
        # Apply transformation
        predicted_field = test_field + best_transformation
        
        # Decode with pattern guidance
        best_pattern = patterns[0] if patterns else None
        predicted_grid = self.decoder.decode_to_grid(
            predicted_field, target_shape, best_pattern
        )
        
        return predicted_grid
    
    def _apply_best_pattern(self, pattern, test_input):
        """Apply the best discovered pattern."""
        if pattern.pattern_type == 'color_mapping':
            mapping = pattern.params['mapping']
            output = np.zeros_like(test_input)
            for in_color, out_color in mapping.items():
                output[test_input == in_color] = out_color
            return output
        
        # Add more pattern applications
        
        return test_input
    
    def _vote_predictions(self, predictions):
        """Vote among multiple predictions."""
        if not predictions:
            return None
        
        # Find most common prediction
        from collections import Counter
        
        # Convert to tuples for hashing
        pred_tuples = [tuple(p.flatten()) for p in predictions]
        counter = Counter(pred_tuples)
        
        # Get most common
        most_common_tuple, _ = counter.most_common(1)[0]
        
        # Convert back to array
        shape = predictions[0].shape
        return np.array(most_common_tuple).reshape(shape)
    
    def _infer_output_shape(self, train_examples, test_input):
        """Infer output shape from examples."""
        # Check if output shape is consistent
        output_shapes = [out.shape for _, out in train_examples]
        
        if all(s == output_shapes[0] for s in output_shapes):
            return output_shapes[0]
        
        # Check if shape relates to input
        for inp, out in train_examples:
            if inp.shape == test_input.shape:
                return out.shape
        
        # Default: same as input
        return test_input.shape
    
    def batch_solve(self, puzzles):
        """Solve multiple puzzles with detailed metrics."""
        results = {
            'correct': 0,
            'total': 0,
            'predictions': [],
            'times': [],
            'reasoning_used': defaultdict(int)
        }
        
        for puzzle in tqdm(puzzles, desc="Solving puzzles"):
            start_time = time.time()
            
            # Extract examples
            train_examples = [(ex['input'], ex['output']) for ex in puzzle['train']]
            test_input = puzzle['test'][0]['input']
            test_output = puzzle['test'][0]['output']
            
            # Solve
            try:
                prediction = self.solve_puzzle(train_examples, test_input)
                
                # Check correctness
                correct = np.array_equal(prediction, test_output)
                results['correct'] += int(correct)
                
                # Track reasoning methods used
                results['reasoning_used']['total'] += 1
                if self.use_symbolic:
                    results['reasoning_used']['symbolic'] += 1
                if self.use_cot:
                    results['reasoning_used']['cot'] += 1
                results['reasoning_used']['gda'] += 1
                
                results['predictions'].append({
                    'puzzle_id': puzzle.get('id', results['total']),
                    'prediction': prediction,
                    'ground_truth': test_output,
                    'correct': correct
                })
                
            except Exception as e:
                print(f"Error solving puzzle: {e}")
                results['predictions'].append({
                    'puzzle_id': puzzle.get('id', results['total']),
                    'prediction': None,
                    'ground_truth': test_output,
                    'correct': False,
                    'error': str(e)
                })
            
            results['total'] += 1
            results['times'].append(time.time() - start_time)
        
        # Compute statistics
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
        results['avg_time'] = np.mean(results['times']) if results['times'] else 0
        
        return results
    
def run_benchmark(dataset_path=None, num_puzzles=10):
        """
        Run ARC-AGI benchmark using ARCGDASolver.

        Args:
            dataset_path: Path to ARC dataset JSON files (default: None, uses default path)
            num_puzzles: Number of puzzles to process (default: 10)

        Returns:
            Dictionary with benchmark results
        """
        if dataset_path is None:
            # Replace with the actual default path to your ARC dataset
            dataset_path = "/path/to/arc/dataset"  # Update this path

        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path {dataset_path} does not exist")
            return {
                'correct': 0,
                'total': 0,
                'predictions': [],
                'times': [],
                'accuracy': 0.0,
                'avg_time': 0.0,
                'reasoning_used': defaultdict(int)
            }

        # Load puzzles
        puzzles = []
        for file in os.listdir(dataset_path)[:num_puzzles]:
            if file.endswith(".json"):
                with open(os.path.join(dataset_path, file)) as f:
                    puzzle = json.load(f)
                    # Ensure puzzle has required structure
                    if 'train' in puzzle and 'test' in puzzle:
                        puzzles.append(puzzle)

        if not puzzles:
            print("No valid ARC puzzles found in dataset")
            return {
                'correct': 0,
                'total': 0,
                'predictions': [],
                'times': [],
                'accuracy': 0.0,
                'avg_time': 0.0,
                'reasoning_used': defaultdict(int)
            }

        # Initialize solver
        solver = ARCGDASolver(use_symbolic=True, use_cot=True, ensemble_size=3)
        
        # Solve puzzles
        results = solver.batch_solve(puzzles)
        
        print(f"Benchmark completed: {results['correct']}/{results['total']} correct "
            f"(Accuracy: {results['accuracy']:.2%}, Avg time: {results['avg_time']:.2f}s)")
        
        return results
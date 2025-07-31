"""
Enhanced multimodal fusion with hierarchical attention and evidence-based alignment.
Implements state-of-the-art cross-modal fusion techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from .gravity_kernel import GravityKernel


class Reshape(nn.Module):
    """Custom Reshape layer since torch.nn doesn't have one."""
    
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


class HierarchicalAttention(nn.Module):
    """Hierarchical attention for multi-scale feature fusion."""
    
    def __init__(self, dim, num_levels=3, heads=8):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.heads = heads
        
        # Multi-level attention layers
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True)
            for _ in range(num_levels)
        ])
        
        # Level combination
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
        # Feature pyramids
        self.downsample = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])
        
        self.upsample = nn.ModuleList([
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])
    
    def forward(self, query, key, value):
        """Apply hierarchical attention across multiple scales."""
        B, N, D = query.shape
        
        # Create feature pyramids
        query_pyramid = [query]
        key_pyramid = [key]
        value_pyramid = [value]
        
        # Downsample for coarser levels
        for i in range(self.num_levels - 1):
            # Reshape for conv1d
            q_down = self.downsample[i](query_pyramid[-1].transpose(1, 2)).transpose(1, 2)
            k_down = self.downsample[i](key_pyramid[-1].transpose(1, 2)).transpose(1, 2)
            v_down = self.downsample[i](value_pyramid[-1].transpose(1, 2)).transpose(1, 2)
            
            query_pyramid.append(q_down)
            key_pyramid.append(k_down)
            value_pyramid.append(v_down)
        
        # Apply attention at each level
        attended_levels = []
        attention_weights = []
        
        for i in range(self.num_levels):
            attended, weights = self.level_attentions[i](
                query_pyramid[i], key_pyramid[i], value_pyramid[i]
            )
            
            # Upsample back to original resolution
            if i > 0:
                for j in range(i):
                    attended = self.upsample[j](attended.transpose(1, 2)).transpose(1, 2)
            
            # Ensure correct shape
            if attended.shape[1] != N:
                attended = F.interpolate(
                    attended.transpose(1, 2), size=N, mode='linear'
                ).transpose(1, 2)
            
            attended_levels.append(attended)
            attention_weights.append(weights)
        
        # Combine levels with learned weights
        level_weights = F.softmax(self.level_weights, dim=0)
        combined = sum(w * att for w, att in zip(level_weights, attended_levels))
        
        return combined, attention_weights


class EvidenceBasedFusion(nn.Module):
    """Evidence-based fusion with uncertainty modeling."""
    
    def __init__(self, dim, num_modalities=3):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        
        # Evidence networks for each modality
        self.evidence_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 2)  # (belief, uncertainty)
            ) for _ in range(num_modalities)
        ])
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(dim * num_modalities + num_modalities * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, features_list):
        """
        Fuse features with evidence-based weighting.
        
        Args:
            features_list: List of features from different modalities
            
        Returns:
            Fused features with uncertainty estimates
        """
        beliefs = []
        uncertainties = []
        
        # Compute evidence for each modality
        for i, features in enumerate(features_list):
            evidence = self.evidence_nets[i](features)
            belief = torch.sigmoid(evidence[:, 0])
            uncertainty = torch.exp(evidence[:, 1])
            
            beliefs.append(belief)
            uncertainties.append(uncertainty)
        
        # Dempster-Shafer combination
        combined_belief = beliefs[0]
        combined_uncertainty = uncertainties[0]
        
        for i in range(1, len(beliefs)):
            # Combine beliefs
            k = 1 - combined_belief * beliefs[i]  # Conflict
            combined_belief = (combined_belief * beliefs[i]) / (k + 1e-8)
            
            # Combine uncertainties
            combined_uncertainty = combined_uncertainty * uncertainties[i]
        
        # Weight features by evidence
        weighted_features = []
        for i, features in enumerate(features_list):
            weight = beliefs[i] / (uncertainties[i] + 1e-8)
            weight = weight.unsqueeze(-1)
            weighted_features.append(features * weight)
        
        # Concatenate all information
        evidence_info = torch.stack(beliefs + uncertainties, dim=-1)
        all_features = torch.cat(weighted_features + [evidence_info], dim=-1)
        
        # Final fusion
        fused = self.fusion_net(all_features)
        
        return fused, combined_belief, combined_uncertainty


class RelationalCrossModal(nn.Module):
    """Relational learning for cross-modal correlations."""
    
    def __init__(self, dim, num_relations=4):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations
        
        # Relation networks
        self.relation_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 1)
            ) for _ in range(num_relations)
        ])
        
        # Relation aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(dim * num_relations, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, features1, features2):
        """
        Learn relations between modalities.
        
        Args:
            features1, features2: Features from two modalities (B, N, D)
            
        Returns:
            Relational features
        """
        B, N1, D = features1.shape
        _, N2, _ = features2.shape
        
        # Compute all pairwise relations
        relations = []
        
        for rel_idx in range(self.num_relations):
            # Compute relation scores for all pairs
            rel_scores = []
            
            for i in range(N1):
                for j in range(N2):
                    pair = torch.cat([features1[:, i], features2[:, j]], dim=-1)
                    score = self.relation_nets[rel_idx](pair)
                    rel_scores.append(score)
            
            rel_scores = torch.stack(rel_scores, dim=1)  # (B, N1*N2, 1)
            rel_scores = rel_scores.view(B, N1, N2)
            
            # Aggregate relations
            # Use attention-like aggregation
            rel_weights = F.softmax(rel_scores, dim=-1)
            rel_features = torch.bmm(rel_weights, features2)  # (B, N1, D)
            
            relations.append(rel_features)
        
        # Combine all relations
        all_relations = torch.cat(relations, dim=-1)  # (B, N1, D*num_relations)
        
        # Final aggregation
        relational_features = self.aggregator(all_relations)
        
        return relational_features


class CrossModalAlignment(nn.Module):
    """Enhanced cross-modal alignment with multiple strategies."""
    
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(
            dim, num_levels=3, heads=num_heads
        )
        
        # Standard cross-attention as baseline
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Relational learning
        self.relational = RelationalCrossModal(dim)
        
        # Alignment projection
        self.align_proj = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # 3 sources: hierarchical, standard, relational
            nn.ReLU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )
        
        # Modality-specific normalizations
        self.modal_norms = nn.ModuleDict({
            'text': nn.LayerNorm(dim),
            'image': nn.LayerNorm(dim),
            'audio': nn.LayerNorm(dim)
        })
        
        # Learnable temperature for alignment
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, embeddings_dict):
        """
        Align embeddings from different modalities.
        
        Args:
            embeddings_dict: Dict with modality names as keys
            
        Returns:
            Aligned embeddings dict
        """
        aligned = {}
        
        # Normalize each modality
        for modality, embeddings in embeddings_dict.items():
            if modality in self.modal_norms:
                aligned[modality] = self.modal_norms[modality](embeddings)
            else:
                aligned[modality] = F.layer_norm(embeddings, [embeddings.shape[-1]])
        
        # Cross-modal attention for each pair
        modalities = list(aligned.keys())
        
        for i, mod1 in enumerate(modalities):
            aligned_features = []
            
            for j, mod2 in enumerate(modalities):
                if i != j:
                    # Multiple alignment strategies
                    
                    # 1. Hierarchical attention
                    hier_attended, _ = self.hierarchical_attention(
                        aligned[mod1], aligned[mod2], aligned[mod2]
                    )
                    
                    # 2. Standard cross-attention
                    std_attended, _ = self.cross_attention(
                        aligned[mod1], aligned[mod2], aligned[mod2]
                    )
                    
                    # 3. Relational features
                    rel_features = self.relational(aligned[mod1], aligned[mod2])
                    
                    # Combine strategies
                    combined = torch.cat([
                        hier_attended, std_attended, rel_features
                    ], dim=-1)
                    
                    aligned_features.append(combined)
            
            if aligned_features:
                # Aggregate alignments from all other modalities
                aggregated = torch.stack(aligned_features, dim=0).mean(dim=0)
                
                # Project and residual connection
                projected = self.align_proj(aggregated)
                aligned[mod1] = aligned[mod1] + 0.1 * projected
        
        return aligned


class AttractorFusion(nn.Module):
    """Enhanced attractor fusion with evidence-based combination."""
    
    def __init__(self, dim=128, gravity_strength=1e-3):
        super().__init__()
        self.dim = dim
        
        # Gravity kernel for fusion dynamics
        self.gravity_kernel = GravityKernel(G=gravity_strength, use_cuda=True)
        
        # Evidence-based fusion
        self.evidence_fusion = EvidenceBasedFusion(dim, num_modalities=3)
        
        # Dynamic weight prediction
        self.weight_predictor = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        # Unified projection with residual
        self.unified_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )
        
        # Ambiguity detection
        self.ambiguity_detector = nn.Sequential(
            nn.Linear(dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Missing modality embeddings
        self.missing_modality_embeds = nn.Parameter(
            torch.randn(3, dim) * 0.1
        )
    
    def forward(self, attractor_fields):
        """
        Fuse multiple attractor fields with evidence weighting.
        
        Args:
            attractor_fields: Dict of attractor fields by modality
            
        Returns:
            Unified field with fusion metadata
        """
        # Prepare fields list
        available_fields = []
        modality_order = ['text', 'image', 'audio']
        field_indices = []
        
        for i, mod in enumerate(modality_order):
            if mod in attractor_fields:
                available_fields.append(attractor_fields[mod])
                field_indices.append(i)
        
        if len(available_fields) == 1:
            # Single modality
            unified = self.unified_proj(available_fields[0])
            return unified, {'single_modality': True}
        
        # Pad missing modalities with learned embeddings
        padded_fields = []
        for i in range(3):
            if i in field_indices:
                idx = field_indices.index(i)
                padded_fields.append(available_fields[idx])
            else:
                # Use learned embedding for missing modality
                batch_size = available_fields[0].shape[0]
                missing_embed = self.missing_modality_embeds[i].unsqueeze(0).expand(
                    batch_size, -1
                )
                padded_fields.append(missing_embed)
        
        # Evidence-based fusion
        fused_evidence, belief, uncertainty = self.evidence_fusion(padded_fields)
        
        # Dynamic weight prediction
        concat_fields = torch.cat(padded_fields, dim=-1)
        dynamic_weights = self.weight_predictor(concat_fields)
        
        # Weighted combination
        weighted_fields = sum(
            w.unsqueeze(-1) * f 
            for w, f in zip(dynamic_weights.unbind(-1), padded_fields)
        )
        
        # Ambiguity detection
        ambiguity_score = self.ambiguity_detector(concat_fields)
        
        # Final fusion with evidence and weighted combination
        alpha = (1 - ambiguity_score).unsqueeze(-1)  # Lower weight for ambiguous
        unified = alpha * fused_evidence + (1 - alpha) * weighted_fields
        
        # Apply projection
        unified = self.unified_proj(unified)
        
        # Metadata
        fusion_info = {
            'dynamic_weights': dynamic_weights,
            'belief': belief,
            'uncertainty': uncertainty,
            'ambiguity': ambiguity_score,
            'modalities_present': [modality_order[i] for i in field_indices]
        }
        
        return unified, fusion_info


class MultiModalFusion(nn.Module):
    """Complete multimodal fusion pipeline with advanced techniques."""
    
    def __init__(self, dim=128, gravity_strength=1e-3):
        super().__init__()
        
        # Components
        self.alignment = CrossModalAlignment(dim=dim)
        self.fusion = AttractorFusion(dim=dim, gravity_strength=gravity_strength)
        
        # Contrastive projection
        self.contrastive_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2)
        )
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # VMD-style decomposition for low-level fusion
        self.decomposition_nets = nn.ModuleDict({
            'text': self._make_decomposition_net(dim),
            'image': self._make_decomposition_net(dim),
            'audio': self._make_decomposition_net(dim)
        })
    
    def _make_decomposition_net(self, dim):
        """Create variational mode decomposition network."""
        return nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 3),  # 3 modes
            Reshape(-1, 3, dim)  # Using custom Reshape
        )
    
    def forward(self, multimodal_embeddings, multimodal_attractors):
        """
        Perform hierarchical multimodal fusion.
        
        Args:
            multimodal_embeddings: Dict of embeddings by modality
            multimodal_attractors: Dict of attractor fields by modality
            
        Returns:
            Fusion results with multiple levels
        """
        # Low-level decomposition
        decomposed = {}
        for modality, embeddings in multimodal_embeddings.items():
            if modality in self.decomposition_nets:
                # Decompose into modes
                modes = self.decomposition_nets[modality](embeddings.mean(dim=1))
                decomposed[modality] = modes
        
        # Mid-level alignment
        aligned_embeddings = self.alignment(multimodal_embeddings)
        
        # High-level fusion
        unified_field, fusion_info = self.fusion(multimodal_attractors)
        
        # Compute contrastive features
        contrastive_features = {}
        for mod, field in multimodal_attractors.items():
            contrastive_features[mod] = self.contrastive_proj(field)
        
        return {
            'unified_field': unified_field,
            'aligned_embeddings': aligned_embeddings,
            'decomposed_modes': decomposed,
            'contrastive_features': contrastive_features,
            'fusion_info': fusion_info,
            'temperature': self.temperature
        }
    
    def compute_contrastive_loss(self, features_dict):
        """Enhanced contrastive loss with hard negative mining."""
        modalities = list(features_dict.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i < j:
                    feat1 = F.normalize(features_dict[mod1], p=2, dim=-1)
                    feat2 = F.normalize(features_dict[mod2], p=2, dim=-1)
                    
                    # Similarity matrix
                    sim_matrix = torch.matmul(feat1, feat2.t()) / self.temperature
                    
                    batch_size = feat1.shape[0]
                    labels = torch.arange(batch_size, device=feat1.device)
                    
                    # Hard negative mining
                    # Find hardest negatives (highest similarity among negatives)
                    mask = torch.eye(batch_size, device=feat1.device).bool()
                    neg_sim = sim_matrix.masked_fill(mask, -float('inf'))
                    hard_negatives = neg_sim.max(dim=1)[0]
                    
                    # Contrastive loss with hard negatives
                    pos_sim = sim_matrix[torch.arange(batch_size), labels]
                    loss = -torch.log(
                        torch.exp(pos_sim) / 
                        (torch.exp(pos_sim) + torch.exp(hard_negatives))
                    ).mean()
                    
                    total_loss += loss
                    num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else total_loss
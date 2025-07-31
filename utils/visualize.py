"""
Enhanced visualization utilities for attractor fields and dynamics.
Provides interactive plots, 3D animations, and real-time monitoring.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')


class AttractorVisualizer:
    """Enhanced visualization class with interactive features."""
    
    def __init__(self, style='default', interactive=True):
        # Set style
        if style == 'dark':
            plt.style.use('dark_background')
            self.plotly_template = 'plotly_dark'
        elif style == 'paper':
            plt.style.use('seaborn-paper')
            self.plotly_template = 'plotly_white'
        else:
            plt.style.use('default')
            self.plotly_template = 'plotly'
        
        self.interactive = interactive
        
        # Enhanced color palettes
        self.colors = sns.color_palette('husl', 10)
        self.cmap = plt.cm.viridis
        
    def plot_attractor_field_2d(self, field, title="Attractor Field", 
                               method='pca', save_path=None, show_density=True):
        """
        Enhanced 2D projection with multiple visualization options.
        
        Args:
            field: Attractor field tensor (B, D) or (D,)
            title: Plot title
            method: Reduction method ('pca', 'tsne', 'umap')
            save_path: Path to save figure
            show_density: Show density estimation
        """
        # Ensure 2D tensor
        if field.dim() == 1:
            field = field.unsqueeze(0)
        
        # Convert to numpy
        field_np = field.detach().cpu().numpy()
        
        # Reduce dimensions
        field_2d, axis_labels = self._reduce_to_2d(field_np, method)
        
        if self.interactive:
            # Interactive plotly visualization
            fig = self._create_interactive_2d(field_2d, title, axis_labels, show_density)
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            
            fig.show()
            return fig
        else:
            # Static matplotlib visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot
            scatter = ax.scatter(
                field_2d[:, 0], field_2d[:, 1],
                c=np.arange(len(field_2d)),
                cmap='viridis', s=100, alpha=0.7,
                edgecolors='white', linewidth=0.5
            )
            
            # Density contours
            if show_density and field_2d.shape[0] > 5:
                try:
                    sns.kdeplot(
                        x=field_2d[:, 0], y=field_2d[:, 1],
                        ax=ax, levels=5, alpha=0.3, 
                        cmap='Blues', fill=True
                    )
                except:
                    pass
            
            # Annotations
            ax.set_xlabel(axis_labels[0], fontsize=12)
            ax.set_ylabel(axis_labels[1], fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Point Index')
            cbar.ax.tick_params(labelsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Stats text
            stats_text = self._compute_field_stats(field_np)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
    
    def plot_attractor_field_3d(self, field, title="3D Attractor Field", 
                               method='pca', animate=False, save_path=None):
        """
        Enhanced 3D visualization with animation support.
        
        Args:
            field: Attractor field tensor
            title: Plot title
            method: Reduction method
            animate: Create rotating animation
            save_path: Save path
        """
        # Prepare data
        if field.dim() == 1:
            field = field.unsqueeze(0)
        
        field_np = field.detach().cpu().numpy()
        
        # Reduce to 3D
        field_3d, axis_labels = self._reduce_to_3d(field_np, method)
        
        if self.interactive:
            # Interactive plotly 3D visualization
            fig = self._create_interactive_3d(field_3d, title, axis_labels, animate)
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            
            fig.show()
            return fig
        else:
            # Static matplotlib 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create scatter plot
            scatter = ax.scatter(
                field_3d[:, 0], field_3d[:, 1], 
                field_3d[:, 2] if field_3d.shape[1] > 2 else np.zeros(len(field_3d)),
                c=np.arange(len(field_3d)), 
                cmap='viridis', s=100, alpha=0.8,
                edgecolors='white', linewidth=0.5
            )
            
            # Add connections for nearest neighbors
            if field_3d.shape[0] > 1 and field_3d.shape[0] < 50:
                self._add_connections_3d(ax, field_3d)
            
            # Labels
            ax.set_xlabel(axis_labels[0], fontsize=12)
            ax.set_ylabel(axis_labels[1], fontsize=12)
            ax.set_zlabel(axis_labels[2] if len(axis_labels) > 2 else 'Dim 3', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Colorbar
            plt.colorbar(scatter, ax=ax, label='Point Index', pad=0.1)
            
            # Adjust view
            ax.view_init(elev=20, azim=45)
            
            if animate:
                # Create rotation animation
                def rotate(frame):
                    ax.view_init(elev=20, azim=frame)
                    return ax,
                
                anim = FuncAnimation(fig, rotate, frames=360, interval=50, blit=False)
                
                if save_path:
                    anim.save(save_path.replace('.png', '.gif'), writer='pillow', fps=20)
            else:
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
    
    def plot_dynamics_evolution(self, position_history, energy_history=None, 
                              velocity_history=None, save_path=None):
        """
        Enhanced dynamics visualization with multiple metrics.
        
        Args:
            position_history: List of position tensors
            energy_history: List of energy values
            velocity_history: List of velocity tensors
            save_path: Save path
        """
        if self.interactive:
            # Interactive plotly dashboard
            fig = self._create_dynamics_dashboard(
                position_history, energy_history, velocity_history
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            
            fig.show()
            return fig
        else:
            # Static matplotlib grid
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Position trajectories
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_trajectories(ax1, position_history)
            
            # 2. Energy evolution
            ax2 = fig.add_subplot(gs[0, 2])
            if energy_history:
                self._plot_energy(ax2, energy_history)
            
            # 3. Phase space
            ax3 = fig.add_subplot(gs[1, 0])
            if velocity_history:
                self._plot_phase_space(ax3, position_history, velocity_history)
            
            # 4. Distance from origin
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_distance_evolution(ax4, position_history)
            
            # 5. Convergence metric
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_convergence(ax5, position_history)
            
            # 6. Velocity distribution
            ax6 = fig.add_subplot(gs[2, 0])
            if velocity_history:
                self._plot_velocity_distribution(ax6, velocity_history)
            
            # 7. Attractor basin visualization
            ax7 = fig.add_subplot(gs[2, 1])
            self._plot_attractor_basins(ax7, position_history)
            
            # 8. Stability analysis
            ax8 = fig.add_subplot(gs[2, 2])
            self._plot_stability_analysis(ax8, position_history, energy_history)
            
            plt.suptitle('Attractor Dynamics Analysis', fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
    
    def create_potential_field_plot(self, attractor_positions, attractor_masses, 
                                  grid_size=100, xlim=(-5, 5), ylim=(-5, 5),
                                  show_streamlines=True, save_path=None):
        """
        Enhanced gravitational potential field visualization.
        
        Args:
            attractor_positions: Positions (N, 2)
            attractor_masses: Masses (N,)
            grid_size: Resolution
            xlim, ylim: Plot limits
            show_streamlines: Show force field streamlines
            save_path: Save path
        """
        # Create grid
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Compute potential and forces
        potential, force_x, force_y = self._compute_potential_and_forces(
            X, Y, attractor_positions, attractor_masses
        )
        
        if self.interactive:
            # Interactive plotly visualization
            fig = go.Figure()
            
            # Potential field as heatmap
            fig.add_trace(go.Heatmap(
                x=x, y=y, z=potential,
                colorscale='RdBu_r',
                name='Potential',
                colorbar=dict(title='Potential')
            ))
            
            # Attractors
            pos_np = attractor_positions.cpu().numpy()
            mass_np = attractor_masses.cpu().numpy()
            
            fig.add_trace(go.Scatter(
                x=pos_np[:, 0], y=pos_np[:, 1],
                mode='markers',
                marker=dict(
                    size=mass_np * 100,
                    color='black',
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                name='Attractors'
            ))
            
            # Streamlines
            if show_streamlines:
                # Sample streamline starting points
                stream_x, stream_y = np.meshgrid(
                    np.linspace(xlim[0], xlim[1], 15),
                    np.linspace(ylim[0], ylim[1], 15)
                )
                
                for i in range(stream_x.shape[0]):
                    for j in range(stream_x.shape[1]):
                        streamline = self._compute_streamline(
                            stream_x[i, j], stream_y[i, j],
                            attractor_positions, attractor_masses
                        )
                        
                        if len(streamline) > 1:
                            fig.add_trace(go.Scatter(
                                x=streamline[:, 0], y=streamline[:, 1],
                                mode='lines',
                                line=dict(color='gray', width=1),
                                showlegend=False,
                                opacity=0.5
                            ))
            
            fig.update_layout(
                title='Gravitational Potential Field',
                xaxis_title='X',
                yaxis_title='Y',
                template=self.plotly_template,
                height=600,
                width=700
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            
            fig.show()
            return fig
        else:
            # Static matplotlib visualization
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Potential field
            contourf = ax.contourf(X, Y, potential, levels=50, cmap='RdBu_r', alpha=0.8)
            contour = ax.contour(X, Y, potential, levels=20, colors='black', 
                               alpha=0.4, linewidths=0.5)
            
            # Streamlines
            if show_streamlines:
                # Normalize forces for visualization
                magnitude = np.sqrt(force_x**2 + force_y**2)
                magnitude[magnitude == 0] = 1
                
                stream = ax.streamplot(
                    X, Y, -force_x/magnitude, -force_y/magnitude,
                    color=magnitude, cmap='viridis', 
                    density=1.5, linewidth=1.5, arrowsize=1.5
                )
            
            # Plot attractors
            pos_np = attractor_positions.cpu().numpy()
            mass_np = attractor_masses.cpu().numpy()
            
            scatter = ax.scatter(
                pos_np[:, 0], pos_np[:, 1],
                s=mass_np * 1000, c='black',
                marker='*', edgecolors='white', linewidth=2,
                zorder=5, label='Attractors'
            )
            
            # Colorbar
            cbar = plt.colorbar(contourf, ax=ax, label='Gravitational Potential')
            
            # Labels
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_title('Gravitational Potential Field', fontsize=14, fontweight='bold')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
    
    def animate_dynamics(self, position_history, save_path=None, fps=30, 
                        show_trails=True, trail_length=10):
        """
        Enhanced animation with trails and multiple views.
        
        Args:
            position_history: List of position tensors
            save_path: Save path
            fps: Frames per second
            show_trails: Show particle trails
            trail_length: Length of trails
        """
        # Prepare data
        positions_2d = self._prepare_positions_for_animation(position_history)
        
        if self.interactive:
            # Interactive plotly animation
            fig = self._create_interactive_animation(
                positions_2d, show_trails, trail_length
            )
            
            if save_path:
                fig.write_html(save_path.replace('.gif', '.html'))
            
            fig.show()
            return fig
        else:
            # Matplotlib animation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Setup axes
            all_positions = positions_2d.reshape(-1, 2)
            margin = 0.1 * (all_positions.max() - all_positions.min())
            
            xlim = (all_positions[:, 0].min() - margin, 
                   all_positions[:, 0].max() + margin)
            ylim = (all_positions[:, 1].min() - margin, 
                   all_positions[:, 1].max() + margin)
            
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_xlabel('Dimension 1')
            ax1.set_ylabel('Dimension 2')
            ax1.grid(True, alpha=0.3)
            
            # Phase space plot
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Velocity')
            ax2.set_title('Phase Space')
            ax2.grid(True, alpha=0.3)
            
            # Initialize plots
            num_particles = positions_2d.shape[1]
            colors = plt.cm.viridis(np.linspace(0, 1, num_particles))
            
            scatters = []
            trails = []
            phase_lines = []
            
            for i in range(num_particles):
                scatter = ax1.scatter([], [], c=[colors[i]], s=100, 
                                    edgecolors='white', linewidth=1)
                scatters.append(scatter)
                
                if show_trails:
                    trail, = ax1.plot([], [], c=colors[i], alpha=0.3, linewidth=2)
                    trails.append(trail)
                
                phase_line, = ax2.plot([], [], c=colors[i], alpha=0.7)
                phase_lines.append(phase_line)
            
            title = ax1.text(0.5, 1.05, '', transform=ax1.transAxes, 
                            ha='center', fontsize=14)
            
            # Animation function
            def animate(frame):
                # Update positions
                current_pos = positions_2d[frame]
                
                for i in range(num_particles):
                    scatters[i].set_offsets([current_pos[i]])
                    
                    if show_trails:
                        # Update trails
                        start = max(0, frame - trail_length)
                        trail_data = positions_2d[start:frame+1, i]
                        trails[i].set_data(trail_data[:, 0], trail_data[:, 1])
                    
                    # Update phase space
                    if frame > 0:
                        positions = positions_2d[:frame+1, i, 0]
                        velocities = np.diff(positions, prepend=positions[0])
                        phase_lines[i].set_data(positions, velocities)
                
                # Update title
                title.set_text(f'Step {frame}/{len(positions_2d)-1}')
                
                return scatters + trails + phase_lines + [title]
            
            # Create animation
            anim = FuncAnimation(
                fig, animate, frames=len(positions_2d),
                interval=1000/fps, blit=True, repeat=True
            )
            
            if save_path:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
                print(f"Animation saved to {save_path}")
            
            return anim
    
    def plot_multimodal_alignment(self, embeddings_dict, method='cosine',
                                 save_path=None):
        """
        Visualize cross-modal alignment quality.
        
        Args:
            embeddings_dict: Dict of embeddings by modality
            method: Similarity method ('cosine', 'euclidean')
            save_path: Save path
        """
        modalities = list(embeddings_dict.keys())
        n_modalities = len(modalities)
        
        if n_modalities < 2:
            print("Need at least 2 modalities for alignment visualization")
            return
        
        # Compute similarity matrices
        similarities = {}
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i < j:
                    sim = self._compute_similarity_matrix(
                        embeddings_dict[mod1], 
                        embeddings_dict[mod2],
                        method
                    )
                    similarities[(mod1, mod2)] = sim
        
        if self.interactive:
            # Interactive plotly heatmaps
            fig = make_subplots(
                rows=1, cols=len(similarities),
                subplot_titles=[f"{m1} vs {m2}" for m1, m2 in similarities.keys()]
            )
            
            for idx, ((mod1, mod2), sim) in enumerate(similarities.items()):
                fig.add_trace(
                    go.Heatmap(
                        z=sim.cpu().numpy(),
                        colorscale='RdBu_r',
                        zmid=0,
                        showscale=idx == 0
                    ),
                    row=1, col=idx+1
                )
            
            fig.update_layout(
                title='Cross-Modal Alignment Matrices',
                height=400,
                width=400 * len(similarities)
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            
            fig.show()
            return fig
        else:
            # Static matplotlib
            fig, axes = plt.subplots(1, len(similarities), 
                                   figsize=(6*len(similarities), 5))
            
            if len(similarities) == 1:
                axes = [axes]
            
            for idx, ((mod1, mod2), sim) in enumerate(similarities.items()):
                ax = axes[idx]
                
                im = ax.imshow(sim.cpu().numpy(), cmap='RdBu_r', 
                             vmin=-1, vmax=1, aspect='auto')
                
                ax.set_title(f'{mod1} vs {mod2}')
                ax.set_xlabel(f'{mod2} samples')
                ax.set_ylabel(f'{mod1} samples')
                
                # Colorbar
                plt.colorbar(im, ax=ax)
            
            plt.suptitle('Cross-Modal Alignment Analysis', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
    
    # Helper methods
    
    def _reduce_to_2d(self, data, method):
        """Reduce high-dimensional data to 2D."""
        if data.shape[1] <= 2:
            return data[:, :2], ['Dim 1', 'Dim 2']
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(data)
            var_explained = reducer.explained_variance_ratio_
            labels = [f'PC1 ({var_explained[0]:.1%})', 
                     f'PC2 ({var_explained[1]:.1%})']
        
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
            reduced = reducer.fit_transform(data)
            labels = ['t-SNE 1', 't-SNE 2']
        
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(data)
            labels = ['UMAP 1', 'UMAP 2']
        
        else:
            reduced = data[:, :2]
            labels = ['Dim 1', 'Dim 2']
        
        return reduced, labels
    
    def _reduce_to_3d(self, data, method):
        """Reduce high-dimensional data to 3D."""
        if data.shape[1] <= 3:
            if data.shape[1] == 3:
                return data, ['Dim 1', 'Dim 2', 'Dim 3']
            else:
                # Pad with zeros
                padded = np.zeros((data.shape[0], 3))
                padded[:, :data.shape[1]] = data
                return padded, ['Dim 1', 'Dim 2', 'Dim 3']
        
        if method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
            reduced = reducer.fit_transform(data)
            var_explained = reducer.explained_variance_ratio_
            labels = [f'PC{i+1} ({var_explained[i]:.1%})' for i in range(3)]
        
        elif method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(data)-1))
            reduced = reducer.fit_transform(data)
            labels = ['t-SNE 1', 't-SNE 2', 't-SNE 3']
        
        elif method == 'umap':
            reducer = umap.UMAP(n_components=3, random_state=42)
            reduced = reducer.fit_transform(data)
            labels = ['UMAP 1', 'UMAP 2', 'UMAP 3']
        
        else:
            reduced = data[:, :3]
            labels = ['Dim 1', 'Dim 2', 'Dim 3']
        
        return reduced, labels
    
    def _compute_field_stats(self, field):
        """Compute statistics about the field."""
        stats = {
            'mean': np.mean(field),
            'std': np.std(field),
            'min': np.min(field),
            'max': np.max(field),
            'sparsity': np.mean(np.abs(field) < 0.01)
        }
        
        text = "Field Statistics:\n"
        text += f"Mean: {stats['mean']:.3f}\n"
        text += f"Std: {stats['std']:.3f}\n"
        text += f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]\n"
        text += f"Sparsity: {stats['sparsity']:.1%}"
        
        return text
    
    def _create_interactive_2d(self, data, title, labels, show_density):
        """Create interactive 2D plotly visualization."""
        fig = go.Figure()
        
        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=np.arange(len(data)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Point Index'),
                line=dict(color='white', width=1)
            ),
            text=[f'Point {i}' for i in range(len(data))],
            hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
        
        # Density contours
        if show_density and len(data) > 5:
            try:
                from scipy.stats import gaussian_kde
                
                kde = gaussian_kde(data.T)
                x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
                y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 50)
                X, Y = np.meshgrid(x_range, y_range)
                
                positions = np.vstack([X.ravel(), Y.ravel()])
                density = kde(positions).reshape(X.shape)
                
                fig.add_trace(go.Contour(
                    x=x_range, y=y_range, z=density,
                    showscale=False,
                    opacity=0.3,
                    colorscale='Blues',
                    contours=dict(coloring='heatmap')
                ))
            except:
                pass
        
        # Layout
        fig.update_layout(
            title=title,
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            template=self.plotly_template,
            hovermode='closest',
            height=600,
            width=700
        )
        
        return fig
    
    def _create_interactive_3d(self, data, title, labels, animate):
        """Create interactive 3D plotly visualization."""
        fig = go.Figure()
        
        # 3D scatter plot
        scatter = go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=np.arange(len(data)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Point Index')
            ),
            text=[f'Point {i}' for i in range(len(data))],
            hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        )
        
        fig.add_trace(scatter)
        
        # Add connections for small datasets
        if len(data) < 20:
            for i in range(len(data)):
                for j in range(i+1, len(data)):
                    dist = np.linalg.norm(data[i] - data[j])
                    if dist < np.percentile(
                        [np.linalg.norm(data[i] - data[j]) 
                         for i in range(len(data)) 
                         for j in range(i+1, len(data))], 20):
                        
                        fig.add_trace(go.Scatter3d(
                            x=[data[i, 0], data[j, 0]],
                            y=[data[i, 1], data[j, 1]],
                            z=[data[i, 2], data[j, 2]],
                            mode='lines',
                            line=dict(color='gray', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Camera animation
        if animate:
            n_frames = 100
            
            # Define camera path
            camera_positions = []
            for i in range(n_frames):
                angle = 2 * np.pi * i / n_frames
                camera_positions.append(dict(
                    eye=dict(
                        x=2 * np.cos(angle),
                        y=2 * np.sin(angle),
                        z=1.5
                    )
                ))
            
            # Create frames
            frames = [go.Frame(
                data=[scatter],
                layout=go.Layout(scene_camera=camera_positions[i])
            ) for i in range(n_frames)]
            
            fig.frames = frames
            
            # Add play button
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    }]
                }]
            )
        
        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
                zaxis_title=labels[2] if len(labels) > 2 else 'Dim 3'
            ),
            template=self.plotly_template,
            height=700,
            width=800
        )
        
        return fig
    
    def _create_dynamics_dashboard(self, position_history, energy_history, velocity_history):
        """Create interactive dynamics dashboard."""
        # Convert history to arrays
        positions = torch.stack(position_history) if position_history else None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Trajectories', 'Energy Evolution', 'Phase Space',
                          'Distance from Origin', 'Convergence', 'Velocity Distribution'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. Trajectories
        if positions is not None:
            positions_2d = self._prepare_positions_for_animation(position_history)
            
            for i in range(min(10, positions_2d.shape[1])):
                trajectory = positions_2d[:, i, :]
                fig.add_trace(
                    go.Scatter(
                        x=trajectory[:, 0], y=trajectory[:, 1],
                        mode='lines+markers',
                        name=f'Particle {i}',
                        marker=dict(size=4),
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # 2. Energy evolution
        if energy_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(energy_history))),
                    y=energy_history,
                    mode='lines',
                    name='System Energy',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        # 3. Phase space
        if positions is not None and velocity_history:
            velocities = torch.stack(velocity_history)
            
            # Plot for first particle
            pos_flat = positions[:, 0, 0].cpu().numpy()
            vel_flat = velocities[:, 0, 0].cpu().numpy()
            
            fig.add_trace(
                go.Scatter(
                    x=pos_flat, y=vel_flat,
                    mode='lines',
                    name='Phase Trajectory',
                    line=dict(color='green', width=2)
                ),
                row=1, col=3
            )
        
        # 4. Distance from origin
        if positions is not None:
            distances = torch.norm(positions.reshape(len(positions), -1, positions.shape[-1]), dim=-1).mean(dim=1)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(distances))),
                    y=distances.cpu().numpy(),
                    mode='lines',
                    name='Avg Distance',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        # 5. Convergence
        if positions is not None and len(positions) > 1:
            position_changes = []
            for i in range(1, len(positions)):
                change = torch.norm(positions[i] - positions[i-1]).item()
                position_changes.append(change)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(position_changes) + 1)),
                    y=position_changes,
                    mode='lines',
                    name='Position Change',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
            
            # Log scale y-axis for convergence
            fig.update_yaxes(type="log", row=2, col=2)
        
        # 6. Velocity distribution
        if velocity_history:
            velocities = torch.stack(velocity_history)
            vel_norms = torch.norm(velocities, dim=-1).flatten().cpu().numpy()
            
            fig.add_trace(
                go.Histogram(
                    x=vel_norms,
                    nbinsx=30,
                    name='Velocity Magnitude',
                    marker_color='orange'
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title='Attractor Dynamics Dashboard',
            height=800,
            width=1400,
            showlegend=False,
            template=self.plotly_template
        )
        
        return fig
    
    def _plot_trajectories(self, ax, position_history):
        """Plot particle trajectories."""
        positions_2d = self._prepare_positions_for_animation(position_history)
        
        for i in range(min(10, positions_2d.shape[1])):
            trajectory = positions_2d[:, i, :]
            
            # Plot trajectory
            line = ax.plot(trajectory[:, 0], trajectory[:, 1], 
                          alpha=0.7, linewidth=2, 
                          color=self.colors[i % len(self.colors)])[0]
            
            # Mark start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                      c='green', s=50, marker='o', zorder=5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                      c='red', s=50, marker='x', zorder=5)
        
        ax.set_title('Particle Trajectories')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
    
    def _plot_energy(self, ax, energy_history):
        """Plot energy evolution."""
        ax.plot(energy_history, 'r-', linewidth=2)
        ax.set_title('System Energy')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(energy_history) > 10:
            from scipy.optimize import curve_fit
            x = np.arange(len(energy_history))
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            try:
                popt, _ = curve_fit(exp_decay, x, energy_history, p0=[1, 0.01, min(energy_history)])
                y_fit = exp_decay(x, *popt)
                ax.plot(x, y_fit, 'r--', alpha=0.5, label=f'Fit: {popt[1]:.3f} decay rate')
                ax.legend()
            except:
                pass
    
    def _plot_phase_space(self, ax, position_history, velocity_history):
        """Plot phase space diagram."""
        positions = torch.stack(position_history)
        velocities = torch.stack(velocity_history)
        
        # Plot for first few particles
        for i in range(min(3, positions.shape[1])):
            pos = positions[:, i, 0].cpu().numpy()
            vel = velocities[:, i, 0].cpu().numpy()
            
            ax.plot(pos, vel, alpha=0.7, linewidth=2, 
                   color=self.colors[i % len(self.colors)],
                   label=f'Particle {i}')
        
        ax.set_title('Phase Space (Position vs Velocity)')
        ax.set_xlabel('Position (Dim 1)')
        ax.set_ylabel('Velocity (Dim 1)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_distance_evolution(self, ax, position_history):
        """Plot distance from origin over time."""
        positions = torch.stack(position_history)
        distances = torch.norm(positions, dim=-1)
        
        # Mean distance
        mean_dist = distances.mean(dim=1).cpu().numpy()
        std_dist = distances.std(dim=1).cpu().numpy()
        
        x = np.arange(len(mean_dist))
        ax.plot(x, mean_dist, 'b-', linewidth=2, label='Mean')
        ax.fill_between(x, mean_dist - std_dist, mean_dist + std_dist, 
                       alpha=0.3, color='blue', label='±1 STD')
        
        ax.set_title('Distance from Origin')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_convergence(self, ax, position_history):
        """Plot convergence metric."""
        if len(position_history) < 2:
            return
        
        position_changes = []
        for i in range(1, len(position_history)):
            change = torch.norm(position_history[i] - position_history[i-1]).item()
            position_changes.append(change)
        
        ax.semilogy(position_changes, 'g-', linewidth=2)
        ax.set_title('Convergence (Position Change)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Change (log scale)')
        ax.grid(True, alpha=0.3)
        
        # Add convergence threshold
        threshold = 1e-6
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  alpha=0.5, label=f'Threshold: {threshold}')
        ax.legend()
    
    def _plot_velocity_distribution(self, ax, velocity_history):
        """Plot velocity magnitude distribution."""
        velocities = torch.stack(velocity_history)
        vel_norms = torch.norm(velocities, dim=-1).flatten().cpu().numpy()
        
        ax.hist(vel_norms, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title('Velocity Magnitude Distribution')
        ax.set_xlabel('Velocity Magnitude')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_vel = np.mean(vel_norms)
        ax.axvline(x=mean_vel, color='red', linestyle='--', 
                  label=f'Mean: {mean_vel:.3f}')
        ax.legend()
    
    def _plot_attractor_basins(self, ax, position_history):
        """Visualize attractor basins."""
        final_positions = position_history[-1]
        
        if final_positions.dim() > 2:
            final_positions = final_positions.reshape(-1, final_positions.shape[-1])
        
        # Reduce to 2D for visualization
        pos_2d, _ = self._reduce_to_2d(final_positions.cpu().numpy(), 'pca')
        
        # Cluster analysis
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(pos_2d)
        
        # Plot clusters
        unique_labels = set(clustering.labels_)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise
                color = 'gray'
                marker = 'x'
            else:
                marker = 'o'
            
            mask = clustering.labels_ == label
            ax.scatter(pos_2d[mask, 0], pos_2d[mask, 1], 
                      c=[color], s=50, marker=marker,
                      edgecolors='black', linewidth=0.5,
                      label=f'Basin {label}' if label != -1 else 'Noise')
        
        ax.set_title('Attractor Basins (Final State)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_stability_analysis(self, ax, position_history, energy_history):
        """Plot stability metrics."""
        # Compute various stability metrics
        metrics = {
            'iterations': len(position_history),
            'converged': False,
            'energy_ratio': 1.0,
            'final_spread': 0.0
        }
        
        if len(position_history) > 1:
            final_change = torch.norm(position_history[-1] - position_history[-2]).item()
            metrics['converged'] = final_change < 1e-6
        
        if energy_history and len(energy_history) > 1:
            metrics['energy_ratio'] = energy_history[-1] / energy_history[0]
        
        if position_history:
            final_pos = position_history[-1]
            if final_pos.dim() > 1:
                final_pos = final_pos.reshape(-1, final_pos.shape[-1])
                metrics['final_spread'] = torch.std(final_pos).item()
        
        # Create bar plot
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        # Normalize boolean to 0/1
        values = [float(v) if isinstance(v, bool) else v for v in values]
        
        bars = ax.bar(labels, values, color=['green' if v < 0.5 else 'orange' for v in values])
        
        ax.set_title('Stability Analysis')
        ax.set_ylabel('Value')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom')
    
    def _compute_potential_and_forces(self, X, Y, positions, masses):
        """Compute gravitational potential and force field."""
        potential = np.zeros_like(X)
        force_x = np.zeros_like(X)
        force_y = np.zeros_like(Y)
        
        pos_np = positions.cpu().numpy()
        mass_np = masses.cpu().numpy()
        
        for i in range(len(positions)):
            # Distance from each grid point to attractor
            dx = X - pos_np[i, 0]
            dy = Y - pos_np[i, 1]
            dist = np.sqrt(dx**2 + dy**2 + 0.1)  # Avoid singularity
            
            # Potential V = -G * m / r
            potential += -mass_np[i] / dist
            
            # Force F = -grad(V) = G * m * r_hat / r^2
            force_x += mass_np[i] * dx / (dist**3)
            force_y += mass_np[i] * dy / (dist**3)
        
        return potential, force_x, force_y
    
    def _compute_streamline(self, x0, y0, positions, masses, max_steps=100, dt=0.1):
        """Compute a single streamline."""
        streamline = [(x0, y0)]
        x, y = x0, y0
        
        pos_np = positions.cpu().numpy()
        mass_np = masses.cpu().numpy()
        
        for _ in range(max_steps):
            # Compute force at current position
            fx, fy = 0, 0
            
            for i in range(len(positions)):
                dx = pos_np[i, 0] - x
                dy = pos_np[i, 1] - y
                dist = np.sqrt(dx**2 + dy**2 + 0.1)
                
                fx += mass_np[i] * dx / (dist**3)
                fy += mass_np[i] * dy / (dist**3)
            
            # Update position
            x += fx * dt
            y += fy * dt
            
            streamline.append((x, y))
            
            # Stop if converged to attractor
            if np.sqrt(fx**2 + fy**2) < 0.01:
                break
        
        return np.array(streamline)
    
    def _prepare_positions_for_animation(self, position_history):
        """Prepare position data for animation."""
        positions = []
        
        for pos in position_history:
            if pos.dim() > 2:
                pos = pos.reshape(-1, pos.shape[-1])
            
            # Convert to 2D
            pos_np = pos.cpu().numpy()
            if pos_np.shape[1] > 2:
                if not hasattr(self, '_anim_pca'):
                    self._anim_pca = PCA(n_components=2, random_state=42)
                    pos_2d = self._anim_pca.fit_transform(pos_np)
                else:
                    pos_2d = self._anim_pca.transform(pos_np)
            else:
                pos_2d = pos_np[:, :2]
            
            positions.append(pos_2d)
        
        return np.array(positions)
    
    def _create_interactive_animation(self, positions_2d, show_trails, trail_length):
        """Create interactive plotly animation."""
        num_frames = len(positions_2d)
        num_particles = positions_2d.shape[1]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each particle
        for i in range(num_particles):
            # Current position
            fig.add_trace(go.Scatter(
                x=[positions_2d[0, i, 0]],
                y=[positions_2d[0, i, 1]],
                mode='markers',
                marker=dict(size=10, color=i, colorscale='Viridis'),
                name=f'Particle {i}'
            ))
            
            # Trail
            if show_trails:
                fig.add_trace(go.Scatter(
                    x=[positions_2d[0, i, 0]],
                    y=[positions_2d[0, i, 1]],
                    mode='lines',
                    line=dict(color=f'rgba({i*25}, {100}, {255-i*25}, 0.3)', width=2),
                    showlegend=False
                ))
        
        # Create frames
        frames = []
        for frame_idx in range(num_frames):
            frame_data = []
            
            for i in range(num_particles):
                # Current position
                frame_data.append(go.Scatter(
                    x=[positions_2d[frame_idx, i, 0]],
                    y=[positions_2d[frame_idx, i, 1]]
                ))
                
                # Trail
                if show_trails:
                    start = max(0, frame_idx - trail_length)
                    trail_x = positions_2d[start:frame_idx+1, i, 0]
                    trail_y = positions_2d[start:frame_idx+1, i, 1]
                    
                    frame_data.append(go.Scatter(
                        x=trail_x,
                        y=trail_y
                    ))
            
            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
        
        fig.frames = frames
        
        # Animation settings
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Step: '},
                'steps': [
                    {
                        'args': [[str(k)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': str(k),
                        'method': 'animate'
                    }
                    for k in range(num_frames)
                ]
            }]
        )
        
        # Set axis properties
        all_x = positions_2d[:, :, 0].flatten()
        all_y = positions_2d[:, :, 1].flatten()
        
        fig.update_xaxes(range=[all_x.min() - 0.5, all_x.max() + 0.5])
        fig.update_yaxes(range=[all_y.min() - 0.5, all_y.max() + 0.5])
        
        fig.update_layout(
            title='Attractor Dynamics Animation',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            template=self.plotly_template,
            height=600,
            width=700
        )
        
        return fig
    
    def _add_connections_3d(self, ax, positions):
        """Add connections between nearby points in 3D."""
        from scipy.spatial.distance import pdist, squareform
        
        # Compute pairwise distances
        distances = squareform(pdist(positions))
        
        # Connect nearest neighbors
        threshold = np.percentile(distances[distances > 0], 20)
        
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if distances[i, j] < threshold:
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           [positions[i, 2], positions[j, 2]],
                           'gray', alpha=0.3, linewidth=0.5)
    
    def _compute_similarity_matrix(self, embeddings1, embeddings2, method='cosine'):
        """Compute similarity matrix between embeddings."""
        if method == 'cosine':
            # Normalize embeddings
            emb1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
            emb2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.matmul(emb1_norm, emb2_norm.t())
            
        elif method == 'euclidean':
            # Negative euclidean distance as similarity
            distances = torch.cdist(embeddings1, embeddings2, p=2)
            similarity = -distances
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity
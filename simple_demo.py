"""
Simple GDA demo that works without all the complex training infrastructure.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simple versions of core components for demo
class SimpleEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(100, 256),  # Assume text tokens are embedded to 100
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, text, modality='text'):
        # For demo, assume text is already embedded
        if isinstance(text, str):
            # Simple string to tensor conversion for demo
            text_tensor = torch.randn(1, 100)  # Mock embedding
        else:
            text_tensor = text
            
        embeddings = self.text_encoder(text_tensor)
        masses = torch.ones(embeddings.shape[0], embeddings.shape[1])
        
        return embeddings, masses

class SimpleAttractorCore(nn.Module):
    def __init__(self, latent_dim=128, num_attractors=10, gravity_strength=0.001):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attractors = num_attractors
        self.gravity_strength = gravity_strength
        
        # Attractor positions
        self.attractors = nn.Parameter(torch.randn(num_attractors, latent_dim) * 0.1)
        
    def forward(self, embeddings, num_iterations=50, **kwargs):
        batch_size, seq_len, dim = embeddings.shape
        
        # Simple gravitational dynamics
        positions = embeddings.clone()
        convergence_info = {'iterations': 0, 'converged': False}
        
        for i in range(num_iterations):
            # Compute forces from attractors
            forces = torch.zeros_like(positions)
            
            for j in range(self.num_attractors):
                # Distance to attractor
                diff = self.attractors[j].unsqueeze(0).unsqueeze(0) - positions
                dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
                
                # Gravitational force
                force = self.gravity_strength * diff / (dist ** 3)
                forces += force
            
            # Update positions
            old_positions = positions.clone()
            positions = positions + 0.01 * forces
            
            # Check convergence
            change = torch.norm(positions - old_positions)
            if change < 1e-6:
                convergence_info = {'iterations': i+1, 'converged': True, 'final_change': change.item()}
                break
        
        if not convergence_info['converged']:
            convergence_info = {'iterations': num_iterations, 'converged': False, 'final_change': change.item()}
        
        return positions, convergence_info

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # Back to original embedding size
        )
        
    def forward(self, attractor_field, target_modality='text'):
        return self.decoder(attractor_field)

class SimpleGDAModel(nn.Module):
    def __init__(self, latent_dim=128, num_attractors=10, gravity_strength=0.001):
        super().__init__()
        
        self.encoder = SimpleEncoder(latent_dim)
        self.attractor_core = SimpleAttractorCore(latent_dim, num_attractors, gravity_strength)
        self.decoder = SimpleDecoder(latent_dim)
        
    def forward(self, inputs, target_modality='auto', return_all=False):
        # Encode
        embeddings, masses = self.encoder(inputs)
        
        # Attractor dynamics
        attractor_field, convergence_info = self.attractor_core(embeddings)
        
        # Decode
        reconstructions = self.decoder(attractor_field)
        
        return {
            'attractor_field': attractor_field,
            'reconstructions': reconstructions,
            'convergence_info': convergence_info
        }

def demo_gda():
    """Run simple GDA demonstration."""
    print("=" * 50)
    print("GRAVITATIONAL DATA ATTRACTOR - Simple Demo")
    print("=" * 50)
    
    # Create model
    model = SimpleGDAModel(latent_dim=64, num_attractors=5, gravity_strength=0.01)
    model.eval()
    
    # Create synthetic input
    batch_size = 3
    seq_len = 10
    embedding_dim = 100
    
    synthetic_input = torch.randn(batch_size, seq_len, embedding_dim)
    
    print(f"Input shape: {synthetic_input.shape}")
    
    # Process through GDA
    with torch.no_grad():
        outputs = model(synthetic_input)
    
    print(f"Attractor field shape: {outputs['attractor_field'].shape}")
    print(f"Reconstruction shape: {outputs['reconstructions'].shape}")
    print(f"Converged: {outputs['convergence_info']['converged']}")
    print(f"Iterations: {outputs['convergence_info']['iterations']}")
    print(f"Final change: {outputs['convergence_info']['final_change']:.6f}")
    
    # Visualize attractor dynamics
    visualize_attractors(model, synthetic_input)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    return model, outputs

def visualize_attractors(model, input_data):
    """Visualize the attractor positions and dynamics."""
    try:
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            # Get attractor positions
            attractors = model.attractor_core.attractors.numpy()
            
            # Process input
            embeddings, _ = model.encoder(input_data)
            initial_positions = embeddings[0].numpy()  # First batch
            
            final_field, _ = model.attractor_core(embeddings)
            final_positions = final_field[0].numpy()  # First batch
            
            # Create 2D visualization (take first 2 dimensions)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Initial state
            ax1.scatter(attractors[:, 0], attractors[:, 1], c='red', s=100, alpha=0.7, label='Attractors')
            ax1.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', s=50, alpha=0.7, label='Initial Positions')
            ax1.set_title('Initial State')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Final state
            ax2.scatter(attractors[:, 0], attractors[:, 1], c='red', s=100, alpha=0.7, label='Attractors')
            ax2.scatter(final_positions[:, 0], final_positions[:, 1], c='green', s=50, alpha=0.7, label='Final Positions')
            ax2.set_title('After Gravitational Dynamics')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Draw arrows showing movement
            for i in range(min(len(initial_positions), len(final_positions))):
                ax2.arrow(initial_positions[i, 0], initial_positions[i, 1],
                         final_positions[i, 0] - initial_positions[i, 0],
                         final_positions[i, 1] - initial_positions[i, 1],
                         head_width=0.02, head_length=0.03, fc='gray', ec='gray', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('attractor_dynamics.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("\nVisualization saved as 'attractor_dynamics.png'")
            
    except ImportError:
        print("\nMatplotlib not available for visualization")
    except Exception as e:
        print(f"\nVisualization error: {e}")

def interactive_demo():
    """Interactive demo with user input."""
    print("\n" + "="*30)
    print("INTERACTIVE GDA DEMO")
    print("="*30)
    
    model = SimpleGDAModel(latent_dim=32, num_attractors=3, gravity_strength=0.05)
    
    while True:
        print("\nOptions:")
        print("1. Run with random data")
        print("2. Adjust gravity strength")
        print("3. Change number of attractors")
        print("4. Show model info")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            print("\nRunning GDA with random data...")
            input_data = torch.randn(2, 8, 100)
            
            with torch.no_grad():
                outputs = model(input_data)
            
            conv_info = outputs['convergence_info']
            print(f"✓ Converged: {conv_info['converged']}")
            print(f"✓ Iterations: {conv_info['iterations']}")
            print(f"✓ Final change: {conv_info['final_change']:.6f}")
            
            # Show compression ratio
            input_size = input_data.numel()
            compressed_size = outputs['attractor_field'].numel()
            ratio = input_size / compressed_size
            print(f"✓ Compression ratio: {ratio:.2f}x")
            
        elif choice == '2':
            try:
                gravity = float(input("Enter gravity strength (0.001-0.1): "))
                model.attractor_core.gravity_strength = gravity
                print(f"✓ Gravity strength set to {gravity}")
            except ValueError:
                print("✗ Invalid number")
                
        elif choice == '3':
            try:
                num_attractors = int(input("Enter number of attractors (1-20): "))
                if 1 <= num_attractors <= 20:
                    # Recreate attractor core with new number
                    model.attractor_core = SimpleAttractorCore(
                        model.attractor_core.latent_dim, 
                        num_attractors, 
                        model.attractor_core.gravity_strength
                    )
                    print(f"✓ Number of attractors set to {num_attractors}")
                else:
                    print("✗ Number must be between 1 and 20")
            except ValueError:
                print("✗ Invalid number")
                
        elif choice == '4':
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel Information:")
            print(f"  Latent dimension: {model.attractor_core.latent_dim}")
            print(f"  Number of attractors: {model.attractor_core.num_attractors}")
            print(f"  Gravity strength: {model.attractor_core.gravity_strength}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Model size: {total_params * 4 / 1024:.2f} KB (float32)")
        else:
            print("✗ Invalid option")

def benchmark_performance():
    """Benchmark GDA performance."""
    print("\n" + "="*30)
    print("PERFORMANCE BENCHMARK")
    print("="*30)
    
    import time
    
    # Test different model sizes
    configs = [
        {'latent_dim': 32, 'num_attractors': 5, 'name': 'Small'},
        {'latent_dim': 64, 'num_attractors': 10, 'name': 'Medium'},
        {'latent_dim': 128, 'num_attractors': 20, 'name': 'Large'},
    ]
    
    batch_sizes = [1, 4, 16]
    seq_lengths = [10, 50, 100]
    
    for config in configs:
        print(f"\n{config['name']} Model (dim={config['latent_dim']}, attractors={config['num_attractors']}):")
        
        model = SimpleGDAModel(**{k: v for k, v in config.items() if k != 'name'})
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
        
        model.eval()
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                input_data = torch.randn(batch_size, seq_len, 100)
                
                # Warmup
                with torch.no_grad():
                    _ = model(input_data)
                
                # Benchmark
                start_time = time.time()
                num_runs = 10
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        outputs = model(input_data)
                
                avg_time = (time.time() - start_time) / num_runs * 1000  # ms
                
                print(f"    Batch {batch_size}, Seq {seq_len}: {avg_time:.2f}ms")
                
                # Check sub-millisecond inference for small inputs
                if batch_size == 1 and seq_len == 10 and avg_time < 1.0:
                    print(f"    ✓ Sub-millisecond inference achieved!")

if __name__ == "__main__":
    try:
        print("Starting GDA Simple Demo...")
        
        # Run basic demo
        model, outputs = demo_gda()
        
        # Interactive demo
        interactive_demo()
        
        # Performance benchmark
        benchmark_performance()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("This is a simplified demo - some features may not work without full dependencies")
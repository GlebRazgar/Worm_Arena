"""
Visualization for Simple GCN Test Model
Compares GCN performance against baseline and generates plots.
"""

import sys
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from data.preprocessing import load_or_create_dataset
from models.test_gcn import SimpleTestGCN, evaluate, compute_naive_baseline


def setup_device():
    """Setup compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_baseline_comparison(gcn_metrics, baseline_metrics, output_dir="plots"):
    """
    Create bar chart comparing GCN vs naive baseline.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics = ['MSE', 'MAE', 'RMSE', 'R²']
    gcn_values = [gcn_metrics['mse'], gcn_metrics['mae'], 
                  gcn_metrics['rmse'], gcn_metrics['r2']]
    baseline_values = [baseline_metrics['mse'], baseline_metrics['mae'],
                       baseline_metrics['rmse'], baseline_metrics['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, gcn_values, width, label='Simple GCN', color='#4ECDC4')
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Naive Baseline', color='#FF6B6B')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Simple GCN vs Naive Baseline (X̂(t+1) = X(t))', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars1, gcn_values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar, val in zip(bars2, baseline_values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "simple_gcn_vs_baseline.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'simple_gcn_vs_baseline.png'}")


def plot_loss_curves(history, output_dir="plots"):
    """
    Plot training and validation loss curves.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² curve
    ax2.plot(epochs, history['val_r2'], 'g-', linewidth=2)
    ax2.axhline(y=0.87, color='r', linestyle='--', label='Naive Baseline (≈0.87)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Validation R² Over Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "simple_gcn_loss_curves.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'simple_gcn_loss_curves.png'}")


def plot_predictions(model, loader, device, output_dir="plots", num_samples=6):
    """
    Plot predicted vs actual traces for sample neurons.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Get some samples
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_samples:
                break
            data = data.to(device)
            pred = model(data)
            
            all_preds.append(pred.cpu())
            all_targets.append(data.y.cpu())
            all_masks.append(data.mask.cpu())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(all_preds))):
        ax = axes[i]
        pred = all_preds[i].view(-1).numpy()
        target = all_targets[i].view(-1).numpy()
        mask = all_masks[i].view(-1).numpy()
        
        # Plot only valid neurons
        valid_idx = np.where(mask == 1)[0][:50]  # First 50 valid neurons
        
        ax.scatter(target[valid_idx], pred[valid_idx], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(target[valid_idx].min(), pred[valid_idx].min())
        max_val = max(target[valid_idx].max(), pred[valid_idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual X(t+1)')
        ax.set_ylabel('Predicted X̂(t+1)')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('GCN Predictions vs Actual (per sample)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "simple_gcn_predictions.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'simple_gcn_predictions.png'}")


def main():
    """
    Main visualization function.
    """
    print("=" * 70)
    print("SIMPLE GCN TEST MODEL - VISUALIZATION")
    print("=" * 70)
    
    device = setup_device()
    print(f"Using device: {device}")
    
    # Check for saved results
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    results_path = checkpoint_dir / "simple_gcn_results.json"
    model_path = checkpoint_dir / "simple_gcn_best.pt"
    
    if not results_path.exists():
        print("\nNo training results found. Please run training first:")
        print("  python src/train_test_gcn.py --epochs 100")
        return
    
    # Load results
    print("\n1. Loading training results...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    gcn_metrics = results['gcn_metrics']
    baseline_metrics = results['baseline_metrics']
    history = results['training_history']
    
    print(f"   GCN R²: {gcn_metrics['r2']:.4f}")
    print(f"   Baseline R²: {baseline_metrics['r2']:.4f}")
    
    # Create visualizations
    output_dir = Path(__file__).parent / "plots"
    
    print("\n2. Creating comparison plot...")
    plot_baseline_comparison(gcn_metrics, baseline_metrics, output_dir)
    
    print("\n3. Creating loss curves...")
    plot_loss_curves(history, output_dir)
    
    # Load model for prediction visualization
    if model_path.exists():
        print("\n4. Creating prediction plots...")
        
        # Load data
        connectome = load_connectome(verbose=False)
        df = load_functional_data(connectome=connectome, verbose=False)
        dataset = load_or_create_dataset(df, connectome, 
                                          max_worms=results['config'].get('max_worms'),
                                          use_cache=True, verbose=False)
        _, _, test_ds = dataset.split_by_worm()
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        
        # Load model
        model = SimpleTestGCN(hidden_dim=results['config']['hidden_dim'])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        plot_predictions(model, test_loader, device, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Visualization complete!")
    print(f"  Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


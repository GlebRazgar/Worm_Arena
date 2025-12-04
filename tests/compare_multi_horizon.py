"""
Multi-Horizon Evaluation Comparison
Compares baseline vs GCN model at short-term (t+1), mid-term (t+10), and long-term (t+20) horizons.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch_geometric.loader import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from models.test_gcn import SimpleTestGCN, evaluate_multi_horizon, compute_naive_baseline_multi_horizon


def setup_device():
    """Setup compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_multi_horizon_comparison(baseline_results, gcn_results, output_dir="plots"):
    """
    Create comparison plot across multiple horizons.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    horizons = sorted(baseline_results.keys())
    baseline_r2 = [baseline_results[h]['r2'] for h in horizons]
    gcn_r2 = [gcn_results[h]['r2'] for h in horizons]
    baseline_mse = [baseline_results[h]['mse'] for h in horizons]
    gcn_mse = [gcn_results[h]['mse'] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    ax1.plot(horizons, baseline_r2, 'o-', label='Naive Baseline', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.plot(horizons, gcn_r2, 's-', label='Simple GCN', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)
    
    # Add value labels
    for h, b_r2, g_r2 in zip(horizons, baseline_r2, gcn_r2):
        ax1.annotate(f'{b_r2:.3f}', (h, b_r2), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='#FF6B6B')
        ax1.annotate(f'{g_r2:.3f}', (h, g_r2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='#4ECDC4')
    
    # MSE comparison
    ax2.plot(horizons, baseline_mse, 'o-', label='Naive Baseline', linewidth=2, markersize=8, color='#FF6B6B')
    ax2.plot(horizons, gcn_mse, 's-', label='Simple GCN', linewidth=2, markersize=8, color='#4ECDC4')
    ax2.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('MSE vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(horizons)
    ax2.set_yscale('log')  # Log scale for MSE
    
    plt.tight_layout()
    plt.savefig(output_path / "multi_horizon_comparison.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'multi_horizon_comparison.png'}")


def print_comparison_table(baseline_results, gcn_results):
    """
    Print formatted comparison table.
    """
    horizons = sorted(baseline_results.keys())
    
    print("\n" + "=" * 80)
    print("MULTI-HORIZON EVALUATION COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Horizon':<12} {'Metric':<12} {'Baseline':<15} {'GCN':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for horizon in horizons:
        h_name = f"t+{horizon}"
        b = baseline_results[horizon]
        g = gcn_results[horizon]
        
        # R²
        r2_improvement = ((g['r2'] - b['r2']) / abs(b['r2']) * 100) if not np.isnan(b['r2']) else 0
        print(f"{h_name:<12} {'R²':<12} {b['r2']:<15.4f} {g['r2']:<15.4f} {r2_improvement:+.1f}%")
        
        # MSE
        mse_improvement = ((b['mse'] - g['mse']) / b['mse'] * 100) if b['mse'] > 0 else 0
        print(f"{'':<12} {'MSE':<12} {b['mse']:<15.4f} {g['mse']:<15.4f} {mse_improvement:+.1f}%")
        
        # MAE
        mae_improvement = ((b['mae'] - g['mae']) / b['mae'] * 100) if b['mae'] > 0 else 0
        print(f"{'':<12} {'MAE':<12} {b['mae']:<15.4f} {g['mae']:<15.4f} {mae_improvement:+.1f}%")
        
        print("-" * 80)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Short-term (t+1):  Baseline R²={baseline_results[1]['r2']:.4f}, GCN R²={gcn_results[1]['r2']:.4f}")
    print(f"  Mid-term (t+10):   Baseline R²={baseline_results[10]['r2']:.4f}, GCN R²={gcn_results[10]['r2']:.4f}")
    print(f"  Long-term (t+20):  Baseline R²={baseline_results[20]['r2']:.4f}, GCN R²={gcn_results[20]['r2']:.4f}")
    
    # Check if GCN beats baseline at longer horizons
    improvements = []
    for h in horizons:
        if h > 1:  # Skip t+1 (baseline is too good)
            b_r2 = baseline_results[h]['r2']
            g_r2 = gcn_results[h]['r2']
            if not np.isnan(b_r2) and not np.isnan(g_r2):
                improvements.append(g_r2 > b_r2)
    
    if improvements:
        wins = sum(improvements)
        print(f"\n  GCN beats baseline at {wins}/{len(improvements)} longer horizons (t+10, t+20)")


def main():
    """
    Main comparison function.
    """
    print("=" * 80)
    print("MULTI-HORIZON EVALUATION: Baseline vs GCN")
    print("=" * 80)
    
    device = setup_device()
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    connectome = load_connectome(verbose=False)
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Connectome: {connectome.num_nodes} neurons, {connectome.num_edges} edges")
    print(f"   Functional: {len(df)} samples, {df['worm'].nunique()} worms")
    
    # Load trained model
    print("\n2. Loading trained GCN model...")
    weights_dir = Path(__file__).parent.parent / "models" / "weights"
    
    # Try to find GCN model weights (new naming scheme)
    weights_path = weights_dir / "gcn_model_weights.pt"
    
    # Fallback to old naming if new doesn't exist
    if not weights_path.exists():
        old_weights_path = weights_dir / "model_weights.pt"
        if old_weights_path.exists():
            weights_path = old_weights_path
            print(f"   ⚠ Using old weight filename, consider retraining with new naming")
        else:
            print(f"   ERROR: Model weights not found at {weights_path} or {old_weights_path}")
            print("   Please train the model first:")
            print("   python src/train_test_gcn.py --epochs 20 --batch-size 128 --hidden-dim 32")
            return
    
    # Try to infer hidden_dim from weights
    # SimpleTestGCN has: gcn.weight [hidden_dim, 1] and predictor.weight [1, hidden_dim]
    weights = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Infer hidden_dim from weight shapes
    if 'gcn.weight' in weights:
        hidden_dim = weights['gcn.weight'].shape[0]
    elif 'predictor.weight' in weights:
        hidden_dim = weights['predictor.weight'].shape[1]
    else:
        # Default fallback
        hidden_dim = 32
        print(f"   ⚠ Could not infer hidden_dim, using default: {hidden_dim}")
    
    model = SimpleTestGCN(hidden_dim=hidden_dim)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    print(f"   ✓ Model loaded from {weights_path}")
    print(f"   Inferred hidden_dim: {hidden_dim}")
    
    # Evaluate at multiple horizons
    horizons = [1, 10, 20]
    
    print(f"\n3. Evaluating Naive Baseline at horizons {horizons}...")
    baseline_results = compute_naive_baseline_multi_horizon(
        df, connectome, device, 
        horizons=horizons,
        max_worms=None,  # Use all worms
        batch_size=128
    )
    
    print(f"\n4. Evaluating GCN Model at horizons {horizons}...")
    gcn_results = evaluate_multi_horizon(
        model, df, connectome, device,
        horizons=horizons,
        max_worms=None,
        batch_size=128
    )
    
    # Print comparison
    print_comparison_table(baseline_results, gcn_results)
    
    # Create visualization
    print("\n5. Creating visualization...")
    plot_multi_horizon_comparison(baseline_results, gcn_results, output_dir="plots")
    
    # Save results
    results = {
        'baseline': baseline_results,
        'gcn': gcn_results,
        'horizons': horizons
    }
    
    results_path = Path(__file__).parent / "plots" / "multi_horizon_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Multi-horizon evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


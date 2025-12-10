"""
Multi-Horizon Evaluation Comparison
Compares baseline vs GAT-LSTM model at short-term (t+1), mid-term (t+10), and long-term (t+20) horizons.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import yaml

import torch
from torch_geometric.loader import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from data.preprocessing import load_or_create_dataset
from models.gat import GATLSTM, masked_mse_loss


def setup_device():
    """Setup compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_at_horizon(model, df, connectome, device, horizon, window_size=5, max_worms=10, batch_size=32):
    """Evaluate model at a specific horizon."""
    # Create dataset for this horizon
    dataset = load_or_create_dataset(
        df, connectome, 
        max_worms=max_worms,
        window_size=window_size,
        target_horizon=horizon,
        use_cache=True,
        verbose=False
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            mask = data.mask.view(-1)
            all_preds.append(pred.view(-1)[mask == 1].cpu())
            all_targets.append(data.y.view(-1)[mask == 1].cpu())
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    return {'mse': mse, 'mae': mae, 'r2': r2}


def compute_baseline_at_horizon(df, connectome, device, horizon, window_size=5, max_worms=10, batch_size=32):
    """Compute naive baseline at a specific horizon."""
    dataset = load_or_create_dataset(
        df, connectome,
        max_worms=max_worms,
        window_size=window_size,
        target_horizon=horizon,
        use_cache=True,
        verbose=False
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        # Naive baseline: X(t) predicts X(t+h)
        pred = data.x[:, -1:] if data.x.dim() == 2 and data.x.shape[1] > 1 else data.x
        mask = data.mask.view(-1)
        all_preds.append(pred.view(-1)[mask == 1].cpu())
        all_targets.append(data.y.view(-1)[mask == 1].cpu())
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    return {'mse': mse, 'mae': mae, 'r2': r2}


def plot_multi_horizon_comparison(baseline_results, model_results, output_dir="plots"):
    """
    Create comparison plot across multiple horizons.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    horizons = sorted(baseline_results.keys())
    baseline_r2 = [baseline_results[h]['r2'] for h in horizons]
    model_r2 = [model_results[h]['r2'] for h in horizons]
    baseline_mse = [baseline_results[h]['mse'] for h in horizons]
    model_mse = [model_results[h]['mse'] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    ax1.plot(horizons, baseline_r2, 'o-', label='Naive Baseline', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.plot(horizons, model_r2, 's-', label='GAT-LSTM', linewidth=2, markersize=8, color='#4ECDC4')
    ax1.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)
    
    # Add value labels
    for h, b_r2, g_r2 in zip(horizons, baseline_r2, model_r2):
        ax1.annotate(f'{b_r2:.3f}', (h, b_r2), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='#FF6B6B')
        ax1.annotate(f'{g_r2:.3f}', (h, g_r2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=9, color='#4ECDC4')
    
    # MSE comparison
    ax2.plot(horizons, baseline_mse, 'o-', label='Naive Baseline', linewidth=2, markersize=8, color='#FF6B6B')
    ax2.plot(horizons, model_mse, 's-', label='GAT-LSTM', linewidth=2, markersize=8, color='#4ECDC4')
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


def print_comparison_table(baseline_results, model_results):
    """
    Print formatted comparison table.
    """
    horizons = sorted(baseline_results.keys())
    
    print("\n" + "=" * 80)
    print("MULTI-HORIZON EVALUATION COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Horizon':<12} {'Metric':<12} {'Baseline':<15} {'GAT-LSTM':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for horizon in horizons:
        h_name = f"t+{horizon}"
        b = baseline_results[horizon]
        g = model_results[horizon]
        
        # R²
        r2_improvement = ((g['r2'] - b['r2']) / abs(b['r2']) * 100) if not np.isnan(b['r2']) and b['r2'] != 0 else 0
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
    print(f"  Short-term (t+1):  Baseline R²={baseline_results[1]['r2']:.4f}, GAT-LSTM R²={model_results[1]['r2']:.4f}")
    if 10 in horizons:
        print(f"  Mid-term (t+10):   Baseline R²={baseline_results[10]['r2']:.4f}, GAT-LSTM R²={model_results[10]['r2']:.4f}")
    if 20 in horizons:
        print(f"  Long-term (t+20):  Baseline R²={baseline_results[20]['r2']:.4f}, GAT-LSTM R²={model_results[20]['r2']:.4f}")
    
    # Check if model beats baseline at longer horizons
    improvements = []
    for h in horizons:
        if h > 1:  # Skip t+1 (baseline is too good)
            b_r2 = baseline_results[h]['r2']
            g_r2 = model_results[h]['r2']
            if not np.isnan(b_r2) and not np.isnan(g_r2):
                improvements.append(g_r2 > b_r2)
    
    if improvements:
        wins = sum(improvements)
        print(f"\n  GAT-LSTM beats baseline at {wins}/{len(improvements)} longer horizons")


def main():
    """
    Main comparison function.
    """
    print("=" * 80)
    print("MULTI-HORIZON EVALUATION: Baseline vs GAT-LSTM")
    print("=" * 80)
    
    device = setup_device()
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    connectome = load_connectome(verbose=False)
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Connectome: {connectome.num_nodes} neurons, {connectome.num_edges} edges")
    print(f"   Functional: {len(df)} samples, {df['worm'].nunique()} worms")
    
    # Load model config
    config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    gat_config = config.get('gat_lstm', {})
    train_config = config.get('training', {})
    window_size = train_config.get('window_size', 5)
    max_worms = train_config.get('max_worms', 10)
    batch_size = train_config.get('batch_size', 32)
    
    # Load trained model
    print("\n2. Loading trained GAT-LSTM model...")
    weights_dir = Path(__file__).parent.parent / "models" / "weights"
    weights_path = weights_dir / "gat_lstm_model_weights.pt"
    
    if not weights_path.exists():
        print(f"   ERROR: Model weights not found at {weights_path}")
        print("   Please train the model first:")
        print("   python src/train.py --model gat --epochs 50")
        return
    
    # Create model with config
    model = GATLSTM(config=gat_config)
    weights = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    
    print(f"   ✓ Model loaded from {weights_path}")
    print(f"   Parameters: {model.num_params:,}")
    print(f"   Window size: {window_size}")
    
    # Evaluate at multiple horizons
    horizons = [1, 5, 10, 20]
    
    baseline_results = {}
    model_results = {}
    
    for horizon in horizons:
        print(f"\n3. Evaluating at horizon t+{horizon}...")
        
        # Clear cache for different horizons
        print(f"   Computing baseline...")
        baseline_results[horizon] = compute_baseline_at_horizon(
            df, connectome, device, 
            horizon=horizon,
            window_size=window_size,
            max_worms=max_worms,
            batch_size=batch_size
        )
        print(f"   Baseline R²: {baseline_results[horizon]['r2']:.4f}")
        
        print(f"   Evaluating GAT-LSTM...")
        model_results[horizon] = evaluate_at_horizon(
            model, df, connectome, device,
            horizon=horizon,
            window_size=window_size,
            max_worms=max_worms,
            batch_size=batch_size
        )
        print(f"   GAT-LSTM R²: {model_results[horizon]['r2']:.4f}")
    
    # Print comparison
    print_comparison_table(baseline_results, model_results)
    
    # Create visualization
    print("\n4. Creating visualization...")
    output_dir = Path(__file__).parent / "plots"
    plot_multi_horizon_comparison(baseline_results, model_results, output_dir=output_dir)
    
    # Save results
    results = {
        'baseline': {str(k): v for k, v in baseline_results.items()},
        'gat_lstm': {str(k): v for k, v in model_results.items()},
        'horizons': horizons
    }
    
    results_path = output_dir / "multi_horizon_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Multi-horizon evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


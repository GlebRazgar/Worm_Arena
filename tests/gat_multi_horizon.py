"""
Multi-Horizon Evaluation for GAT-LSTM Model
Compares GAT-LSTM vs naive baseline at t+1, t+5, t+10, t+20 horizons.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch_geometric.loader import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from data.preprocessing import create_graph_temporal_dataset, WormGraphDataset
from models.gat import GATLSTM, masked_mse_loss


def setup_device():
    """Setup compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_model(model, loader, device):
    """
    Evaluate model on a data loader.
    
    Returns:
        dict: Metrics (mse, mae, r2)
    """
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
    
    if len(all_preds) == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    if len(preds) == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    # Compute metrics
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    return {'mse': mse, 'mae': mae, 'r2': r2}


def compute_naive_baseline(loader, device):
    """
    Compute naive baseline metrics: X̂(t+h) = X(t)
    Uses the last timestep in the window as the prediction.
    """
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Naive baseline: predict last timestep in window
            if data.x.dim() == 2 and data.x.shape[1] > 1:
                pred = data.x[:, -1:]  # [N, 1]
            else:
                pred = data.x
            
            target = data.y
            mask = data.mask.view(-1)
            
            all_preds.append(pred.view(-1)[mask == 1].cpu())
            all_targets.append(target.view(-1)[mask == 1].cpu())
    
    if len(all_preds) == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    if len(preds) == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    return {'mse': mse, 'mae': mae, 'r2': r2}


def plot_multi_horizon_comparison(baseline_results, gat_results, output_dir="tests/plots"):
    """
    Create comparison plot across multiple horizons.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    horizons = sorted(baseline_results.keys())
    baseline_r2 = [baseline_results[h]['r2'] for h in horizons]
    gat_r2 = [gat_results[h]['r2'] for h in horizons]
    baseline_mse = [baseline_results[h]['mse'] for h in horizons]
    gat_mse = [gat_results[h]['mse'] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    ax1.plot(horizons, baseline_r2, 'o-', label='Naive Baseline', linewidth=2, markersize=10, color='#FF6B6B')
    ax1.plot(horizons, gat_r2, 's-', label='GAT-LSTM', linewidth=2, markersize=10, color='#4ECDC4')
    ax1.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(horizons)
    
    # Add value labels
    for h, b_r2, g_r2 in zip(horizons, baseline_r2, gat_r2):
        if not np.isnan(b_r2):
            ax1.annotate(f'{b_r2:.3f}', (h, b_r2), textcoords="offset points", 
                        xytext=(0, 12), ha='center', fontsize=10, color='#FF6B6B', fontweight='bold')
        if not np.isnan(g_r2):
            ax1.annotate(f'{g_r2:.3f}', (h, g_r2), textcoords="offset points", 
                        xytext=(0, -18), ha='center', fontsize=10, color='#4ECDC4', fontweight='bold')
    
    # MSE comparison
    ax2.plot(horizons, baseline_mse, 'o-', label='Naive Baseline', linewidth=2, markersize=10, color='#FF6B6B')
    ax2.plot(horizons, gat_mse, 's-', label='GAT-LSTM', linewidth=2, markersize=10, color='#4ECDC4')
    ax2.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('MSE vs Prediction Horizon', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(horizons)
    
    plt.tight_layout()
    plt.savefig(output_path / "gat_multi_horizon_comparison.png", dpi=150)
    plt.close()
    
    print(f"   Saved: {output_path / 'gat_multi_horizon_comparison.png'}")


def print_comparison_table(baseline_results, gat_results):
    """
    Print formatted comparison table.
    """
    horizons = sorted(baseline_results.keys())
    
    print("\n" + "=" * 80)
    print("MULTI-HORIZON EVALUATION: GAT-LSTM vs Naive Baseline")
    print("=" * 80)
    
    print(f"\n{'Horizon':<12} {'Metric':<10} {'Baseline':<12} {'GAT-LSTM':<12} {'Improvement':<15}")
    print("-" * 80)
    
    for horizon in horizons:
        h_name = f"t+{horizon}"
        b = baseline_results[horizon]
        g = gat_results[horizon]
        
        # R²
        r2_improvement = ((g['r2'] - b['r2']) / abs(b['r2']) * 100) if not np.isnan(b['r2']) and b['r2'] != 0 else 0
        r2_better = "✓" if g['r2'] > b['r2'] else ""
        print(f"{h_name:<12} {'R²':<10} {b['r2']:<12.4f} {g['r2']:<12.4f} {r2_improvement:+.1f}% {r2_better}")
        
        # MSE
        mse_improvement = ((b['mse'] - g['mse']) / b['mse'] * 100) if b['mse'] > 0 else 0
        mse_better = "✓" if g['mse'] < b['mse'] else ""
        print(f"{'':<12} {'MSE':<10} {b['mse']:<12.2f} {g['mse']:<12.2f} {mse_improvement:+.1f}% {mse_better}")
        
        # MAE
        mae_improvement = ((b['mae'] - g['mae']) / b['mae'] * 100) if b['mae'] > 0 else 0
        print(f"{'':<12} {'MAE':<10} {b['mae']:<12.4f} {g['mae']:<12.4f} {mae_improvement:+.1f}%")
        
        print("-" * 80)
    
    # Summary
    print("\n📊 SUMMARY:")
    wins = 0
    for h in horizons:
        b_r2 = baseline_results[h]['r2']
        g_r2 = gat_results[h]['r2']
        if not np.isnan(b_r2) and not np.isnan(g_r2) and g_r2 > b_r2:
            wins += 1
    
    print(f"   GAT-LSTM beats naive baseline at {wins}/{len(horizons)} horizons (by R²)")
    
    # Show which horizons GAT-LSTM wins
    for h in horizons:
        b_r2 = baseline_results[h]['r2']
        g_r2 = gat_results[h]['r2']
        diff = g_r2 - b_r2
        pct = ((g_r2 - b_r2) / abs(b_r2) * 100) if b_r2 != 0 else 0
        status = "✓ WINS" if g_r2 > b_r2 else "✗ loses"
        print(f"   t+{h:>2}: GAT-LSTM R²={g_r2:.4f} vs Baseline R²={b_r2:.4f} → {status} ({pct:+.1f}%)")


def main():
    """
    Main comparison function.
    """
    print("=" * 80)
    print("🧠 MULTI-HORIZON EVALUATION: GAT-LSTM")
    print("=" * 80)
    
    device = setup_device()
    print(f"\nUsing device: {device}")
    
    # Configuration (match training settings)
    window_size = 5
    batch_size = 32
    max_worms = 5  # Same as training
    horizons = [1, 5, 10, 20]
    
    print(f"\nSettings:")
    print(f"   Window size: {window_size}")
    print(f"   Max worms: {max_worms}")
    print(f"   Horizons: {horizons}")
    
    # Load data
    print("\n1. Loading data...")
    start = time.time()
    connectome = load_connectome(verbose=False)
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Connectome: {connectome.num_nodes} neurons, {connectome.num_edges} edges")
    print(f"   Functional: {len(df)} samples, {df['worm'].nunique()} worms")
    print(f"   Loaded in {time.time()-start:.1f}s")
    
    # Load trained model
    print("\n2. Loading trained GAT-LSTM model...")
    weights_dir = Path(__file__).parent.parent / "models" / "weights"
    weights_path = weights_dir / "gat_lstm_model_weights.pt"
    
    if not weights_path.exists():
        print(f"   ❌ ERROR: Model weights not found at {weights_path}")
        print("   Please train the model first:")
        print("   python src/train.py --model gat --epochs 50")
        return
    
    # Load model with same config as training
    model = GATLSTM(
        gat_hidden=[32],
        gat_heads=[2],
        lstm_hidden=32,
        lstm_layers=1,
        dropout=0.0,
        residual=True,
        use_vectorized=True
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print(f"   ✓ Model loaded from {weights_path}")
    print(f"   Parameters: {model.num_params:,}")
    
    # Evaluate at each horizon
    baseline_results = {}
    gat_results = {}
    
    print(f"\n3. Evaluating at horizons {horizons}...")
    
    for horizon in horizons:
        print(f"\n   --- Horizon t+{horizon} ---")
        start = time.time()
        
        # Create dataset for this horizon (no caching to avoid mixing horizons)
        data_list = create_graph_temporal_dataset(
            df, connectome,
            max_worms=max_worms,
            window_size=window_size,
            target_horizon=horizon,
            verbose=False
        )
        
        if len(data_list) == 0:
            print(f"   ⚠ No samples for horizon t+{horizon}")
            baseline_results[horizon] = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
            gat_results[horizon] = {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
            continue
        
        dataset = WormGraphDataset(data_list)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"   Samples: {len(dataset)}, Batches: {len(loader)}")
        
        # Evaluate naive baseline
        baseline_results[horizon] = compute_naive_baseline(loader, device)
        print(f"   Baseline: R²={baseline_results[horizon]['r2']:.4f}, MSE={baseline_results[horizon]['mse']:.2f}")
        
        # Evaluate GAT-LSTM
        gat_results[horizon] = evaluate_model(model, loader, device)
        print(f"   GAT-LSTM: R²={gat_results[horizon]['r2']:.4f}, MSE={gat_results[horizon]['mse']:.2f}")
        
        print(f"   Time: {time.time()-start:.1f}s")
    
    # Print comparison table
    print_comparison_table(baseline_results, gat_results)
    
    # Create visualization
    print("\n4. Creating visualization...")
    plot_multi_horizon_comparison(baseline_results, gat_results)
    
    # Save results
    results = {
        'baseline': {str(k): v for k, v in baseline_results.items()},
        'gat_lstm': {str(k): v for k, v in gat_results.items()},
        'horizons': horizons,
        'window_size': window_size,
        'max_worms': max_worms
    }
    
    results_path = Path(__file__).parent / "plots" / "gat_multi_horizon_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Multi-horizon evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


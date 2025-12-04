"""
GAT-LSTM Evaluation and Visualization

Evaluates the trained GAT-LSTM model and generates visualization plots.

Usage:
    python tests/gat_vis.py
"""

import sys
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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


def load_trained_model(device, checkpoint_path=None):
    """
    Load trained GAT-LSTM model.
    
    Args:
        device: torch device
        checkpoint_path: Path to checkpoint (default: models/checkpoints/gat_lstm_best.pt)
    
    Returns:
        model: Trained GATLSTM model
        config: Model configuration
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent / "models" / "checkpoints" / "gat_lstm_best.pt"
    
    if not checkpoint_path.exists():
        # Try weights file
        weights_path = Path(__file__).parent.parent / "models" / "weights" / "gat_lstm_model_weights.pt"
        if weights_path.exists():
            print(f"   Loading from weights: {weights_path}")
            model = GATLSTM()
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
            model = model.to(device)
            return model, {}
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or {weights_path}")
    
    print(f"   Loading from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    model = GATLSTM(config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config


@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Evaluate model on data loader.
    
    Returns:
        dict: Metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    total_loss = 0.0
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        
        pred = model(data)
        loss = masked_mse_loss(pred, data.y, data.mask)
        total_loss += loss.item()
        num_batches += 1
        
        mask = data.mask.view(-1)
        all_preds.append(pred.view(-1)[mask == 1].cpu())
        all_targets.append(data.y.view(-1)[mask == 1].cpu())
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = mse ** 0.5
    
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    if len(preds) > 1:
        pred_centered = preds - torch.mean(preds)
        target_centered = targets - torch.mean(targets)
        correlation = (torch.sum(pred_centered * target_centered) / 
                      (torch.sqrt(torch.sum(pred_centered ** 2)) * 
                       torch.sqrt(torch.sum(target_centered ** 2)))).item()
    else:
        correlation = float('nan')
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'avg_loss': total_loss / max(num_batches, 1),
        'all_preds': preds.numpy(),
        'all_targets': targets.numpy()
    }


@torch.no_grad()
def compute_baseline(loader, device):
    """Compute naive baseline metrics."""
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        
        if data.x.dim() == 2 and data.x.shape[1] > 1:
            pred = data.x[:, -1:]
        else:
            pred = data.x
        
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


def plot_gat_vs_baseline(gat_metrics, baseline_metrics, output_dir="plots"):
    """
    Create bar chart comparing GAT-LSTM vs baseline.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics = ['MSE', 'MAE', 'R²']
    gat_values = [gat_metrics['mse'], gat_metrics['mae'], gat_metrics['r2']]
    baseline_values = [baseline_metrics['mse'], baseline_metrics['mae'], baseline_metrics['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Naive Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, gat_values, width, label='GAT-LSTM', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('GAT-LSTM vs Naive Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "gat_vs_baseline.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'gat_vs_baseline.png'}")


@torch.no_grad()
def extract_attention_weights(model, data, device):
    """
    Extract attention weights from GAT layers.
    
    Returns:
        dict: Attention weights per layer
    """
    model.eval()
    data = data.to(device)
    
    _, attention_weights = model(data, return_attention=True)
    
    # Convert to numpy
    attn_np = {}
    for k, v in attention_weights.items():
        attn_np[k] = v.cpu().numpy()
    
    return attn_np


def plot_attention_weights(attention_weights, connectome, output_dir="plots", top_k=100):
    """
    Visualize attention weights on connectome graph.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get edge_index
    edge_index = connectome.edge_index.numpy()
    
    # Average attention across heads for last layer
    last_layer = list(attention_weights.keys())[-1]
    attn = attention_weights[last_layer]  # [E, heads]
    attn_mean = attn.mean(axis=1)  # [E]
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    node_labels = list(connectome.node_label)
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)
    
    # Add edges with attention weights
    for i in range(min(len(attn_mean), edge_index.shape[1])):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst, weight=float(attn_mean[i]))
    
    # Get top-k edges by attention
    edge_weights = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    edge_weights.sort(key=lambda x: x[2], reverse=True)
    top_edges = edge_weights[:top_k]
    
    # Create subgraph with top edges
    G_sub = nx.DiGraph()
    nodes_in_top = set()
    for u, v, w in top_edges:
        G_sub.add_edge(u, v, weight=w)
        nodes_in_top.add(u)
        nodes_in_top.add(v)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout
    pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
    
    # Draw edges with varying width based on attention
    edges = G_sub.edges(data=True)
    weights = [d['weight'] * 10 for u, v, d in edges]
    
    nx.draw_networkx_edges(G_sub, pos, edge_color=weights, edge_cmap=plt.cm.Reds,
                          width=2, alpha=0.7, arrows=True, arrowsize=10,
                          connectionstyle="arc3,rad=0.1", ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_sub, pos, node_size=100, node_color='lightblue',
                          edgecolors='black', linewidths=0.5, ax=ax)
    
    # Add labels for some nodes
    if len(nodes_in_top) < 30:
        labels = {n: node_labels[n] for n in nodes_in_top}
        nx.draw_networkx_labels(G_sub, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f'GAT Attention Weights (Top {top_k} edges)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "gat_attention.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'gat_attention.png'}")


def plot_predictions_over_time(model, loader, device, num_samples=6, output_dir="plots"):
    """
    Plot predicted vs actual traces for sample neurons.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Get one batch
    data = next(iter(loader))
    data = data.to(device)
    
    with torch.no_grad():
        pred = model(data)
    
    # Get predictions and targets
    pred = pred.cpu().numpy()
    target = data.y.cpu().numpy()
    mask = data.mask.cpu().numpy()
    
    # Find valid neurons
    valid_neurons = np.where(mask.flatten() == 1)[0]
    sample_neurons = valid_neurons[:num_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, neuron_idx in enumerate(sample_neurons):
        ax = axes[i]
        
        ax.scatter([0], [target.flatten()[neuron_idx]], color='blue', s=100, label='Target', zorder=3)
        ax.scatter([0], [pred.flatten()[neuron_idx]], color='red', s=100, marker='x', label='Prediction', zorder=3)
        
        ax.axhline(y=target.flatten()[neuron_idx], color='blue', linestyle='--', alpha=0.5)
        ax.axhline(y=pred.flatten()[neuron_idx], color='red', linestyle='--', alpha=0.5)
        
        error = abs(pred.flatten()[neuron_idx] - target.flatten()[neuron_idx])
        ax.set_title(f'Neuron {neuron_idx}\nError: {error:.4f}', fontsize=10)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Calcium Activity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('GAT-LSTM Predictions vs Targets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "gat_predictions.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'gat_predictions.png'}")


def plot_error_distribution(gat_metrics, output_dir="plots"):
    """
    Plot error distribution.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preds = gat_metrics['all_preds']
    targets = gat_metrics['all_targets']
    errors = preds - targets
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error histogram
    ax1.hist(errors, bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax1.set_xlabel('Prediction Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter: predicted vs actual
    sample_idx = np.random.choice(len(preds), min(5000, len(preds)), replace=False)
    ax2.scatter(targets[sample_idx], preds[sample_idx], alpha=0.3, s=1, c='#4ECDC4')
    
    # Perfect prediction line
    min_val, max_val = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax2.set_xlabel('Actual', fontsize=12)
    ax2.set_ylabel('Predicted', fontsize=12)
    ax2.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "gat_error_distribution.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'gat_error_distribution.png'}")


def plot_training_curves(results_path=None, output_dir="plots"):
    """
    Plot training curves from results file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if results_path is None:
        results_path = Path(__file__).parent.parent / "models" / "checkpoints" / "gat_lstm_results.json"
    
    if not results_path.exists():
        print(f"   No results file found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    history = results.get('history', {})
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_r2 = history.get('val_r2', [])
    
    if not train_loss:
        print("   No training history found")
        return
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_loss, label='Train Loss', color='#FF6B6B', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', color='#4ECDC4', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² curve
    ax2.plot(epochs, val_r2, label='Val R²', color='#4ECDC4', linewidth=2)
    ax2.axhline(y=results.get('baseline_metrics', {}).get('r2', 0), 
                color='red', linestyle='--', label='Baseline R²')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Validation R² Score', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "gat_training_curves.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_path / 'gat_training_curves.png'}")


def main():
    print("=" * 70)
    print("GAT-LSTM EVALUATION AND VISUALIZATION")
    print("=" * 70)
    
    device = setup_device()
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    connectome = load_connectome(verbose=False)
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Connectome: {connectome.num_nodes} neurons, {connectome.num_edges} edges")
    print(f"   Functional: {len(df)} samples")
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = load_or_create_dataset(
        df, connectome,
        max_worms=None,
        window_size=50,
        target_horizon=1,
        use_cache=True,
        verbose=False
    )
    
    _, _, test_ds = dataset.split_by_worm()
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"   Test samples: {len(test_ds)}")
    
    # Load model
    print("\n3. Loading trained model...")
    try:
        model, config = load_trained_model(device)
        print(f"   ✓ Model loaded with {model.num_params:,} parameters")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("   Please train the model first: python src/train.py --model gat")
        return
    
    # Evaluate
    print("\n4. Evaluating model...")
    gat_metrics = evaluate_model(model, test_loader, device)
    baseline = compute_baseline(test_loader, device)
    
    print(f"\n   GAT-LSTM Results:")
    print(f"   {'Metric':<12} {'GAT-LSTM':<12} {'Baseline':<12}")
    print(f"   {'-'*36}")
    print(f"   {'MSE':<12} {gat_metrics['mse']:<12.4f} {baseline['mse']:<12.4f}")
    print(f"   {'MAE':<12} {gat_metrics['mae']:<12.4f} {baseline['mae']:<12.4f}")
    print(f"   {'R²':<12} {gat_metrics['r2']:<12.4f} {baseline['r2']:<12.4f}")
    
    # Generate plots
    print("\n5. Generating visualizations...")
    output_dir = Path(__file__).parent / "plots"
    
    plot_gat_vs_baseline(gat_metrics, baseline, output_dir)
    plot_error_distribution(gat_metrics, output_dir)
    plot_predictions_over_time(model, test_loader, device, output_dir=output_dir)
    plot_training_curves(output_dir=output_dir)
    
    # Extract and plot attention weights
    print("\n6. Extracting attention weights...")
    sample_data = next(iter(test_loader))
    attn_weights = extract_attention_weights(model, sample_data, device)
    print(f"   Attention layers: {list(attn_weights.keys())}")
    plot_attention_weights(attn_weights, connectome, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()


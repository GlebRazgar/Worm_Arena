"""
Training Pipeline for GAT-LSTM Model

Trains the GAT-LSTM model on C. elegans neural activity data.
Optimized for MPS (Apple Silicon) acceleration.
Includes optional Weights & Biases integration.

Usage:
    python src/train.py --model gat --epochs 50 --batch-size 64
    python src/train.py --model gat --epochs 50 --wandb  # With W&B logging
    python src/train.py --model gcn --epochs 20  # For simple GCN baseline
"""

import sys
from pathlib import Path
import argparse
import time
import json
import yaml

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from data.preprocessing import load_or_create_dataset, WormGraphDataset
from models.gat import GATLSTM, masked_mse_loss, create_gat_lstm_model
from models.test_gcn import SimpleTestGCN


def setup_device():
    """
    Setup compute device with priority: MPS > CUDA > CPU.
    
    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def load_config(config_path=None):
    """
    Load configuration from model.yaml.
    
    Args:
        config_path: Path to config file (default: configs/model.yaml)
    
    Returns:
        dict: Configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_weight_filename(model_name):
    """
    Generate weight filename based on model name.
    
    Args:
        model_name: Name of the model
    
    Returns:
        str: Weight filename
    """
    name = model_name.lower()
    if "gat" in name and "lstm" in name:
        return "gat_lstm_model_weights.pt"
    elif "gat" in name:
        return "gat_model_weights.pt"
    elif "gcn" in name and "lstm" in name:
        return "gcn_lstm_model_weights.pt"
    elif "gcn" in name:
        return "gcn_model_weights.pt"
    else:
        return f"{name.replace(' ', '_')}_model_weights.pt"


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a data loader.
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: torch device
    
    Returns:
        dict: Metrics (mse, mae, rmse, r2, correlation)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
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
    
    if num_batches == 0:
        return {
            'mse': float('nan'), 
            'mae': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'), 
            'correlation': float('nan'),
            'avg_loss': float('nan')
        }
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
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
        'avg_loss': total_loss / num_batches
    }


@torch.no_grad()
def compute_naive_baseline(loader, device):
    """
    Compute naive baseline metrics (X̂(t+1) = X(t)).
    """
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        
        # Naive: predict last timestep of window
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
    
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    return {'mse': mse, 'r2': r2}


def train_epoch(model, loader, optimizer, device, pbar=None):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader
        optimizer: Optimizer
        device: torch device
        pbar: Optional progress bar
    
    Returns:
        float: Average loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        pred = model(data)
        loss = masked_mse_loss(pred, data.y, data.mask)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if pbar is not None:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)
    
    return total_loss / max(num_batches, 1)


def train_gat_model(args):
    """
    Main training function for GAT-LSTM model.
    OPTIMIZED for M4 Max with limited memory.
    """
    # Set seeds for reproducibility
    set_seed(42)
    
    print("=" * 70)
    print("GAT-LSTM MODEL TRAINING (OPTIMIZED)")
    print("=" * 70)
    
    # Setup device
    device = setup_device()
    
    # Load config
    config = load_config()
    gat_config = config.get('gat_lstm', {})
    train_config = config.get('training', {})
    
    # Override with args (ULTRA optimized for 36GB RAM - NO SWAP)
    window_size = args.window_size or train_config.get('window_size', 5)    # Minimal
    batch_size = args.batch_size or train_config.get('batch_size', 32)      # Small to avoid swap
    epochs = args.epochs or train_config.get('epochs', 100)
    lr = args.lr or train_config.get('lr', 0.001)
    max_worms = args.max_worms or train_config.get('max_worms', 5)          # ~33K samples, fits in RAM
    
    # Load data
    print(f"\n1. Loading data...")
    t_start = time.time()
    
    connectome = load_connectome(verbose=False)
    df = load_functional_data(connectome=connectome, verbose=False)
    
    print(f"   Connectome: {connectome.num_nodes} neurons, {connectome.num_edges} edges")
    print(f"   Functional: {len(df)} samples, {df['worm'].nunique()} worms")
    print(f"   Data loaded in {time.time() - t_start:.2f}s")
    
    # Create dataset
    print(f"\n2. Creating graph-temporal dataset...")
    t_data = time.time()
    
    dataset = load_or_create_dataset(
        df, connectome,
        max_worms=max_worms,
        window_size=window_size,
        target_horizon=1,
        use_cache=True,
        verbose=True
    )
    
    print(f"   Dataset created in {time.time() - t_data:.2f}s")
    
    # Split by worm
    print(f"\n3. Splitting dataset by worm...")
    train_ds, val_ds, test_ds = dataset.split_by_worm()
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val:   {len(val_ds)} samples")
    print(f"   Test:  {len(test_ds)} samples")
    
    # Create loaders with optimized settings
    # Use larger batch size for validation (no gradients needed)
    # Use generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    print(f"\n4. Initializing GAT-LSTM model...")
    model = GATLSTM(config=gat_config)
    model = model.to(device)
    model_name = "GATLSTM"
    weight_filename = get_weight_filename(model_name)
    
    print(f"   {model_name} with {model.num_params:,} parameters")
    print(f"   GAT layers: {model.gat_hidden}")
    print(f"   GAT heads: {model.gat_heads}")
    print(f"   LSTM hidden: {model.lstm_hidden}, layers: {model.lstm_layers}")
    print(f"   Window size: {window_size}")
    print(f"   Residual: {model.residual}")
    print(f"   Vectorized: {model.use_vectorized}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=train_config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Compute baseline
    print(f"\n5. Computing naive baseline...")
    baseline = compute_naive_baseline(test_loader, device)
    print(f"   Naive Baseline R²: {baseline['r2']:.4f}")
    print(f"   Naive Baseline MSE: {baseline['mse']:.6f}")
    
    # Initialize Weights & Biases (if enabled)
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("   ⚠ wandb not installed. Run: pip install wandb")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"gat-lstm-w{window_size}-b{batch_size}",
                config={
                    "model": "GAT-LSTM",
                    "gat_hidden": model.gat_hidden,
                    "gat_heads": model.gat_heads,
                    "lstm_hidden": model.lstm_hidden,
                    "lstm_layers": model.lstm_layers,
                    "num_params": model.num_params,
                    "window_size": window_size,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "max_worms": max_worms,
                    "baseline_r2": baseline['r2'],
                    "baseline_mse": baseline['mse'],
                }
            )
            print(f"   ✓ W&B initialized: {wandb.run.url}")
    
    # Setup directories
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(__file__).parent.parent / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n6. Training for {epochs} epochs...")
    print(f"   Estimated time per epoch: ~{len(train_loader) * 0.05:.0f}s ({len(train_loader) * 0.05 / 60:.1f} min)")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = train_config.get('patience', 15)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'epoch_times': []
    }
    
    t_train_start = time.time()
    
    epoch_pbar = tqdm(range(epochs), desc="Epochs", position=0, ncols=100)
    
    for epoch in epoch_pbar:
        t_epoch = time.time()
        
        # Train
        batch_pbar = tqdm(total=len(train_loader), desc=f"  Epoch {epoch+1}", 
                         position=1, leave=False, ncols=80)
        train_loss = train_epoch(model, train_loader, optimizer, device, pbar=batch_pbar)
        batch_pbar.close()
        
        train_time = time.time() - t_epoch
        
        # Validate
        t_val = time.time()
        val_metrics = evaluate(model, val_loader, device)
        val_time = time.time() - t_val
        val_loss = val_metrics['avg_loss']
        val_r2 = val_metrics['r2']
        
        epoch_time = time.time() - t_epoch
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['epoch_times'].append(epoch_time)
        
        # Log to Weights & Biases
        if args.wandb and WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mse": val_metrics.get('mse', 0),
                "val/mae": val_metrics.get('mae', 0),
                "val/r2": val_metrics.get('r2', 0),
                "val/correlation": val_metrics.get('correlation', 0),
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "best_val_loss": best_val_loss,
                "beats_baseline": val_r2 > baseline['r2'] if val_r2 else False,
            }
            wandb.log(log_dict)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'config': gat_config
            }, checkpoint_dir / "gat_lstm_best.pt")
        else:
            patience_counter += 1
        
        # Save weights
        torch.save(model.state_dict(), weights_dir / weight_filename)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'R²': f'{val_r2:.3f}',
            'time': f'{epoch_time:.1f}s'
        })
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(
                f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} ({train_time:.1f}s) | "
                f"Val: {val_loss:.4f} ({val_time:.1f}s) | R²: {val_r2:.4f} | Total: {epoch_time:.1f}s"
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n   Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    total_time = time.time() - t_train_start
    avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
    
    print("-" * 70)
    print(f"   Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Average epoch time: {avg_epoch_time:.1f}s")
    print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
    print(f"   Final weights saved to: {weights_dir / weight_filename}")
    
    # Final evaluation
    print(f"\n7. Final evaluation on test set...")
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / "gat_lstm_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n   {'Metric':<12} {'GAT-LSTM':<12} {'Naive':<12} {'Improvement':<12}")
    print(f"   {'-'*48}")
    
    for metric in ['mse', 'mae', 'rmse', 'r2', 'correlation']:
        val = test_metrics.get(metric, 0)
        baseline_val = baseline.get(metric, val)
        
        if metric in ['mse', 'mae', 'rmse']:
            improvement = (baseline_val - val) / baseline_val * 100 if baseline_val > 0 else 0
        else:
            improvement = (val - baseline_val) / abs(baseline_val) * 100 if baseline_val != 0 else 0
        
        print(f"   {metric.upper():<12} {val:<12.6f} {baseline_val:<12.6f} {improvement:+.1f}%")
    
    # Save results
    results = {
        'model': model_name,
        'test_metrics': test_metrics,
        'baseline_metrics': baseline,
        'best_epoch': best_epoch + 1,
        'best_val_loss': best_val_loss,
        'total_time_seconds': total_time,
        'config': {
            'gat': gat_config,
            'window_size': window_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'max_worms': max_worms
        },
        'history': history
    }
    
    with open(checkpoint_dir / "gat_lstm_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   Results saved to {checkpoint_dir / 'gat_lstm_results.json'}")
    
    # Log final results to W&B
    if args.wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "test/mse": test_metrics['mse'],
            "test/mae": test_metrics['mae'],
            "test/rmse": test_metrics['rmse'],
            "test/r2": test_metrics['r2'],
            "final/best_epoch": best_epoch + 1,
            "final/total_time_min": total_time / 60,
            "final/beats_baseline": test_metrics['r2'] > baseline['r2'],
        })
        wandb.finish()
        print("   ✓ W&B run finished")
    
    # Summary
    print("\n" + "=" * 70)
    if test_metrics['r2'] > baseline['r2']:
        improvement = (test_metrics['r2'] - baseline['r2']) / baseline['r2'] * 100
        print(f"✓ GAT-LSTM beats naive baseline by {improvement:.1f}% R²")
    else:
        print("⚠ GAT-LSTM did not beat naive baseline")
    print("=" * 70)
    
    return model, history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GAT-LSTM or GCN Model")
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gcn'],
                        help='Model type: gat (GAT-LSTM) or gcn (SimpleTestGCN)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--window-size', type=int, default=None, help='Temporal window size')
    parser.add_argument('--max-worms', type=int, default=None, help='Max worms to use')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='worm-arena', help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None, help='W&B run name')
    
    args = parser.parse_args()
    
    if args.model == 'gat':
        train_gat_model(args)
    else:
        # Redirect to existing GCN training
        print("For GCN training, use: python src/train_test_gcn.py")
        print("This script is for GAT-LSTM training.")


if __name__ == "__main__":
    main()

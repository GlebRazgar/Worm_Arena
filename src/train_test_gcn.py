"""
Training Script for Simple GCN Test Model
Validates the data pipeline and tests basic learning capability.
Optimized for Apple MPS (M4 Max).
"""

import sys
from pathlib import Path
import argparse
import time
import json

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from data.preprocessing import load_or_create_dataset
from models.test_gcn import SimpleTestGCN, masked_mse_loss, evaluate, compute_naive_baseline
# DISABLED: GCN-LSTM for now
# from models.gcn_lstm import GCNLSTM
# from models.gcn_lstm_advanced import AdvancedGCNLSTM


def setup_device():
    """
    Setup compute device with MPS priority for Apple Silicon.
    
    Priority: MPS > CUDA > CPU
    
    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    
    return device


def train_epoch(model, loader, optimizer, device, pbar=None):
    """
    Train for one epoch (MPS optimized).
    
    Returns:
        average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for data in loader:
        # Move to device (ensure all tensors stay on MPS)
        data = data.to(device, non_blocking=False)
        
        optimizer.zero_grad()
        pred = model(data)
        loss = masked_mse_loss(pred, data.y, data.mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


def train_model(args):
    """
    Main training function.
    """
    print("=" * 70)
    print("SIMPLE GCN TEST MODEL - TRAINING")
    print("=" * 70)
    
    # Setup device
    device = setup_device()
    
    # Load data
    print("\n1. Loading data...")
    t0 = time.time()
    
    connectome = load_connectome(verbose=False)
    print(f"   Connectome: {connectome.num_nodes} neurons, {connectome.num_edges} edges")
    
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Functional: {len(df)} samples, {df['worm'].nunique()} worms")
    print(f"   Data loaded in {time.time() - t0:.2f}s")
    
    # Create dataset
    print(f"\n2. Creating graph-temporal dataset...")
    t0 = time.time()
    
    # SimpleTestGCN only works with window_size=1 (single timestep input)
    # Force window_size=1 regardless of args
    dataset = load_or_create_dataset(
        df, connectome, 
        max_worms=args.max_worms,
        window_size=1,  # Always use 1 for SimpleTestGCN
        target_horizon=1,  # Always train on t+1
        use_cache=True, 
        verbose=True
    )
    print(f"   Dataset created in {time.time() - t0:.2f}s")
    
    if len(dataset) == 0:
        print("ERROR: No samples in dataset!")
        return
    
    # Split dataset
    print(f"\n3. Splitting dataset by worm...")
    train_ds, val_ds, test_ds = dataset.split_by_worm(
        train_ratio=0.8, 
        val_ratio=0.1
    )
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val:   {len(val_ds)} samples")
    print(f"   Test:  {len(test_ds)} samples")
    
    # Create data loaders with MPS optimization
    # Pin memory and num_workers=0 for MPS (MPS doesn't support multiprocessing)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing
        pin_memory=False  # MPS doesn't use pinned memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model
    print(f"\n4. Initializing model...")
    # DISABLED: GCN-LSTM for now (not working yet)
    # Only use SimpleTestGCN
    model = SimpleTestGCN(hidden_dim=args.hidden_dim)
    model_name = "SimpleTestGCN"
    
    # Generate model-specific weight filename
    def get_weight_filename(model_name):
        """Convert model name to weight filename."""
        # Convert to lowercase and replace common patterns
        name = model_name.lower()
        # Handle specific cases
        if "gcn" in name and "lstm" in name:
            if "advanced" in name:
                return "advanced_gcn_lstm_model_weights.pt"
            return "gcn_lstm_model_weights.pt"
        elif "gcn" in name:
            return "gcn_model_weights.pt"
        elif "gat" in name:
            return "gat_model_weights.pt"
        elif "lstm" in name:
            return "lstm_model_weights.pt"
        else:
            # Generic fallback
            return f"{name.lower().replace(' ', '_')}_model_weights.pt"
    
    weight_filename = get_weight_filename(model_name)
    
    model = model.to(device)
    
    # DISABLED: torch.compile() causes shape inference issues with temporal windows
    # MPS Optimization: Compile model for better GPU utilization
    # if device.type == 'mps' and hasattr(torch, 'compile'):
    #     try:
    #         print(f"   Compiling model for MPS acceleration...")
    #         model = torch.compile(model, mode='reduce-overhead')
    #         print(f"   ✓ Model compiled for MPS")
    #     except Exception as e:
    #         print(f"   ⚠ Compilation failed (using eager mode): {e}")
    
    print(f"   {model_name} with {model.num_params} parameters")
    print(f"   Hidden dim: {args.hidden_dim}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Compute naive baseline for comparison
    print(f"\n5. Computing naive baseline (X̂(t+1) = X(t))...")
    baseline_metrics = compute_naive_baseline(test_loader, device)
    print(f"   Naive Baseline R²: {baseline_metrics['r2']:.4f}")
    print(f"   Naive Baseline MSE: {baseline_metrics['mse']:.6f}")
    
    # Training loop
    print(f"\n6. Training for {args.epochs} epochs...")
    print(f"   Batches per epoch: {len(train_loader)}")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'epoch_times': []}
    
    checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Weights directory (save after each epoch)
    weights_dir = Path(__file__).parent.parent / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(args.epochs), desc="Epochs", position=0, ncols=100)
    
    for epoch in epoch_pbar:
        t_epoch = time.time()
        
        # Inner progress bar for batches within epoch
        batch_pbar = tqdm(total=len(train_loader), desc=f"  Epoch {epoch+1}", 
                         position=1, leave=False, ncols=80)
        
        # Train
        t_train = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, pbar=batch_pbar)
        train_time = time.time() - t_train
        
        batch_pbar.close()
        
        # Validate
        t_val = time.time()
        val_metrics = evaluate(model, val_loader, device)
        val_time = time.time() - t_val
        val_loss = val_metrics['avg_loss']
        val_r2 = val_metrics['r2']
        
        epoch_time = time.time() - t_epoch
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['epoch_times'].append(epoch_time)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, checkpoint_dir / "simple_gcn_best.pt")
        
        # Save weights after each epoch (overwrite previous)
        torch.save(model.state_dict(), weights_dir / weight_filename)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'R²': f'{val_r2:.3f}',
            'time': f'{epoch_time:.1f}s'
        })
        
        # Print detailed timing every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"   Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train: {train_loss:.4f} ({train_time:.1f}s) | "
                      f"Val: {val_loss:.4f} ({val_time:.1f}s) | "
                      f"R²: {val_r2:.4f} | "
                      f"Total: {epoch_time:.1f}s")
    
    epoch_pbar.close()
    
    total_time = time.time() - t_start
    avg_epoch_time = sum(history['epoch_times']) / len(history['epoch_times'])
    print("-" * 70)
    print(f"   Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Average epoch time: {avg_epoch_time:.1f}s")
    print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
    print(f"   Final weights saved to: {weights_dir / weight_filename}")
    
    # Load best model for final evaluation
    print(f"\n7. Final evaluation on test set...")
    checkpoint = torch.load(checkpoint_dir / "simple_gcn_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n   {'Metric':<12} {'GCN':<12} {'Naive':<12} {'Improvement'}")
    print(f"   {'-'*48}")
    print(f"   {'MSE':<12} {test_metrics['mse']:<12.6f} {baseline_metrics['mse']:<12.6f} "
          f"{(baseline_metrics['mse'] - test_metrics['mse']) / baseline_metrics['mse'] * 100:+.1f}%")
    print(f"   {'MAE':<12} {test_metrics['mae']:<12.6f} {baseline_metrics['mae']:<12.6f} "
          f"{(baseline_metrics['mae'] - test_metrics['mae']) / baseline_metrics['mae'] * 100:+.1f}%")
    print(f"   {'RMSE':<12} {test_metrics['rmse']:<12.6f} {baseline_metrics['rmse']:<12.6f} "
          f"{(baseline_metrics['rmse'] - test_metrics['rmse']) / baseline_metrics['rmse'] * 100:+.1f}%")
    print(f"   {'R²':<12} {test_metrics['r2']:<12.4f} {baseline_metrics['r2']:<12.4f} "
          f"{(test_metrics['r2'] - baseline_metrics['r2']) / abs(baseline_metrics['r2']) * 100:+.1f}%")
    print(f"   {'Correlation':<12} {test_metrics['correlation']:<12.4f} {baseline_metrics['correlation']:<12.4f}")
    
    # Save results
    results = {
        'gcn_metrics': test_metrics,
        'baseline_metrics': baseline_metrics,
        'training_history': history,
        'config': {
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_worms': args.max_worms,
            'window_size': args.window_size,
            'lstm_hidden': getattr(args, 'lstm_hidden', None),
            'num_lstm_layers': getattr(args, 'num_lstm_layers', None),
            'num_gcn_layers': getattr(args, 'num_gcn_layers', None),
            'advanced': getattr(args, 'advanced', False),
        },
        'best_epoch': best_epoch,
        'total_time': total_time,
    }
    
    with open(checkpoint_dir / "simple_gcn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   Results saved to {checkpoint_dir / 'simple_gcn_results.json'}")
    
    # Final summary
    print("\n" + "=" * 70)
    if test_metrics['r2'] > baseline_metrics['r2']:
        print("✓ SUCCESS: GCN outperforms naive baseline!")
        print(f"  R² improvement: {test_metrics['r2'] - baseline_metrics['r2']:.4f}")
    else:
        print("⚠ GCN did not beat naive baseline. Check data alignment or model capacity.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCN or GCN-LSTM Model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=64, help='GCN hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max-worms', type=int, default=None, help='Max worms to use')
    parser.add_argument('--window-size', type=int, default=10, help='Temporal window size (1=GCN only, >1=GCN-LSTM)')
    parser.add_argument('--lstm-hidden', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--num-lstm-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--num-gcn-layers', type=int, default=3, help='Number of GCN layers (advanced model only)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--advanced', action='store_true', help='Use advanced model (multi-layer GCN + edge-gated + bidirectional LSTM)')
    
    args = parser.parse_args()
    train_model(args)


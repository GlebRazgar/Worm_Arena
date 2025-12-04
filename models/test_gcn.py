"""
Simple GCN Test Model
Minimal 1-layer GCN to validate the data pipeline before building complex architectures.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SimpleTestGCN(nn.Module):
    """
    Minimal GCN with RESIDUAL connection to test data pipeline.
    
    Architecture:
        Input [N, 1] → GCN(1→hidden) → ReLU → Linear(hidden→1) → δ [N, 1]
        Output = Input + δ  (Residual: predict the CHANGE, not absolute value)
    
    Uses connectome structure for message passing.
    Predicts next-timestep activation by learning the delta from current activation.
    
    Why residual? When baseline R²=0.95 (X(t+1) ≈ X(t)), the model should focus
    on predicting the small change, not reproducing the whole signal.
    """
    
    def __init__(self, hidden_dim=32):
        """
        Args:
            hidden_dim: Hidden dimension for GCN layer (default 32)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GCN layer: 1 feature (calcium) → hidden_dim
        self.gcn = GCNConv(1, hidden_dim)
        
        # Prediction head: hidden_dim → 1 (predicts DELTA, not absolute value)
        self.predictor = nn.Linear(hidden_dim, 1)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data):
        """
        Forward pass with RESIDUAL connection.
        
        Args:
            data: PyG Data object with:
                - x: [N, 1] or [N, window_size] current activation(s)
                - edge_index: [2, E] connectome edges
                - mask: [N, 1] valid neurons (optional)
        
        Returns:
            predictions: [N, 1] predicted next-timestep activation
                         = x_t + delta (residual prediction)
        """
        x = data.x  # [N, 1] or [N, window_size]
        edge_index = data.edge_index  # [2, E]
        
        # Handle temporal windows: use last timestep
        if x.dim() == 2 and x.shape[1] > 1:
            x_t = x[:, -1:]  # [N, 1] - take last timestep
        else:
            x_t = x  # [N, 1]
        
        # Graph convolution
        h = self.gcn(x_t, edge_index)  # [N, hidden_dim]
        h = torch.relu(h)
        
        # Predict the CHANGE (delta), not the absolute value
        delta = self.predictor(h)  # [N, 1]
        
        # RESIDUAL: Output = Input + Delta
        # This means: if delta=0, we get the naive baseline (X(t+1) = X(t))
        # The model only needs to learn the correction!
        out = x_t + delta
        
        return out


def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss only on masked (valid) neurons.
    
    Args:
        pred: [N, 1] predictions
        target: [N, 1] targets
        mask: [N, 1] mask (1=valid, 0=ignore)
    
    Returns:
        scalar loss
    """
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    mask = mask.view(-1)
    
    # Only compute loss on valid entries
    valid_pred = pred[mask == 1]
    valid_target = target[mask == 1]
    
    if len(valid_pred) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    loss = torch.mean((valid_pred - valid_target) ** 2)
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a data loader.
    
    Args:
        model: SimpleTestGCN model
        loader: DataLoader with graph data
        device: torch device
    
    Returns:
        dict with metrics: mse, mae, rmse, r2, correlation
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
        
        # Collect for metrics
        mask = data.mask.view(-1)
        all_preds.append(pred.view(-1)[mask == 1].cpu())
        all_targets.append(data.y.view(-1)[mask == 1].cpu())
    
    if num_batches == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'rmse': float('nan'), 
                'r2': float('nan'), 'correlation': float('nan')}
    
    # Concatenate all predictions and targets
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = mse ** 0.5
    
    # R² score
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    # Correlation
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


def compute_naive_baseline(loader, device):
    """
    Compute naive baseline metrics (X̂(t+1) = X(t)).
    
    Args:
        loader: DataLoader with graph data
        device: torch device
    
    Returns:
        dict with metrics
    """
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        
        # Naive baseline: predict X(t) for X(t+1)
        # Handle both single timestep [N, 1] and temporal window [N, window_size]
        if data.x.shape[1] == 1:
            pred = data.x  # [N, 1]
        else:
            # For temporal windows, use the last timestep
            pred = data.x[:, -1:]  # [N, 1] - last timestep of window
        
        target = data.y  # [N, 1]
        mask = data.mask.view(-1)
        
        all_preds.append(pred.view(-1)[mask == 1].cpu())
        all_targets.append(target.view(-1)[mask == 1].cpu())
    
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
        'correlation': correlation
    }


@torch.no_grad()
def evaluate_multi_horizon(model, df, connectome, device, horizons=[1, 10, 20], max_worms=None, batch_size=128):
    """
    Evaluate model at multiple prediction horizons.
    
    Creates separate datasets for each horizon and evaluates.
    
    Args:
        model: Model to evaluate
        df: DataFrame with functional data
        connectome: Connectome graph
        device: torch device
        horizons: List of horizons to evaluate [1, 10, 20]
        max_worms: Max worms to use
        batch_size: Batch size for evaluation
    
    Returns:
        dict: {horizon: {mse, mae, rmse, r2, correlation}} for each horizon
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.preprocessing import load_or_create_dataset
    from torch_geometric.loader import DataLoader
    
    results = {}
    
    for horizon in horizons:
        print(f"   Evaluating horizon t+{horizon}...")
        # Create dataset for this specific horizon
        dataset = load_or_create_dataset(
            df, connectome, 
            max_worms=max_worms,
            window_size=1,  # Single timestep input
            target_horizon=horizon,
            use_cache=True,
            verbose=False
        )
        
        if len(dataset) == 0:
            results[horizon] = {'mse': float('nan'), 'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan')}
            continue
        
        # Split and use test set
        _, _, test_ds = dataset.split_by_worm()
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Evaluate model
        metrics = evaluate(model, test_loader, device)
        results[horizon] = metrics
        print(f"      R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")
    
    return results


def compute_naive_baseline_multi_horizon(df, connectome, device, horizons=[1, 10, 20], max_worms=None, batch_size=128):
    """
    Compute naive baseline at multiple horizons.
    
    For horizon h: X̂(t+h) = X(t) (always predict current state)
    
    Args:
        df: DataFrame with functional data
        connectome: Connectome graph
        device: torch device
        horizons: List of horizons to evaluate
        max_worms: Max worms to use
        batch_size: Batch size
    
    Returns:
        dict: {horizon: {mse, mae, rmse, r2, correlation}}
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.preprocessing import load_or_create_dataset
    from torch_geometric.loader import DataLoader
    
    results = {}
    
    for horizon in horizons:
        print(f"   Computing baseline for horizon t+{horizon}...")
        # Create dataset for this horizon
        dataset = load_or_create_dataset(
            df, connectome,
            max_worms=max_worms,
            window_size=1,
            target_horizon=horizon,
            use_cache=True,
            verbose=False
        )
        
        if len(dataset) == 0:
            results[horizon] = {'mse': float('nan'), 'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan')}
            continue
        
        _, _, test_ds = dataset.split_by_worm()
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_preds = []
        all_targets = []
        
        for data in test_loader:
            data = data.to(device)
            
            # Naive baseline: always predict X(t) regardless of horizon
            if data.x.shape[1] == 1:
                pred = data.x  # [N, 1]
            else:
                pred = data.x[:, -1:]  # [N, 1] - last timestep
            
            target = data.y  # [N, 1] - this is now t+horizon
            mask = data.mask.view(-1)
            
            all_preds.append(pred.view(-1)[mask == 1].cpu())
            all_targets.append(target.view(-1)[mask == 1].cpu())
        
        if len(all_preds) == 0:
            results[horizon] = {'mse': float('nan'), 'r2': float('nan')}
            continue
        
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
        
        results[horizon] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation
        }
        print(f"      R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return results


if __name__ == "__main__":
    # Test the model
    print("Testing SimpleTestGCN...")
    print("=" * 70)
    
    # Create dummy data
    from torch_geometric.data import Data
    
    N = 196  # Neurons
    E = 4085  # Edges
    
    # Dummy graph
    edge_index = torch.randint(0, N, (2, E))
    x = torch.randn(N, 1)
    y = torch.randn(N, 1)
    mask = torch.ones(N, 1)
    
    data = Data(x=x, edge_index=edge_index, mask=mask, y=y)
    
    # Create model
    model = SimpleTestGCN(hidden_dim=32)
    print(f"Model parameters: {model.num_params}")
    
    # Forward pass
    pred = model(data)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    
    # Loss
    loss = masked_mse_loss(pred, y, mask)
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n" + "=" * 70)
    print("✓ Model test complete!")


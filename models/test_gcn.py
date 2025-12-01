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
                - x: [N, 1] current activation
                - edge_index: [2, E] connectome edges
                - mask: [N, 1] valid neurons (optional)
        
        Returns:
            predictions: [N, 1] predicted next-timestep activation
                         = x + delta (residual prediction)
        """
        x = data.x  # [N, 1]
        edge_index = data.edge_index  # [2, E]
        
        # Graph convolution
        h = self.gcn(x, edge_index)  # [N, hidden_dim]
        h = torch.relu(h)
        
        # Predict the CHANGE (delta), not the absolute value
        delta = self.predictor(h)  # [N, 1]
        
        # RESIDUAL: Output = Input + Delta
        # This means: if delta=0, we get the naive baseline (X(t+1) = X(t))
        # The model only needs to learn the correction!
        out = x + delta
        
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


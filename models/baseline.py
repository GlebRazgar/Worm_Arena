"""
Baseline Models for Temporal Prediction
Naive predictor: X̂(t+1) = X(t)
"""

import torch
import torch.nn as nn
import numpy as np


class NaivePredictor(nn.Module):
    """
    Naive baseline predictor that assumes next state equals current state.
    Prediction: X̂(t+1) = X(t)
    
    This is a "thunk" model with no trainable parameters.
    Properly handles masking where neurons may be missing in recordings.
    """
    
    def __init__(self):
        super().__init__()
        # No parameters - just a placeholder
        self.num_params = 0
    
    def forward(self, x, mask=None):
        """
        Forward pass that simply returns the input.
        
        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, num_neurons]
            mask: Optional mask [same shape as x], 1=valid, 0=missing/padding
        
        Returns:
            predictions: Copy of input (predicting t+1 = t)
            mask: Copy of input mask
        """
        return x.clone(), mask.clone() if mask is not None else None
    
    def predict_next_step(self, x_t, mask_t=None):
        """
        Predict single next timestep.
        
        Args:
            x_t: Current state [batch, num_neurons] or [batch]
            mask_t: Current mask [same shape as x_t]
        
        Returns:
            x_t_plus_1: Predicted next state (copy of x_t)
            mask_t_plus_1: Mask for prediction
        """
        return x_t.clone(), mask_t.clone() if mask_t is not None else None


def create_temporal_pairs(data, mask, horizon=1):
    """
    Create (X_t, X_{t+1}) pairs from time series data.
    VECTORIZED VERSION - No Python loops!
    
    Args:
        data: Tensor [num_samples, seq_len] - Time series data
        mask: Tensor [num_samples, seq_len] - Valid data mask (1=valid, 0=missing)
        horizon: int - Prediction horizon (default=1 for next timestep)
    
    Returns:
        dict containing:
            - X: Input states [num_pairs, 1]
            - Y: Target states [num_pairs, 1]
            - X_mask: Input masks [num_pairs, 1]
            - Y_mask: Target masks [num_pairs, 1]
            - valid_pairs: Number of valid pairs created
    """
    num_samples, seq_len = data.shape
    
    # Create X (t) and Y (t+horizon) by slicing
    X = data[:, :-horizon]  # [num_samples, seq_len-horizon]
    Y = data[:, horizon:]    # [num_samples, seq_len-horizon]
    X_mask = mask[:, :-horizon]
    Y_mask = mask[:, horizon:]
    
    # Find valid pairs where both X and Y are valid
    valid = (X_mask == 1) & (Y_mask == 1)  # [num_samples, seq_len-horizon]
    
    # Flatten and filter using boolean indexing
    X_flat = X[valid].unsqueeze(-1)  # [num_valid_pairs, 1]
    Y_flat = Y[valid].unsqueeze(-1)  # [num_valid_pairs, 1]
    X_mask_flat = X_mask[valid].unsqueeze(-1)
    Y_mask_flat = Y_mask[valid].unsqueeze(-1)
    
    if X_flat.shape[0] == 0:
        raise ValueError("No valid temporal pairs found. Check your mask.")
    
    return {
        'X': X_flat,
        'Y': Y_flat,
        'X_mask': X_mask_flat,
        'Y_mask': Y_mask_flat,
        'valid_pairs': X_flat.shape[0]
    }


def evaluate_predictions(predictions, targets, mask=None, metrics=['mse', 'mae', 'rmse']):
    """
    Evaluate predictions against targets.
    
    Args:
        predictions: Predicted values [num_samples, ...]
        targets: Ground truth values [same shape as predictions]
        mask: Optional mask (1=valid, 0=ignore) [same shape]
        metrics: List of metrics to compute
    
    Returns:
        dict: Computed metrics
    """
    # Apply mask if provided
    if mask is not None:
        predictions = predictions[mask == 1]
        targets = targets[mask == 1]
    
    if len(predictions) == 0:
        return {metric: float('nan') for metric in metrics}
    
    results = {}
    
    # Compute requested metrics
    if 'mse' in metrics:
        results['mse'] = torch.mean((predictions - targets) ** 2).item()
    
    if 'mae' in metrics:
        results['mae'] = torch.mean(torch.abs(predictions - targets)).item()
    
    if 'rmse' in metrics:
        results['rmse'] = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    if 'r2' in metrics:
        # R² score
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        results['r2'] = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    
    # Correlation
    if 'correlation' in metrics:
        if len(predictions) > 1:
            pred_centered = predictions - torch.mean(predictions)
            target_centered = targets - torch.mean(targets)
            correlation = torch.sum(pred_centered * target_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2)) * 
                torch.sqrt(torch.sum(target_centered ** 2))
            )
            results['correlation'] = correlation.item()
        else:
            results['correlation'] = float('nan')
    
    return results


def compare_baselines(data, mask):
    """
    Compare different baseline strategies.
    
    Args:
        data: Tensor [num_samples, seq_len]
        mask: Tensor [same shape]
    
    Returns:
        dict: Performance of different baselines
    """
    # Create temporal pairs
    pairs = create_temporal_pairs(data, mask)
    X, Y = pairs['X'], pairs['Y']
    
    results = {}
    
    # 1. Naive predictor (X̂(t+1) = X(t))
    naive_pred = X.clone()
    results['naive'] = evaluate_predictions(naive_pred, Y, metrics=['mse', 'mae', 'rmse', 'correlation'])
    
    # 2. Random predictor (X̂(t+1) ~ N(μ, σ²) from data)
    random_pred = torch.randn_like(X) * torch.std(X) + torch.mean(X)
    results['random'] = evaluate_predictions(random_pred, Y, metrics=['mse', 'mae', 'rmse', 'correlation'])
    
    return results, pairs


if __name__ == "__main__":
    # Test the baseline model
    print("Testing Naive Baseline Predictor")
    print("=" * 70)
    
    # Create dummy data
    torch.manual_seed(42)
    num_samples = 10
    seq_len = 100
    
    # Create synthetic time series with some pattern
    t = torch.linspace(0, 10, seq_len)
    data = torch.sin(t).unsqueeze(0).repeat(num_samples, 1) + torch.randn(num_samples, seq_len) * 0.1
    
    # Create mask with some missing data
    mask = torch.ones(num_samples, seq_len)
    mask[:, ::10] = 0  # Remove every 10th timestep
    
    print(f"Data shape: {data.shape}")
    print(f"Valid data points: {mask.sum().item()} / {mask.numel()}")
    
    # Test naive predictor
    model = NaivePredictor()
    print(f"\nNaive Predictor - Trainable parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create temporal pairs
    pairs = create_temporal_pairs(data, mask)
    print(f"\nTemporal pairs created: {pairs['valid_pairs']}")
    
    # Make predictions
    predictions, _ = model.predict_next_step(pairs['X'].squeeze(-1))
    
    # Evaluate
    metrics = evaluate_predictions(predictions, pairs['Y'].squeeze(-1), 
                                   metrics=['mse', 'mae', 'rmse', 'correlation'])
    
    print("\nNaive Predictor Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")
    
    # Compare baselines
    print("\n" + "=" * 70)
    print("Baseline Comparison")
    print("=" * 70)
    comparison, _ = compare_baselines(data, mask)
    
    for baseline_name, metrics in comparison.items():
        print(f"\n{baseline_name.upper()} Baseline:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.6f}")


"""
Test baseline model on real calcium imaging data
Evaluate naive predictor performance and create visualizations
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import NaivePredictor, create_temporal_pairs, evaluate_predictions, compare_baselines
from data.functional.functional_loader import load_functional_data, dataframe_to_tensors
from data.connectomes.connectome_loader import load_connectome


def test_baseline_on_real_data():
    """Load real data and test baseline predictor"""
    import time
    
    print("=" * 70)
    print("BASELINE MODEL EVALUATION ON CALCIUM IMAGING DATA")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    t0 = time.time()
    connectome = load_connectome(verbose=False)
    print(f"   Connectome loaded in {time.time()-t0:.2f}s")
    t0 = time.time()
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Functional data loaded in {time.time()-t0:.2f}s")
    
    print(f"   Loaded {len(df)} samples from {df['worm'].nunique()} worms")
    print(f"   Neurons: {df['neuron'].nunique()}")
    
    # Convert to tensors
    print("\n2. Converting to tensors...")
    t0 = time.time()
    tensors = dataframe_to_tensors(df, verbose=True)
    print(f"   Converted to tensors in {time.time()-t0:.2f}s")
    
    data = tensors['data']  # [num_samples, max_seq_len]
    mask = tensors['mask']
    
    # Use full dataset for robust evaluation
    # Full dataset: ~6774 samples x variable timesteps
    print(f"   Using full dataset: {data.shape}")
    print(f"   Valid data points: {mask.sum().item():.0f} / {mask.numel()}")
    print(f"   Average sequence length: {tensors['lengths'].float().mean():.1f}")
    print(f"   Max sequence length: {tensors['lengths'].max().item()}")
    
    # Create temporal pairs
    print("\n3. Creating temporal (t, t+1) pairs...")
    t0 = time.time()
    pairs = create_temporal_pairs(data, mask)
    print(f"   Valid pairs created: {pairs['valid_pairs']:,} in {time.time()-t0:.2f}s")
    
    # Initialize baseline model
    print("\n4. Evaluating Naive Baseline (X̂(t+1) = X(t))...")
    model = NaivePredictor()
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Make predictions
    X = pairs['X'].squeeze(-1)
    Y = pairs['Y'].squeeze(-1)
    predictions, _ = model.predict_next_step(X)
    
    # Evaluate
    metrics = evaluate_predictions(predictions, Y, 
                                   metrics=['mse', 'mae', 'rmse', 'correlation', 'r2'])
    
    print("\n   Naive Predictor Performance:")
    print(f"      MSE:  {metrics['mse']:.6f}")
    print(f"      MAE:  {metrics['mae']:.6f}")
    print(f"      RMSE: {metrics['rmse']:.6f}")
    print(f"      R²:   {metrics['r2']:.6f}")
    print(f"      Corr: {metrics['correlation']:.6f}")
    
    # Compare with other baselines
    print("\n5. Comparing with other baseline strategies...")
    t0 = time.time()
    comparison, _ = compare_baselines(data, mask)
    print(f"   Comparison done in {time.time()-t0:.2f}s")
    
    print("\n   Baseline Comparison (MSE):")
    for baseline_name, metrics in sorted(comparison.items(), key=lambda x: x[1]['mse']):
        print(f"      {baseline_name.capitalize():10s}: {metrics['mse']:.6f}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    plot_baseline_comparison(comparison, output_dir="plots")
    
    print("\n" + "=" * 70)
    print("✓ BASELINE EVALUATION COMPLETE!")
    print("=" * 70)
    print("   Check plots/ directory for:")
    print("      - baseline_comparison.png")
    
    return model, metrics, comparison


def plot_prediction_examples(X, Y, predictions, num_examples=6, output_dir="plots"):
    """Plot example predictions vs ground truth"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Randomly select examples
    indices = np.random.choice(len(X), size=min(num_examples, len(X)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        x_val = X[idx].item()
        y_true = Y[idx].item()
        y_pred = predictions[idx].item()
        error = abs(y_pred - y_true)
        
        # Plot points
        ax.scatter([0], [x_val], color='blue', s=100, label='X(t)', zorder=3)
        ax.scatter([1], [y_true], color='green', s=100, label='X(t+1) true', zorder=3)
        ax.scatter([1], [y_pred], color='red', marker='x', s=150, 
                  label='X(t+1) pred', zorder=3)
        
        # Draw prediction line
        ax.plot([0, 1], [x_val, y_pred], 'r--', alpha=0.5, linewidth=2)
        ax.plot([0, 1], [x_val, y_true], 'g-', alpha=0.3, linewidth=1)
        
        # Draw error
        if y_pred != y_true:
            ax.vlines(1, y_pred, y_true, colors='orange', linewidth=2, alpha=0.7)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['t', 't+1'])
        ax.set_ylabel('Calcium Activity (ΔF/F)', fontsize=9)
        ax.set_title(f'Example {idx}\nError: {error:.4f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('Naive Baseline: Prediction Examples\n(Predicting X(t+1) = X(t))', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / "baseline_prediction_examples.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"      Saved examples to: {output_file}")
    plt.close()


def plot_error_distribution(predictions, targets, output_dir="plots"):
    """Plot distribution of prediction errors"""
    output_path = Path(output_dir)
    
    errors = (predictions - targets).numpy()
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Error histogram
    axes[0].hist(errors, bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel('Prediction Error (Predicted - True)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Absolute error histogram
    axes[1].hist(abs_errors, bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.median(abs_errors), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(abs_errors):.4f}')
    axes[1].set_xlabel('Absolute Prediction Error', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Error vs prediction magnitude
    pred_mag = predictions.numpy()
    axes[2].scatter(pred_mag, abs_errors, alpha=0.3, s=10, color='#45B7D1')
    axes[2].set_xlabel('Predicted Value', fontsize=11)
    axes[2].set_ylabel('Absolute Error', fontsize=11)
    axes[2].set_title('Error vs Prediction Magnitude', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Naive Baseline: Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / "baseline_error_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"      Saved error analysis to: {output_file}")
    plt.close()


def plot_baseline_comparison(comparison, output_dir="plots"):
    """Compare different baseline strategies"""
    output_path = Path(output_dir)
    
    baselines = list(comparison.keys())
    metrics = ['mse', 'mae', 'rmse', 'correlation']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = {'naive': '#4ECDC4', 'random': '#FF6B6B'}
    
    for idx, metric in enumerate(metrics):
        values = [comparison[b][metric] for b in baselines]
        bars = axes[idx].bar(baselines, values, 
                            color=[colors.get(b, '#95A5A6') for b in baselines],
                            alpha=0.7, edgecolor='black', linewidth=2)
        
        axes[idx].set_ylabel(metric.upper(), fontsize=12)
        axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.5f}',
                          ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Set x-tick labels
        axes[idx].set_xticks(range(len(baselines)))
        axes[idx].set_xticklabels(baselines, fontsize=11)
    
    plt.suptitle('Baseline Strategy Comparison - Leifer2023 Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / "baseline_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"      Saved comparison to: {output_file}")
    plt.close()


def plot_prediction_scatter_unused(predictions, targets, output_dir="plots"):
    """Scatter plot of predictions vs targets"""
    output_path = Path(output_dir)
    
    pred_np = predictions.numpy()
    target_np = targets.numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(target_np, pred_np, alpha=0.3, s=20, color='#4ECDC4', label='Predictions')
    
    # Perfect prediction line
    min_val = min(target_np.min(), pred_np.min())
    max_val = max(target_np.max(), pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
           label='Perfect Prediction', zorder=5)
    
    # Calculate R²
    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Correlation
    correlation = np.corrcoef(target_np, pred_np)[0, 1]
    
    ax.set_xlabel('True X(t+1)', fontsize=12)
    ax.set_ylabel('Predicted X(t+1)', fontsize=12)
    ax.set_title(f'Naive Baseline: Predictions vs Ground Truth\nR² = {r2:.4f}, Correlation = {correlation:.4f}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    output_file = output_path / "baseline_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"      Saved scatter plot to: {output_file}")
    plt.close()


if __name__ == "__main__":
    test_baseline_on_real_data()


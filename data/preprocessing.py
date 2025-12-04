"""
Data Preprocessing for Graph-Temporal Dataset
Transforms functional calcium data + connectome into PyTorch Geometric format
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset
from pathlib import Path
import hashlib
import json
from scipy import interpolate
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Cache directory
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def align_worm_timesteps(worm_df, neuron_list, time_step=0.333):
    """
    Align all neurons for a single worm to common timesteps.
    
    Args:
        worm_df: DataFrame subset for one worm
        neuron_list: List of all neuron names (from connectome)
        time_step: Target time step in seconds (default 0.333s from preprocessing)
    
    Returns:
        matrix: [T, N] aligned calcium data
        mask: [T, N] mask (1=valid, 0=missing)
        timestamps: [T] common time grid
    """
    # Find the common time range across all neurons in this worm
    min_times = []
    max_times = []
    
    for _, row in worm_df.iterrows():
        times = np.array(row['original_time_in_seconds'])
        if len(times) > 0:
            min_times.append(times.min())
            max_times.append(times.max())
    
    if len(min_times) == 0:
        return None, None, None
    
    # Use the intersection of time ranges
    t_start = max(min_times)
    t_end = min(max_times)
    
    if t_end <= t_start:
        return None, None, None
    
    # Create common time grid
    num_timesteps = int((t_end - t_start) / time_step) + 1
    timestamps = np.linspace(t_start, t_end, num_timesteps)
    
    # Initialize output matrices
    N = len(neuron_list)
    T = len(timestamps)
    matrix = np.zeros((T, N), dtype=np.float32)
    mask = np.zeros((T, N), dtype=np.float32)
    
    # Create neuron to index mapping
    neuron_to_idx = {n: i for i, n in enumerate(neuron_list)}
    
    # Fill in each neuron's data
    for _, row in worm_df.iterrows():
        neuron = row['neuron']
        if neuron not in neuron_to_idx:
            continue
        
        idx = neuron_to_idx[neuron]
        times = np.array(row['original_time_in_seconds'])
        values = np.array(row['original_calcium_data'])
        
        if len(times) < 2:
            continue
        
        # Interpolate to common time grid
        try:
            f = interpolate.interp1d(times, values, kind='linear', 
                                      bounds_error=False, fill_value=np.nan)
            interpolated = f(timestamps)
            
            # Fill valid values
            valid_mask = ~np.isnan(interpolated)
            matrix[:, idx] = np.nan_to_num(interpolated, nan=0.0)
            mask[:, idx] = valid_mask.astype(np.float32)
        except Exception:
            # If interpolation fails, skip this neuron
            continue
    
    return matrix, mask, timestamps


def create_graph_temporal_dataset(df, connectome, max_worms=None, window_size=1, target_horizon=1, verbose=True):
    """
    Create graph-temporal dataset from functional data and connectome.
    
    Args:
        df: DataFrame from load_functional_data()
        connectome: PyTorch Geometric graph from load_connectome()
        max_worms: Maximum number of worms to process (None=all)
        window_size: Number of timesteps to use as input (1=single timestep, >1=temporal window)
        target_horizon: How many steps ahead to predict (1=t+1, 10=t+10, etc.)
        verbose: Print progress
    
    Returns:
        list of Data objects, each containing:
            - x: [N, 1] current activation (if window_size=1) or [N, window_size] (if window_size>1)
            - edge_index: [2, E] connectome edges
            - edge_attr: [E, 2] edge attributes (gap junction, chemical synapse weights)
            - mask: [N, 1] valid neurons
            - y: [N, 1] next timestep activation (target)
            - worm_id: worm identifier
            - timestep: timestep index
    """
    # Get neuron list from connectome
    neuron_list = list(connectome.node_label)
    edge_index = connectome.edge_index
    edge_attr = connectome.edge_attr  # [E, 2] - gap and chem weights
    N = len(neuron_list)
    
    if verbose:
        print(f"Creating graph-temporal dataset...")
        print(f"  Connectome neurons: {N}")
        print(f"  Connectome edges: {edge_index.shape[1]}")
    
    # Group by worm
    worms = df['worm'].unique()
    if max_worms is not None:
        worms = worms[:max_worms]
    
    if verbose:
        print(f"  Processing {len(worms)} worms...")
    
    all_data = []
    total_samples = 0
    
    for worm_id in worms:
        worm_df = df[df['worm'] == worm_id]
        
        # Align timesteps for this worm
        matrix, mask, timestamps = align_worm_timesteps(worm_df, neuron_list)
        
        if matrix is None or matrix.shape[0] < 2:
            continue
        
        T = matrix.shape[0]
        
        # Create (X_window, X_{t+target_horizon}) pairs
        for t in range(window_size - 1, T - target_horizon):
            # Create temporal window: [t-window_size+1, t] (inclusive)
            if window_size == 1:
                x_window = torch.tensor(matrix[t], dtype=torch.float32).unsqueeze(-1)  # [N, 1]
                mask_window = torch.tensor(mask[t], dtype=torch.float32).unsqueeze(-1)  # [N, 1]
            else:
                # Extract window: [t-window_size+1, t+1) -> [window_size, N]
                window_matrix = matrix[t-window_size+1:t+1]  # [window_size, N]
                window_mask = mask[t-window_size+1:t+1]  # [window_size, N]
                
                # Transpose to [N, window_size]
                x_window = torch.tensor(window_matrix.T, dtype=torch.float32)  # [N, window_size]
                mask_window = torch.tensor(window_mask.T, dtype=torch.float32)  # [N, window_size]
                
                # Combined mask: all timesteps in window must be valid
                mask_window = mask_window.prod(dim=1, keepdim=True)  # [N, 1]
            
            y_t = torch.tensor(matrix[t + target_horizon], dtype=torch.float32).unsqueeze(-1)  # [N, 1]
            mask_next = torch.tensor(mask[t + target_horizon], dtype=torch.float32).unsqueeze(-1)  # [N, 1]
            
            # Combined mask: window and next timestep must be valid
            combined_mask = mask_window * mask_next
            
            # Only include if we have at least some valid neurons
            if combined_mask.sum() < 5:
                continue
            
            data = Data(
                x=x_window,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask=combined_mask,
                y=y_t,
                worm_id=worm_id,
                timestep=t,
                num_nodes=N,
                window_size=window_size,
                target_horizon=target_horizon
            )
            all_data.append(data)
            total_samples += 1
    
    if verbose:
        print(f"  Created {total_samples} samples from {len(worms)} worms")
        if total_samples > 0:
            avg_valid = np.mean([d.mask.sum().item() for d in all_data])
            print(f"  Average valid neurons per sample: {avg_valid:.1f}")
    
    return all_data


class WormGraphDataset(Dataset):
    """PyTorch Dataset wrapper for graph-temporal data."""
    
    def __init__(self, data_list):
        """
        Args:
            data_list: List of PyG Data objects from create_graph_temporal_dataset()
        """
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def split_by_worm(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Split dataset by worm ID (not by sample) to avoid data leakage.
        Ensures at least 1 worm in each split when possible.
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        np.random.seed(seed)
        
        # Get unique worms
        worms = list(set(d.worm_id for d in self.data_list))
        np.random.shuffle(worms)
        
        n_worms = len(worms)
        
        # Handle small datasets: ensure at least 1 worm in val/test if possible
        if n_worms >= 3:
            # At least 1 worm in each split
            n_val = max(1, int(n_worms * val_ratio))
            n_test = max(1, int(n_worms * (1 - train_ratio - val_ratio)))
            n_train = n_worms - n_val - n_test
        elif n_worms == 2:
            # 1 train, 0 val, 1 test
            n_train, n_val, n_test = 1, 0, 1
        else:
            # Only 1 worm: use all for training
            n_train, n_val, n_test = 1, 0, 0
        
        train_worms = set(worms[:n_train])
        val_worms = set(worms[n_train:n_train + n_val])
        test_worms = set(worms[n_train + n_val:])
        
        # Split data
        train_data = [d for d in self.data_list if d.worm_id in train_worms]
        val_data = [d for d in self.data_list if d.worm_id in val_worms]
        test_data = [d for d in self.data_list if d.worm_id in test_worms]
        
        return WormGraphDataset(train_data), WormGraphDataset(val_data), WormGraphDataset(test_data)


def get_cache_path(source_datasets, max_worms, window_size, target_horizon):
    """Generate cache path for preprocessed dataset."""
    key_parts = {
        'sources': sorted(source_datasets) if source_datasets else [],
        'max_worms': max_worms,
        'window_size': window_size,
        'target_horizon': target_horizon
    }
    key_str = json.dumps(key_parts, sort_keys=True)
    cache_key = hashlib.md5(key_str.encode()).hexdigest()[:12]
    return CACHE_DIR / f"graph_windows_{cache_key}.pt"


def load_or_create_dataset(df, connectome, max_worms=None, window_size=1, target_horizon=1, use_cache=True, verbose=True):
    """
    Load cached dataset or create new one.
    
    Args:
        df: DataFrame from load_functional_data()
        connectome: PyTorch Geometric graph
        max_worms: Maximum worms to process
        window_size: Temporal window size (1=single timestep, >1=window)
        target_horizon: Prediction horizon (1=t+1, 10=t+10, etc.)
        use_cache: Whether to use caching
        verbose: Print progress
    
    Returns:
        WormGraphDataset
    """
    source_datasets = df['source_dataset'].unique().tolist()
    cache_path = get_cache_path(source_datasets, max_worms, window_size, target_horizon)
    
    if use_cache and cache_path.exists():
        if verbose:
            print(f"Loading cached dataset: {cache_path.name}")
        data_list = torch.load(cache_path, weights_only=False)
        return WormGraphDataset(data_list)
    
    # Create dataset
    data_list = create_graph_temporal_dataset(df, connectome, max_worms, window_size, target_horizon, verbose)
    
    # Cache
    if use_cache and len(data_list) > 0:
        if verbose:
            print(f"Caching dataset: {cache_path.name}")
        torch.save(data_list, cache_path)
    
    return WormGraphDataset(data_list)


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing preprocessing module...")
    print("=" * 70)
    
    from data.connectomes.connectome_loader import load_connectome
    from data.functional.functional_loader import load_functional_data
    
    # Load data
    print("\n1. Loading connectome...")
    connectome = load_connectome(verbose=False)
    print(f"   Neurons: {connectome.num_nodes}, Edges: {connectome.num_edges}")
    
    print("\n2. Loading functional data...")
    df = load_functional_data(connectome=connectome, verbose=False)
    print(f"   Samples: {len(df)}, Worms: {df['worm'].nunique()}")
    
    print("\n3. Creating graph-temporal dataset (5 worms for test)...")
    dataset = load_or_create_dataset(df, connectome, max_worms=5, verbose=True)
    
    print(f"\n4. Dataset info:")
    print(f"   Total samples: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   Sample x shape: {sample.x.shape}")
        print(f"   Sample y shape: {sample.y.shape}")
        print(f"   Sample mask sum: {sample.mask.sum().item():.0f}")
        print(f"   Edge index shape: {sample.edge_index.shape}")
    
    print("\n5. Testing train/val/test split...")
    train_ds, val_ds, test_ds = dataset.split_by_worm()
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing test complete!")


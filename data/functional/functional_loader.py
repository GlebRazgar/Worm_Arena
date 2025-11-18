"""
Functional Data Loader for C. elegans
Config-driven loader for calcium imaging time series data with automatic caching
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
import hashlib
import json

# Add configs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import get_config

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_key(source_datasets, worms, connectome_neurons):
    """Generate unique cache key based on filter parameters"""
    key_parts = {
        'sources': sorted(source_datasets) if source_datasets else [],
        'worms': sorted(worms) if worms else None,
        'neurons': sorted(list(connectome_neurons)) if connectome_neurons else None
    }
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _get_cache_paths(cache_key):
    """Get cache file paths for given key"""
    return {
        'parquet': CACHE_DIR / f"functional_{cache_key}.parquet",
        'tensors': CACHE_DIR / f"functional_{cache_key}_tensors.pt",
        'metadata': CACHE_DIR / f"functional_{cache_key}_meta.json"
    }


def load_functional_data(source_datasets=None, worms=None, connectome=None, verbose=True, use_cache=True):
    """
    Load calcium imaging data from HuggingFace dataset with automatic caching.
    
    Args:
        source_datasets: List of source dataset names to load. If None, uses config.
                        Options: 'Venkatachalam2024', 'Flavell2023', 'Leifer2023', etc.
        worms: List of specific worm IDs to load. If None, loads all worms (or uses config).
        connectome: PyTorch Geometric graph object for neuron filtering. If None, uses config setting.
        verbose: Whether to print loading information
        use_cache: Whether to use cached data if available
    
    Returns:
        pd.DataFrame: Filtered calcium imaging data with columns:
            - source_dataset: Origin dataset name
            - raw_data_file: Original file name
            - worm: Worm identifier
            - neuron: Neuron name
            - slot: Recording slot
            - is_labeled_neuron: Whether neuron is confidently identified
            - smooth_method: Smoothing method applied
            - interpolate_method: Interpolation method used
            - normalization_method: Normalization applied
            - original_calcium_data: Time series of calcium fluorescence
            - original_time_in_seconds: Time points in seconds
    """
    # Load config
    config = get_config()
    
    # Use config if parameters not specified
    if source_datasets is None:
        source_datasets = config.functional_sources
        if not source_datasets:
            raise ValueError("No source datasets specified. Set in configs/data.yaml or pass as argument.")
    
    if worms is None:
        worms = config.functional_worms
    
    match_connectome_flag = config.match_connectome if connectome is None else (connectome is not None)
    
    # Get connectome neurons for cache key
    connectome_neurons = set(connectome.node_label) if (match_connectome_flag and connectome is not None) else None
    
    # Check cache
    if use_cache:
        cache_key = _get_cache_key(source_datasets, worms, connectome_neurons)
        cache_paths = _get_cache_paths(cache_key)
        
        if cache_paths['parquet'].exists():
            if verbose:
                print(f"Loading from cache: {cache_paths['parquet'].name}")
            df_filtered = pd.read_parquet(cache_paths['parquet'])
            
            if verbose:
                print(f"\nCached dataset:")
                print(f"  Total samples: {len(df_filtered)}")
                print(f"  Unique neurons: {df_filtered['neuron'].nunique()}")
                print(f"  Unique worms: {df_filtered['worm'].nunique()}")
                print(f"  Sources: {df_filtered['source_dataset'].unique().tolist()}")
            
            return df_filtered
    
    if verbose:
        print(f"Loading functional data from {len(source_datasets)} source(s)...")
        print(f"  Sources: {source_datasets}")
    
    # Load dataset from HuggingFace
    ds = load_dataset("qsimeon/celegans_neural_data", split='train')
    
    # OPTIMIZATION: Filter BEFORE converting to pandas (much faster!)
    if verbose:
        print(f"  Total samples in dataset: {len(ds)}")
    
    # Filter by source datasets using HuggingFace's filter method
    def filter_by_source(example):
        return example['source_dataset'] in source_datasets
    
    ds_filtered = ds.filter(filter_by_source, num_proc=1)
    
    if verbose:
        print(f"  After source filter: {len(ds_filtered)} samples")
    
    # Now convert to pandas (much smaller dataset)
    df_filtered = pd.DataFrame(ds_filtered)
    
    # Filter by worms if specified
    if worms is not None:
        df_filtered = df_filtered[df_filtered['worm'].isin(worms)]
        if verbose:
            print(f"  After worm filter: {len(df_filtered)} samples")
    
    # Filter by connectome neurons if requested
    if match_connectome_flag and connectome is not None:
        connectome_neurons = set(connectome.node_label)
        df_filtered = df_filtered[df_filtered['neuron'].isin(connectome_neurons)]
        if verbose:
            print(f"  After connectome match: {len(df_filtered)} samples")
            print(f"  Matched {df_filtered['neuron'].nunique()} neurons from connectome")
    
    if verbose:
        print(f"\nFinal dataset:")
        print(f"  Total samples: {len(df_filtered)}")
        print(f"  Unique neurons: {df_filtered['neuron'].nunique()}")
        print(f"  Unique worms: {df_filtered['worm'].nunique()}")
        print(f"  Labeled neurons: {df_filtered['is_labeled_neuron'].sum()}")
    
    # Save to cache
    if use_cache:
        cache_key = _get_cache_key(source_datasets, worms, connectome_neurons)
        cache_paths = _get_cache_paths(cache_key)
        
        if verbose:
            print(f"\nSaving to cache: {cache_paths['parquet'].name}")
        
        df_filtered.to_parquet(cache_paths['parquet'], compression='snappy')
        
        # Save metadata
        metadata = {
            'source_datasets': source_datasets,
            'worms': worms,
            'num_samples': len(df_filtered),
            'num_neurons': int(df_filtered['neuron'].nunique()),
            'num_worms': int(df_filtered['worm'].nunique()),
            'has_connectome_filter': connectome_neurons is not None
        }
        with open(cache_paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return df_filtered


def dataframe_to_tensors(df, sequence_column='original_calcium_data', time_column='original_time_in_seconds', 
                         use_cache=True, cache_key=None, verbose=False):
    """
    Convert functional DataFrame to PyTorch tensors for GNN training with caching.
    
    Args:
        df: DataFrame from load_functional_data()
        sequence_column: Column name containing calcium time series
        time_column: Column name containing time points
        use_cache: Whether to use cached tensors if available
        cache_key: Optional cache key (auto-generated if None)
        verbose: Print loading information
    
    Returns:
        dict containing:
            - data: torch.Tensor [num_samples, max_seq_len] - Calcium traces (padded)
            - mask: torch.Tensor [num_samples, max_seq_len] - Padding mask (1=valid, 0=padding)
            - time: torch.Tensor [num_samples, max_seq_len] - Time points (padded)
            - neurons: list of str - Neuron names for each sample
            - worms: list of str - Worm IDs for each sample
            - source_datasets: list of str - Source dataset names
            - lengths: torch.Tensor [num_samples] - Actual sequence lengths
            - neuron_to_idx: dict - Mapping from neuron name to unique index
            - unique_neurons: list of str - Sorted unique neuron names
    """
    # Generate cache key if not provided
    if cache_key is None and use_cache:
        # Use simple hash of dataframe shape and content
        source_datasets = df['source_dataset'].unique().tolist()
        worms = df['worm'].unique().tolist() if len(df['worm'].unique()) < 200 else None
        neurons = sorted(df['neuron'].unique().tolist())
        cache_key = _get_cache_key(source_datasets, worms, set(neurons))
    
    # Check tensor cache
    if use_cache and cache_key:
        cache_paths = _get_cache_paths(cache_key)
        if cache_paths['tensors'].exists():
            if verbose:
                print(f"Loading tensors from cache: {cache_paths['tensors'].name}")
            return torch.load(cache_paths['tensors'])
    
    if verbose:
        print("Converting DataFrame to tensors...")
    if len(df) == 0:
        raise ValueError("Cannot convert empty DataFrame to tensors")
    
    # Extract sequences
    sequences = df[sequence_column].tolist()
    time_sequences = df[time_column].tolist() if time_column in df.columns else None
    
    # Get metadata
    neurons = df['neuron'].tolist()
    worms = df['worm'].tolist()
    source_datasets = df['source_dataset'].tolist()
    
    # Find max sequence length
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    
    # Create padded arrays
    num_samples = len(sequences)
    data_array = np.zeros((num_samples, max_len), dtype=np.float32)
    mask_array = np.zeros((num_samples, max_len), dtype=np.float32)
    time_array = np.zeros((num_samples, max_len), dtype=np.float32) if time_sequences else None
    
    # Fill arrays
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        data_array[i, :length] = seq
        mask_array[i, :length] = 1.0
        if time_sequences and time_sequences[i] is not None:
            time_array[i, :length] = time_sequences[i]
    
    # Create neuron to index mapping
    unique_neurons = sorted(df['neuron'].unique())
    neuron_to_idx = {neuron: idx for idx, neuron in enumerate(unique_neurons)}
    
    # Convert to tensors
    result = {
        'data': torch.from_numpy(data_array),
        'mask': torch.from_numpy(mask_array),
        'neurons': neurons,
        'worms': worms,
        'source_datasets': source_datasets,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'neuron_to_idx': neuron_to_idx,
        'unique_neurons': unique_neurons
    }
    
    if time_array is not None:
        result['time'] = torch.from_numpy(time_array)
    
    # Save tensor cache
    if use_cache and cache_key:
        cache_paths = _get_cache_paths(cache_key)
        if verbose:
            print(f"Saving tensors to cache: {cache_paths['tensors'].name}")
        torch.save(result, cache_paths['tensors'])
    
    return result


def get_neuron_traces(df, neuron_name):
    """
    Extract all calcium traces for a specific neuron.
    
    Args:
        df: DataFrame from load_functional_data()
        neuron_name: Name of neuron to extract
    
    Returns:
        pd.DataFrame: Subset containing only specified neuron
    """
    return df[df['neuron'] == neuron_name].copy()


def get_worm_data(df, worm_id):
    """
    Extract all data for a specific worm.
    
    Args:
        df: DataFrame from load_functional_data()
        worm_id: Worm identifier
    
    Returns:
        pd.DataFrame: Subset containing only specified worm
    """
    return df[df['worm'] == worm_id].copy()


if __name__ == "__main__":
    # Test the loader
    print("Testing functional data loader:")
    print("=" * 70)
    
    # Load from config
    df = load_functional_data()
    
    print("\nDataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    print("\nSample statistics:")
    print(f"  Source datasets: {df['source_dataset'].unique().tolist()}")
    print(f"  Unique neurons: {df['neuron'].nunique()}")
    print(f"  Unique worms: {df['worm'].nunique()}")
    
    # Test tensor conversion
    print("\nConverting to tensors...")
    tensors = dataframe_to_tensors(df.head(100))  # Test with first 100 samples
    print(f"  Data shape: {tensors['data'].shape}")
    print(f"  Mask shape: {tensors['mask'].shape}")
    print(f"  Unique neurons in subset: {len(tensors['unique_neurons'])}")
    print(f"  Average sequence length: {tensors['lengths'].float().mean():.1f}")


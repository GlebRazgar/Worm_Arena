# Functional Data Cache

This directory contains cached functional calcium imaging data for fast loading.

## Cache Structure

Each cache consists of three files:

1. **`functional_<key>.parquet`** - Pandas DataFrame (compressed)
   - Fast to load (~1-2s)
   - Use for exploration, filtering, visualization
   
2. **`functional_<key>_tensors.pt`** - PyTorch tensors
   - Pre-converted for training
   
3. **`functional_<key>_meta.json`** - Metadata
   - Source datasets, sample counts, neuron info

## Cache Key

The cache key is automatically generated based on:
- Source datasets (e.g., Leifer2023)
- Worm filters (if specified)
- Connectome neuron filtering (if enabled)

This ensures the same filter parameters always use the same cache.

## Automatic Caching

The `load_functional_data()` function automatically:
1. Checks if cached data exists for your config
2. Loads from cache if available (~1s)
3. Otherwise downloads from HuggingFace and creates cache (~2-3 minutes first time)
4. Subsequent loads are instant


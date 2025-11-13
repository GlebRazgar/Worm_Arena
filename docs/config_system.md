# Configuration System Documentation

## Overview

Worm Arena uses a YAML-based configuration system for easy experimentation and modular design. The config system eliminates hardcoded values and makes it simple to switch between different connectome datasets.

## Architecture

### Config Files

```
configs/
├── config.py           # Configuration loader (singleton pattern)
├── data.yaml          # Data configuration
├── model.yaml         # Model hyperparameters
└── train.yaml         # Training configuration
```

### Config Loader (`configs/config.py`)

```python
from configs.config import get_config

# Get config singleton
cfg = get_config()

# Access connectome dataset
print(cfg.connectome)  # e.g., 'cook2019'

# Access all configs
print(cfg.data)    # Data config dict
print(cfg.model)   # Model config dict
print(cfg.train)   # Training config dict
```

## Data Configuration

### `configs/data.yaml`

```yaml
# Connectome dataset to load
dataset: cook2019

# Functional calcium imaging data
functional:
  source_datasets:
    - "Venkatachalam2024"
    - "Flavell2023"
    - "Leifer2023"
  worms: null  # null = all worms
  match_connectome: true  # Auto-filter for connectome neurons

# Available connectomes: 
#   cook2019, witvliet2020_7, witvliet2020_8, 
#   white1986_whole, white1986_n2u, white1986_jsh

# Available functional sources:
#   Venkatachalam2024, Flavell2023, Leifer2023, Kato2015,
#   Nichols2017, Nguyen2016, Schrodel2013, Skora2018,
#   Uzel2022, Yemini2021, Kaplan2020, Atanas2023
```

### Connectome Loader Integration

The connectome loader automatically reads from config:

```python
from data.connectomes.connectome_loader import load_connectome

# Loads dataset from configs/data.yaml
graph = load_connectome()

# Or override for specific use
graph = load_connectome('witvliet2020_7')

# Graph includes connectome name
print(graph.connectome_name)  # 'cook2019'
```

### Functional Data Loader Integration

The functional data loader reads from config and auto-matches neurons:

```python
from data.functional.functional_loader import load_functional_data, dataframe_to_tensors

# Load functional data (uses config)
df = load_functional_data(connectome=graph)

# Or override settings
df = load_functional_data(
    source_datasets=['Flavell2023'],
    worms=['worm9', 'worm12'],
    connectome=graph
)

# Convert to PyTorch tensors
tensors = dataframe_to_tensors(df)
print(tensors['data'].shape)  # [num_samples, seq_len]
```

## Visualization Integration

Visualizations automatically include the connectome name in:
- **Plot titles**: "C. elegans Connectome - COOK2019 (3D)"
- **Filenames**: `cook2019_2d.png`, `cook2019_3d.png`, etc.

```python
from wrom.visualisation import draw_connectome

# Auto-generates filename: {connectome_name}_2d.png
draw_connectome(graph, plot_3d=False, output_dir="plots")

# Custom filename still works
draw_connectome(graph, filename="custom.png")
```

## Switching Connectomes

### Method 1: Edit Config (Recommended)

Edit `configs/data.yaml`:
```yaml
dataset: witvliet2020_7
```

Then run any script - it will automatically use the new dataset:
```bash
python tests/test_connectome.py
```

### Method 2: Override in Code

```python
# Temporary override without changing config
graph = load_connectome('white1986_whole')
```

## Benefits

### ✅ Modularity
- Single source of truth for dataset selection
- No hardcoded dataset names in code
- Easy to switch between experiments

### ✅ Clarity
- Clear what dataset is being used
- Automatic naming prevents confusion
- Plots clearly labeled with dataset name

### ✅ Flexibility
- Can still override config in code when needed
- Supports multiple connectomes easily
- Future-proof for adding more datasets

### ✅ Organization
- All config in one place
- Auto-generated filenames prevent conflicts
- Easy to track which results came from which dataset

## Available Connectomes

| Dataset | Neurons | Edges | Description |
|---------|---------|-------|-------------|
| `cook2019` | 236 | 4,085 | Cook et al. 2019 |
| `witvliet2020_7` | 180 | 2,364 | Witvliet et al. 2020 dataset 7 |
| `witvliet2020_8` | - | - | Witvliet et al. 2020 dataset 8 |
| `white1986_whole` | - | - | White et al. 1986 complete |
| `white1986_n2u` | - | - | White et al. 1986 N2U |
| `white1986_jsh` | - | - | White et al. 1986 JSH |

## Example Workflow

```python
# 1. Set dataset in config
# configs/data.yaml: dataset: cook2019

# 2. Load data
from data.connectomes.connectome_loader import load_connectome
graph = load_connectome()  # Reads from config

# 3. Visualize
from wrom.visualisation import draw_connectome
draw_connectome(graph)  # Creates cook2019_2d.png

# 4. Switch dataset
# configs/data.yaml: dataset: witvliet2020_7

# 5. Re-run - automatically uses new dataset
graph = load_connectome()  # Now loads witvliet2020_7
draw_connectome(graph)     # Creates witvliet2020_7_2d.png
```

## Complete Example Workflow

```python
from configs.config import get_config
from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data, dataframe_to_tensors

# Load config
cfg = get_config()
print(f"Connectome: {cfg.connectome}")
print(f"Functional sources: {cfg.functional_sources}")

# Load structure
graph = load_connectome()

# Load dynamics
df = load_functional_data(connectome=graph)

# Prepare for training
tensors = dataframe_to_tensors(df)

print(f"Ready for GNN training:")
print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
print(f"  Data: {tensors['data'].shape}")
print(f"  Matched neurons: {len(tensors['unique_neurons'])}")
```

## Future Extensions

Additional preprocessing options:

```yaml
functional:
  source_datasets: [...]
  preprocessing:
    min_sequence_length: 100
    max_sequence_length: 5000
    filter_unlabeled: true
    resample_rate: 0.5
```

```yaml
# Future model config
model_type: gat
hidden_dim: 128
num_layers: 4
heads: 8
dropout: 0.2
```

```yaml
# Future training config
epochs: 200
batch_size: 32
optimizer: adamw
learning_rate: 1e-3
scheduler: cosine
```

## Best Practices

1. **Use config for experiments**: Don't hardcode dataset names
2. **Document changes**: Update `CHANGELOG.md` when switching datasets
3. **Version control**: Commit config changes with results
4. **Override sparingly**: Only override in code for temporary tests
5. **Name conventions**: Let auto-naming handle filenames

## Migration from Old Code

### Before (hardcoded)
```python
graph = load_cook2019_connectome()
draw_connectome(graph, filename="connectome_2d.png")
```

### After (config-driven)
```python
graph = load_connectome()  # Reads from config
draw_connectome(graph)     # Auto-names: {dataset}_2d.png
```


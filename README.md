# Worm Arena / WromFunctionator  / Struc2Func

A C. elegans simulation framework building structure 2 function computational models.


## Goal:

Desired design: Build a GNN that takes the connectome graph structure and uses calcium activity to predict neuronal activation for the next time-step.

## Next Steps
- [ ] Sift through different signal processing approaches. Select one and implement it. (Spike vs gradient learning)
- [ ] Choose the best output metric (MSE? EMSE? explained variance)
- [ ] GCN → GAT. Implement the graph attention network.
- [ ] Compare two GATs: A trained on just funcitonal, with no structural priors and B with structural priors. 
- [ ] Add evaluation metrics
- [ ] Integrate body simulation


## Project Structure

```
Worm_Arena/
├── configs/
│   ├── config.py                    # Configuration loader
│   ├── data.yaml                    # Data configuration (connectome selection)
│   ├── model.yaml                   # Model hyperparameters
│   └── train.yaml                   # Training configuration
├── data/
│   ├── connectomes/
│   │   └── connectome_loader.py    # Config-driven connectome loader
│   ├── functional/
│   │   └── functional_loader.py     # Load calcium imaging data
│   ├── data_download.py             # Data download utilities
│   └── sandbox.py                   # Quick data exploration
├── wrom/
│   ├── connectome.py                # Connectome data structures
│   └── body.py                      # Worm body simulation
├── models/
│   ├── gnn.py                       # GNN architecture
│   └── baseline.py                  # Baseline models
├── src/
│   ├── main.py                      # Main entry point (simple one-liner viz)
│   └── train.py                     # Training loop
├── tests/
│   ├── connectome_vis.py            # Visualize connectome structure
│   ├── functional_vis.py            # Visualize calcium imaging data
│   └── plots/                       # Generated visualizations
└── requirements.txt                 # Python dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login to HuggingFace (required for dataset access):
```bash
hf auth login
```
4. When prompted, paste your token (this is my token, delete later):

## Quick Start

### Configuration

The project uses YAML configs for easy experimentation. Set your connectome dataset in `configs/data.yaml`:

```yaml
# Options: cook2019, witvliet2020_7, white1986_whole, etc.
dataset: cook2019
```

### Load and Visualize Connectome

Run the main script:
```bash
python src/main.py
```

This will:
- Load the connectome specified in `configs/data.yaml`
- Load functional calcium imaging data
- Generate visualizations in `tests/plots/`:
  - `connectome-{dataset}.png` - Connectome structure
  - `functional_..._diverse_traces.png` - Calcium traces from different recordings

## Switching Connectomes

To use a different connectome, simply edit `configs/data.yaml`:

```yaml
# Use Witvliet 2020 dataset instead
dataset: witvliet2020_7
```

Available connectomes:
- `cook2019` - Cook et al. 2019 (236 neurons, 4,085 edges)
- `witvliet2020_7` - Witvliet et al. 2020 dataset 7
- `witvliet2020_8` - Witvliet et al. 2020 dataset 8
- `white1986_whole` - White et al. 1986 complete
- `white1986_n2u` - White et al. 1986 N2U
- `white1986_jsh` - White et al. 1986 JSH

Then run `python src/main.py` and the visualization will automatically use the new dataset.

## Dataset Information

### Connectome Data
**Source**: [HuggingFace - qsimeon/celegans_connectome_data](https://huggingface.co/datasets/qsimeon/celegans_connectome_data)

- **~300 neurons** from C. elegans
- **~7,000 edges** in Cook2019 dataset
- **Edge attributes**: Gap junction weights, chemical synapse weights
- **Node attributes**: 3D spatial positions, neuron type (sensory, inter, motor)
- **Multiple connectome versions**: Cook2019, White1986, Witvliet2020, etc.

### Functional Calcium Imaging Data
**Source**: [HuggingFace - qsimeon/celegans_neural_data](https://huggingface.co/datasets/qsimeon/celegans_neural_data)

- **42,798 time series samples** from calcium imaging
- **358 unique neurons** across recordings
- **12 source datasets** aggregated and standardized
- **Preprocessed**: Resampled (0.333s), smoothed, normalized
- **Labeled neurons**: Quality-controlled neuron identification
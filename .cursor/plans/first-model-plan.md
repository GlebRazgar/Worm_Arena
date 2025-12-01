<!-- 74d393d4-accd-4320-a4d6-e9aa0cea017a ff0f296d-c07a-4ff2-bd81-2a774a7a227f -->
# GNN-LSTM Implementation Plan

## 🎯 What Should the GNN Learn?

Assess if GNN can create a functional connectome that's better than the structural connectome by learning which connections matter for prediction

## Overview

Build a GNN-LSTM hybrid model that uses the C. elegans connectome graph structure as a prior, and temporal calcium activity as input to predict next-timestep neuronal activation. The implementation will be clean and minimal (4 files modified/created), optimized for M4 Max macbook - MPS acceleration.

# The model:

- Prior: Anatomical connectome graph (neurons = nodes, synapse connections = weights of edges).
    - Format:
- Input: Functional data (continuous calcium singal)
    - Format: 
- Learn: Re-weigting the connectome. Infer spatial relationships that dont exist. Functional weights (which connections matter for dynamics)
    - 
- Output: Calcium activity prediction. 
    - Same as input format
- Objective: Discover structure-function relationships in the nervous system

## GNN architecture:

Type: GAT.

Reason: Learns to adjust edge weights via attention. It will take the connectome as a prior, and learn a more correct connectivity from functional data. Incase some links are missing or connection weights dont correspond perfectly to function.

Adjustments: *Add spatial features to GAT*
attention_ij = f(neural_features, edge_attr, distance_ij)
# - edge_attr_ij = [gap_weight, chem_weight] from connectome (prior)
# - attention_ij = learned functional weight (adjustment)
# - Final weight = attention_ij * edge_attr_ij (combines both)



## Graph Properties

From the code:

- ~196-300 neurons, ~4,000-7,000 edges

- Directed graph: chemical synapses are directional (pre→post)

- Bidirectional edges: gap junctions allow information flow both ways

- Edge features: 2D [gap_weight, chem_weight]

- Node features: 3D one-hot [sensory, inter, motor]

- Spatial information: 3D positions available



## Output plots:

1. Calcium activity of each neuorn. 
2. Basdeline models validation loss across datasets.
3. GNNs validation loss across datasets.
4. GNN validation curve throutghout epochs for each individual dataset. 
5. Visualisation of the models outputs as a connectome graph of a worm passing the message, ligting up, lighting down, etc.
6. GPU utilizartion plot on weights and biases live tracked. 

## Files to Modify/Create

### 1. configs/model.yaml (MODIFY - uncomment and extend)

- Uncomment existing GNN template
- Add specific hyperparameters:
- GAT layers: [64, 128, 256] with heads [4, 4, 2]
- LSTM: 256 hidden, 2 layers, 0.2 dropout
- Window size: 50 timesteps, stride: 10
- Edge dimension: 2 (gap junction + chemical)
- Prediction head: MLP [256 -> 128 -> 1]
- Add device config: "mps" (primary), "cpu" (fallback)
- Add data config: batch_size=16, num_workers=4

### 2. models/gnn.py (CREATE - ~400 lines)

Complete GNN-LSTM implementation in single file:

**Part A: Data Preparation (150 lines)**

- `create_graph_temporal_dataset(df, connectome, window_size=50, stride=10)`:
- Group DataFrame by worm_id
- For each worm: pivot neurons to columns (196 neurons from connectome)
- Align all recordings to common timesteps
- Create mask [T, N] for missing neurons
- Extract sliding windows [W, N]
- Return List[Data] with x=[W,N,1], edge_index, edge_attr, mask, y=[N]

- `WormGraphDataset(Dataset)`:
- PyTorch Dataset wrapper
- Handles batching and shuffling
- Caches preprocessed windows to disk

**Part B: Model Architecture (250 lines)**

- `BioGATLayer(MessagePassing)`:
- GAT with edge features (2D: gap junction + chemical weights)
- Multi-head attention mechanism
- Handles directed edges (chemical synapses)

- `SpatioTemporalGNN(nn.Module)`:
- Stack 3 GAT layers: input features -> spatial embeddings
- LSTM layer: temporal aggregation over window
- MLP predictor: final hidden state -> next timestep prediction
- Forward pass: process each timestep through GNN, stack, feed to LSTM
- Return: predictions [batch, num_neurons]

**Key Design Decisions:**

- Use shared LSTM across neurons (not per-neuron) for efficiency
- Masked MSE loss: only compute error on recorded neurons
- Edge dropout during training for robustness

### 3. src/train.py (MODIFY - add ~200 lines)

Currently just a comment, implement full training pipeline:

**Functions to add:**

- `setup_device()`:
- Priority: MPS (M4 Max GPU) > CUDA > CPU
- Check `torch.backends.mps.is_available()`
- Print device info and memory

- `train_gnn_model()`:
- Load config from model.yaml and data.yaml
- Load connectome via existing `load_connectome()`
- Load functional data via existing `load_functional_data()`
- Create graph-temporal dataset via `create_graph_temporal_dataset()`
- Split by worm: 80% train, 10% val, 10% test
- Initialize SpatioTemporalGNN model
- Training loop:
- Batch iteration with progress bars (tqdm)
- Forward pass, masked MSE loss
- Backward pass, optimizer step
- Validation every epoch
- Save checkpoints to models/checkpoints/
- Early stopping based on validation loss

- `evaluate_model(model, loader, device)`:
- Compute metrics: MSE, MAE, RMSE, R², Correlation
- Per-neuron and aggregate statistics
- Return results dict

- `main()`:
- Argparse: --model [gnn|baseline], --resume PATH, --config PATH
- Call train_gnn_model() or baseline training
- Save final model and metrics

**Integration with existing code:**

- Import from `data.connectomes.connectome_loader`
- Import from `data.functional.functional_loader`
- Import from `models.gnn` (new)
- Import from `models.baseline` (for comparison)

### 4. tests/gnn_vis.py (CREATE - ~250 lines)

Evaluation and visualization script:

**Functions:**

- `evaluate_gnn_on_test_set()`:
- Load trained checkpoint
- Run on held-out test worms
- Compute all metrics
- Compare to baseline performance

- `plot_gnn_vs_baseline()`:
- Bar chart: MSE, MAE, R² comparison
- Save to tests/plots/gnn_vs_baseline.png

- `plot_attention_weights(model, sample_data, connectome)`:
- Extract GAT attention weights
- Overlay on connectome visualization
- Highlight important synapses
- Save to tests/plots/gnn_attention.png

- `plot_predictions_over_time(model, test_window)`:
- Select sample window from test set
- Plot predicted vs actual traces for 6 neurons
- Show prediction accuracy over time
- Save to tests/plots/gnn_predictions.png

- `analyze_error_by_neuron_type()`:
- Group neurons by type (sensory/inter/motor)
- Compute per-type MSE
- Statistical comparison
- Save to tests/plots/gnn_error_by_type.png

- `main()`:
- Load best checkpoint from training
- Run all evaluation functions
- Print summary statistics
- Generate all visualization plots

## Implementation Order

1. configs/model.yaml (5 min)

- Uncomment and fill in hyperparameters
- Set device to "mps"

2. models/gnn.py - Data prep first (1-2 hours)

- Implement `create_graph_temporal_dataset()`
- Test on single worm to verify shapes
- Add caching for repeated runs

3. models/gnn.py - Model architecture (2-3 hours)

- Implement BioGATLayer with edge features
- Implement SpatioTemporalGNN
- Test forward pass with dummy data

4. src/train.py (1-2 hours)

- Implement device setup with MPS priority
- Implement training loop
- Add checkpoint saving and early stopping

5. tests/gnn_vis.py (1-2 hours)

- Implement evaluation metrics
- Create comparison plots
- Add attention visualization

## Key Technical Specifications

**Data Format:**

- Input: DataFrame [6774 rows] -> Reorganized to List[Data] with ~350 windows per worm
- Each Data object:
- x: [window_size=50, num_neurons=196, features=1]
- edge_index: [2, 4085] (from connectome)
- edge_attr: [4085, 2] (gap junction + chemical weights)
- mask: [50, 196] (which neurons recorded)
- y: [196] (target next timestep)

**Model Architecture:**

- Input layer: 1 feature (calcium activity)
- GAT layer 1: 1 -> 64 channels, 4 heads, edge_dim=2
- GAT layer 2: 64 -> 128 channels, 4 heads
- GAT layer 3: 128 -> 256 channels, 2 heads
- LSTM: 256 hidden, 2 layers, processes sequence of GNN outputs
- MLP head: 256 -> 128 -> 1
- Total parameters: ~5-8M

**Training:**

- Loss: Masked MSE (only recorded neurons)
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Batch size: 16 windows
- Device: MPS (M4 Max GPU)
- Expected speed: ~2-3 min/epoch
- Total time: ~3-5 hours for 100 epochs

**Evaluation:**

- Metrics: MSE, MAE, RMSE, R², Correlation
- Compare to baseline R² = 0.87
- Visualize attention weights on connectome
- Analyze per-neuron-type performance

## Success Criteria

1. Model trains without errors on MPS device
2. Validation loss decreases over epochs
3. Test R² > 0.87 (beats naive baseline)
4. Attention weights show biologically plausible patterns
5. Code is clean, modular, and well-documented

## Core Challenge

Current data format: [6774 samples, seq_len] where each row = one neuron's independent time series

Required format: Graph-temporal windows [window_size, num_neurons] with synchronized multi-neuron states per worm

## Notes

- Window size of 50 timesteps chosen for balance of context vs memory
- Can be adjusted in config later
- All data transformations cached to data/cache/graph_temporal/
- Checkpoints saved to models/checkpoints/ (add to .gitignore)
- All plots saved to tests/plots/
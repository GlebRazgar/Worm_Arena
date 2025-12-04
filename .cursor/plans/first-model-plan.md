<!-- 74d393d4-accd-4320-a4d6-e9aa0cea017a ff0f296d-c07a-4ff2-bd81-2a774a7a227f -->
# GNN-LSTM Implementation Plan

## 🎯 What Should the GNN Learn?

Assess if GNN can create a functional connectome that's better than the structural connectome by learning which connections matter for prediction

## Overview

Build a GNN-LSTM hybrid model that uses the C. elegans connectome graph structure as a prior, and temporal calcium activity as input to predict next-timestep neuronal activation. The implementation will be clean and minimal (4 files modified/created), optimized for M4 Max macbook - MPS acceleration.

# The model:

- Prior: Anatomical connectome graph (neurons = nodes, synapse connections = weights of edges).
- Input: Functional data (continuous calcium singal)
- Learn: Re-weigting the connectome. Infer spatial relationships that dont exist. Functional weights (which connections matter for dynamics)
- Output: Calcium activity prediction. 
- Objective: Discover structure-function relationships in the nervous system

## GNN architecture:

Type: GAT.

Reason: Learns the important edge weights via attention. It will take the connectome as a prior, and learn a more correct connectivity from functional data, in cases where connection weights dont correspond perfectly to function.

Adjustments: *Add spatial features to GAT*
attention_ij = f(neural_features, edge_attr, distance_ij)
# - edge_attr_ij = [gap_weight, chem_weight] from connectome (prior)
# - attention_ij = learned functional weight (adjustment)
# - Final weight = attention_ij * edge_attr_ij (combines both)

## Detailed Model Architecture

### Overall Structure: GAT-LSTM Hybrid

The model processes graph-temporal data in two stages:
1. **Spatial Processing (GAT)**: At each timestep, GAT layers aggregate information from neighboring neurons
2. **Temporal Processing (LSTM)**: LSTM processes the sequence of spatial embeddings to capture dynamics

### Forward Pass Flow

```
Input: x = [batch, window_size=50, num_neurons=236, features=1]
       edge_index = [2, 4085]
       edge_attr = [4085, 2]  # [gap_weight, chem_weight]

For each timestep t in window:
    x_t = x[:, t, :, :]  # [batch, 236, 1]
    
    # GAT Layer 1: 1 -> 64, 4 heads
    h1 = GATLayer1(x_t, edge_index, edge_attr)  # [batch, 236, 64]
    
    # GAT Layer 2: 64 -> 128, 4 heads
    h2 = GATLayer2(h1, edge_index, edge_attr)  # [batch, 236, 128]
    
    # GAT Layer 3: 128 -> 256, 2 heads
    h3 = GATLayer3(h2, edge_index, edge_attr)  # [batch, 236, 256]
    
    spatial_embeddings[t] = h3  # Store for LSTM

# Stack all timesteps: [batch, 50, 236, 256]
spatial_sequence = stack(spatial_embeddings)

# Reshape for LSTM: [batch*236, 50, 256]
# Process each neuron's temporal sequence independently
spatial_sequence = spatial_sequence.view(batch*236, 50, 256)

# LSTM: 256 hidden, 2 layers
lstm_out, (h_n, c_n) = LSTM(spatial_sequence)  # [batch*236, 50, 256]
final_hidden = lstm_out[:, -1, :]  # [batch*236, 256] - last timestep

# Reshape back: [batch, 236, 256]
final_hidden = final_hidden.view(batch, 236, 256)

# MLP Prediction Head: 256 -> 128 -> 1
predictions = MLP(final_hidden)  # [batch, 236, 1]

Output: [batch, 236, 1] - predicted calcium activity for next timestep
```

### GAT Layer Architecture

Each GAT layer implements multi-head attention with edge features:

```python
class GATLayer(MessagePassing):
    """
    Graph Attention Layer with edge features.
    
    Attention mechanism:
    - Computes attention scores from node features + edge attributes
    - Multi-head attention (4 heads for layers 1-2, 2 heads for layer 3)
    - Handles directed edges (chemical synapses) and bidirectional (gap junctions)
    """
    
    def forward(x, edge_index, edge_attr):
        # x: [N, in_features]
        # edge_index: [2, E]
        # edge_attr: [E, 2]  # [gap_weight, chem_weight]
        
        # Compute attention scores
        # alpha_ij = softmax(LeakyReLU(a^T [Wx_i || Wx_j || edge_attr_ij]))
        # where || is concatenation
        
        # Aggregate messages with attention weights
        # h_i = sum_j(alpha_ij * Wx_j)
        
        # Concatenate multi-head outputs
        # output = [head1 || head2 || ... || headK]
        
        return output  # [N, out_features]
```

**Key Features:**
- **Edge-aware attention**: Incorporates `edge_attr` (gap + chem weights) into attention computation
- **Multi-head mechanism**: Captures different types of relationships (4 heads → 2 heads)
- **Residual connections**: Optional skip connections between layers
- **Layer normalization**: Stabilizes training

### LSTM Architecture

```python
LSTM(
    input_size=256,      # GAT output dimension
    hidden_size=256,     # Hidden state dimension
    num_layers=2,        # Stacked LSTM layers
    dropout=0.2,         # Dropout between layers
    batch_first=True     # Input format: [batch, seq, features]
)
```

**Design Choice: Per-neuron LSTM**
- Each neuron's temporal sequence processed independently
- Allows neurons to have different temporal dynamics
- More interpretable than shared LSTM

### Prediction Head

```python
MLP(
    Linear(256, 128),
    ReLU(),
    Dropout(0.1),
    Linear(128, 1)  # Single output: predicted calcium activity
)
```

### Loss Function

**Masked MSE Loss:**
```python
loss = MSE(predictions[mask==1], targets[mask==1])
```
- Only computes loss on recorded neurons
- Handles missing data gracefully

### Total Architecture Summary

- **Input**: [batch, 50, 236, 1] - 50 timesteps × 236 neurons × 1 feature
- **GAT Layers**: 3 layers, progressively increasing channels (1→64→128→256)
- **LSTM**: 2 layers, 256 hidden units, processes 50-timestep sequences
- **Output**: [batch, 236, 1] - predicted next-timestep activity
- **Parameters**: ~5-8M total
- **Memory**: ~2-4 GB per batch (MPS optimized)

### Attention Weight Extraction

For visualization, extract attention weights from each GAT layer:
```python
# During forward pass, store attention weights
attention_weights = {
    'layer1': [batch, E, 4],  # 4 heads
    'layer2': [batch, E, 4],
    'layer3': [batch, E, 2]   # 2 heads
}
```
These can be averaged across heads and visualized on the connectome graph.

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
3. GATs validation loss across datasets.
4. GATs validation curve throutghout epochs for each individual dataset (using weights and biases.). 
5. Visualisation of the models outputs as a connectome graph of a worm passing the message, ligting up, lighting down, etc. (we can leave that for later)

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

### 2. models/gat.py (CREATE - ~400 lines)

Complete GAT-LSTM implementation in single file:

**Part A: Data Preparation (150 lines)**

- `create_graph_temporal_dataset(df, connectome, window_size=50, stride=10)`:
- Group DataFrame by worm_id
- For each worm: pivot neurons to columns (196 neurons from connectome)
- Align all recordings to common timesteps
- Create mask [T, N] for missing neurons
- Extract sliding windows [W, N]
- Return List[Data] with x=[W,N,1], edge_index, edge_attr, mask, y=[N]
- (you might have already done this for other models)

- `WormGraphDataset(Dataset)`:
- PyTorch Dataset wrapper
- Handles batching and shuffling
- Caches preprocessed windows to disk

**Part B: Model Architecture (250 lines)**

- `GATLayer(MessagePassing)`:
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
<!-- 74d393d4-accd-4320-a4d6-e9aa0cea017a ff0f296d-c07a-4ff2-bd81-2a774a7a227f -->
# Simple GCN Test Model Architecture Plan



````javascript

**Specific Goals**:

1. **Data Pipeline Validation**: Verify that functional calcium data can be correctly aligned by worm and neuron, synchronized across timesteps, and converted to graph-temporal format
2. **Graph Structure Validation**: Confirm the connectome graph structure is correctly integrated with functional data
3. **Learning Capability Test**: Demonstrate that a simple graph model can learn patterns and beat the naive baseline (R² > 0.87)
4. **Foundation for Complex Models**: Establish working data format and training loop for future GAT-LSTM architectures

**Success Criteria**:

- Model trains without errors
- Loss decreases over epochs
- Validation R² > 0.87 (beats naive baseline: X̂(t+1) = X(t))
- Forward pass completes in <10ms per sample
- Data processing pipeline is validated and reusable

---

## 2. How the GCN Accomplishes This Goal

### 2.1 Graph Convolution Mechanism

The GCN uses **message passing** to aggregate information from connected neurons:

1. **Spatial Aggregation**: Each neuron receives messages from its neighbors in the connectome
2. **Weighted Combination**: Neighbor activations are combined based on graph structure
3. **Feature Transformation**: Learned linear transformations extract predictive patterns
4. **Prediction**: Transformed features predict next-timestep activation

### 2.2 Why GCN Works for This Task

**Biological Justification**:

- Neurons influence each other through synaptic connections (gap junctions + chemical synapses)
- The connectome provides the **structural prior** of which neurons can directly communicate
- Calcium activity propagates through the network following these connections

**Mathematical Justification**:

- GCN learns: `h_i^(l+1) = σ(W^(l) · AGGREGATE({h_j^(l) : j ∈ N(i)})`
- Where `N(i)` = neighbors of neuron i in connectome
- This captures local spatial dependencies in the neural network

**Advantage over Naive Baseline**:

- Naive: `X̂(t+1) = X(t)` - no information sharing between neurons
- GCN: `X̂(t+1) = f(GCN(X(t), connectome))` - uses neighbor information to improve prediction

---

## 3. GCN Architecture

### 3.1 Prior Knowledge (Connectome)

**What We Use**:

- **Graph Structure**: `edge_index` [2, E] - which neurons are connected
- **Node Identity**: `node_label` - mapping from index to neuron name
- **Node Count**: Fixed N = ~196 neurons (from Cook2019 connectome)

**What We Omit (for simplicity)**:

- **Edge Attributes**: Gap junction weights and chemical synapse weights (ignored in simple GCN)
- **Node Types**: Sensory/inter/motor classification (not used as input features)
- **Spatial Positions**: 3D coordinates (not used)
- **Edge Directionality**: Treated as undirected graph (simplification)

**Rationale**: This is a **minimal test** - we only use the graph structure to validate the pipeline. Edge attributes and node types will be added in the full model.

### 3.2 Input

**Per Sample (Single Timestep)**:

- `x`: [N, 1] - Current calcium activation of all N neurons at time t
    - N = number of neurons in connectome (~196)
    - 1 feature = normalized calcium fluorescence value
- `edge_index`: [2, E] - Connectome edges (shared across all samples)
    - E = number of edges (~4,085 for Cook2019)
    - Fixed graph structure, not learnable
- `mask`: [N, 1] - Which neurons have valid recordings (1=valid, 0=missing)
    - Accounts for incomplete recordings (not all neurons recorded in every experiment)

**Batch Format**:

- Batch of B samples: List of B `Data` objects
- Each `Data` object: `{x: [N, 1], edge_index: [2, E], mask: [N, 1], y: [N, 1]}`

### 3.3 Architecture Layers

```
Input: x_t [N, 1]
  ↓
GCN Layer: GCNConv(1 → hidden_dim)
 - Input: [N, 1] (calcium activation)
 - Output: [N, hidden_dim] (spatial embeddings)
 - Operation: Aggregates neighbor activations via graph structure
 - Activation: ReLU
  ↓
Linear Layer: Linear(hidden_dim → 1)
 - Input: [N, hidden_dim]
 - Output: [N, 1] (predicted next activation)
  ↓
Output: x̂_{t+1} [N, 1]
```

**Hyperparameters**:

- `hidden_dim = 32` (small for testing)
- `num_layers = 1` (single GCN layer)
- Total parameters: ~(1×32 + 32×1) × N ≈ 6,272 parameters

### 3.4 Output

**Per Sample**:

- `x̂_{t+1}`: [N, 1] - Predicted calcium activation for all N neurons at time t+1
- Same shape as input `x_t`
- Values are continuous (calcium fluorescence, normalized)

**Training Target**:

- `y`: [N, 1] - Ground truth activation at time t+1
- Loss computed only on masked neurons (where `mask == 1`)

### 3.5 Learning Process

**Loss Function**: Masked Mean Squared Error (MSE)

```
loss = mean((x̂_{t+1} - y_{t+1})²) where mask == 1
```

**Optimization**:

- Optimizer: Adam (lr=1e-3)
- Batch size: 32 samples
- Training: Minimize MSE on (X_t, X_{t+1}) pairs

**What the Model Learns**:

1. **Spatial Patterns**: Which neighbor activations are predictive
2. **Aggregation Weights**: How to combine neighbor information
3. **Non-linear Transformations**: ReLU activation captures non-linear relationships

**What It Cannot Learn (Limitations)**:

- **Temporal Dependencies**: Only uses current timestep, no history
- **Edge Weights**: Ignores gap junction/chemical synapse strengths
- **Long-range Dependencies**: Single layer = 1-hop neighbors only
- **Directionality**: Treats graph as undirected

### 3.6 What This Accomplishes

**Validates**:

1. ✅ Data can be aligned by worm and synchronized across neurons
2. ✅ Graph structure is correctly integrated
3. ✅ Message passing works (neighbors influence predictions)
4. ✅ Model can learn from data (loss decreases)
5. ✅ Pipeline is ready for complex models
````



## 1. Goal

**Primary Objective**: Validate the data pipeline and confirm that a graph-based model can learn from C. elegans neural data before investing in complex architectures.

- Model trains without errors
- Loss decreases over epochs
- Validation R² > 0.87 (beats naive baseline)
- Forward pass <10ms/sample

## 2. How GCN Helps

- Uses connectome structure for spatial aggregation
- Aggregates neighbor activations to predict next-timestep activation
- Validates data alignment, graph integration, and learning capability

## 3. Architecture Summary

- Input: `[N,1]` current neuron activations, `edge_index` (fixed connectome), `mask`
- Layers: `GCNConv(1→32)` + ReLU + `Linear(32→1)`
- Output: `[N,1]` predicted next activation
- Loss: masked MSE; optimizer: Adam 1e-3

## 4. Data Considerations

- Functional data grouped by worm, pivoted to `[T,N]`
- Connectome provides fixed edges
- Mask handles missing neurons; no edge weights, no temporal history

---

## Actionable Implementation Steps

### Step 1: Data Preprocessing Module (`data/preprocessing.py`)

1. Create helper `align_worm_timesteps(worm_df, neuron_list)`:

- Input: subset DataFrame for one worm
- Output: `matrix[T,N]`, `mask[T,N]`, `timestamps[T]`
- Steps:
- Determine common time grid (e.g., union of timestamps, resample to 0.333s)
- For each neuron: interpolate along time grid; fill missing with 0, mask=0

2. Implement `create_graph_temporal_dataset(df, connectome, max_worms=None)`:

- Load neuron index mapping from connectome
- Iterate worms (limit via `max_worms` for debugging)
- For each worm: use `align_worm_timesteps`
- Build (X_t, X_{t+1}) pairs: `[T-1,N]`
- Convert to PyG `Data` objects with fields: `x`, `edge_index`, `mask`, `y`, `worm_id`, `timestep`

3. Add `WormGraphDataset` (inherits PyTorch `Dataset`):

- Stores list of `Data` objects
- Supports train/val split by worm (80/10/10)
- Optional caching to `data/cache/graph_windows_<hash>.pt`

4. Unit test function (within module `if __name__ == "__main__"`):

- Load config-driven connectome & functional data
- Generate dataset for 1 worm, print shapes (#samples, N, T)

### Step 2: Simple GCN Model (`models/test_gcn.py`)

1. Define `SimpleGCN(nn.Module)`:

- `__init__(self, num_nodes, hidden_dim=32)` sets up `GCNConv(1, hidden_dim)` and `Linear(hidden_dim, 1)`
- `forward(self, data)` expects PyG `Data` with `x`, `edge_index`, `mask`
- Apply GCN→ReLU→Linear; return predictions `[N,1]`

2. Add utility `masked_mse(pred, target, mask)`
3. Include `@torch.no_grad()` `evaluate(model, loader, device)` returning metrics

### Step 3: Training Script (`src/train_test_gcn.py`)

1. Parse args: `--config configs/model.yaml`, `--max-worms`, `--epochs`, `--device`
2. Pipeline:

- Load config via `get_config()`
- Load connectome via `load_connectome()`
- Load functional data via `load_functional_data(connectome=graph)`
- Build dataset via `create_graph_temporal_dataset(df, graph, max_worms)`
- Split dataset per worm (train/val/test)
- Wrap in `DataLoader` (batch_size=32, shuffle=True for train)
- Initialize `SimpleTestGCN(num_nodes=graph.num_nodes)`
- Move to device (pref MPS→CUDA→CPU)

3. Training loop:

- For `epoch` in range(E):
- Train: zero grad, forward, loss, backward, step
- Track loss (masked MSE)
- Validation every epoch; compute metrics
- Print progress (loss, R²)
- Save best checkpoint to `models/checkpoints/simple_gcn.pt`

4. After training:

- Run evaluation on test set, compute metrics (MSE, MAE, RMSE, R², Corr)
- Save metrics to `models/checkpoints/simple_gcn_metrics.json`

### Step 4: Visualization & Comparison (`tests/test_gcn_vis.py`)

1. Load best checkpoint + cached dataset
2. Run evaluation vs naive baseline (reuse `models.baseline` functions)
3. Generate plots saved to `tests/plots/`:

- `simple_gcn_vs_baseline.png`: bar chart MSE/R² comparison
- `sample_predictions.png`: predicted vs actual traces for 6 neurons
- `loss_curve.png`: train vs val loss over epochs (store history during training)

### Step 5: Documentation & Configs

1. Update `configs/model.yaml` with `simple_gcn` section (hidden_dim, epochs, lr, batch_size)
2. Document usage in `README.md`:

- How to run training: `python src/train_test_gcn.py --epochs 100`
- Expected results and purpose of test model

3. Add cache path `data/cache/graph_windows_*.pt` to `.gitignore`

### Step 6: Validation Checklist

- [ ] Run preprocessing unit test (Step 1.4)
- [ ] Run training script for 5 epochs (sanity check)
- [ ] Confirm checkpoint & metrics saved
- [ ] Compare vs naive baseline; ensure improvement
- [ ] Review plots in `tests/plots/`
- [ ] Document findings (loss, R², potential issues)

---

## Ready to Implement

This step-by-step plan provides concrete tasks to implement the simple GCN test model, ensuring the pipeline is validated before moving to complex architectures.

### To-dos

- [ ] Create data/preprocessing.py with align_worm_data, create_temporal_pairs, create_graph_dataset, WormGraphDataset, and caching
- [ ] Create models/test_gcn.py with SimpleTestGCN class and masked_mse_loss function
- [ ] Create src/train_test_gcn.py with setup_device, train_epoch, evaluate, train_model, and CLI
- [ ] Create tests/test_gcn_vis.py with plot_training_curves, plot_gcn_vs_baseline, plot_prediction_scatter
- [ ] Update configs/model.yaml with test_gcn section and .gitignore with cache paths
- [ ] Run end-to-end test: train model, verify R² > 0.87, generate all plots
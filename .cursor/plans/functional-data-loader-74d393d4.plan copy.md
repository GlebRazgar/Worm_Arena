<!-- 74d393d4-accd-4320-a4d6-e9aa0cea017a ff0f296d-c07a-4ff2-bd81-2a774a7a227f -->
# Simple GCN Test Model Architecture Plan

## 1. Goal

**Primary Objective**: Validate the entire data processing pipeline and confirm that a graph-based model can learn from the C. elegans neural data.

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

**Tests**:

- Data preprocessing pipeline
- Graph data format (PyTorch Geometric)
- Training loop infrastructure
- Evaluation metrics
- Device handling (MPS/CPU)

**Foundation for**:

- Adding edge attributes (gap/chemical weights)
- Adding temporal modeling (LSTM)
- Adding attention (GAT)
- Adding link prediction head

---

## 4. Data We're Dealing With

### 4.1 Current Data Format

**Functional Data (DataFrame)**:

- Structure: One row per (neuron, worm, recording)
- Columns:
    - `neuron`: Neuron name (e.g., "AVAL", "RIML")
    - `worm`: Worm identifier (e.g., "worm_001")
    - `original_calcium_data`: List of float values (time series)
    - `original_time_in_seconds`: List of float values (timestamps)
    - `source_dataset`: Origin dataset (e.g., "Leifer2023")
- Shape: ~6,774 rows (after filtering to connectome neurons)

**Connectome Data (Graph)**:

- Structure: PyTorch Geometric `Data` object
- Nodes: ~196 neurons (Cook2019)
- Edges: ~4,085 connections
- Edge attributes: [gap_weight, chem_weight] per edge
- Node attributes: [sensory, inter, motor] one-hot encoding

### 4.2 Data Transformation Required

**Challenge**: Functional data is **not synchronized**

- Each row has independent time series
- Different neurons may have different timesteps
- Need to align by worm and create synchronized [T, N] matrices

**Transformation Steps**:

1. **Group by Worm**:
   ```python
   worms = df.groupby('worm')
   ```

2. **For Each Worm - Pivot Neurons to Columns**:
   ```python
   # Pivot: rows = timesteps, columns = neurons
   worm_matrix = pivot_by_neuron(worm_df)  # [T, N]
   ```

3. **Align Timesteps**:

      - Interpolate/resample to common time grid
      - Handle missing neurons (mask = 0)
      - Handle variable sequence lengths

4. **Create (X_t, X_{t+1}) Pairs**:
   ```python
   X = worm_matrix[:-1]  # [T-1, N]
   Y = worm_matrix[1:]   # [T-1, N]
   ```

5. **Convert to PyG Data Objects**:
   ```python
   for t in range(T-1):
       data = Data(
           x=X[t],           # [N, 1]
           edge_index=graph.edge_index,  # [2, E] (shared)
           mask=mask[t],      # [N, 1]
           y=Y[t]             # [N, 1]
       )
   ```


### 4.3 Data Considerations

**What We Consider**:

- ✅ All neurons present in connectome (~196)
- ✅ Synchronized timesteps per worm
- ✅ Missing data handling (masking)
- ✅ Multiple worms (batch across worms)

**What We Omit (for simplicity)**:

- ❌ Edge attributes (gap/chemical weights) - ignored in simple GCN
- ❌ Node type features (sensory/inter/motor) - not used
- ❌ Spatial positions - not used
- ❌ Temporal windows - only single timestep (X_t → X_{t+1})
- ❌ Edge directionality - treated as undirected
- ❌ Variable graph structure - fixed connectome

---

## 5. What the GCN Will Consider

### 5.1 Graph Structure (Connectome)

**Included**:

- **Neighbor Relationships**: Which neurons are directly connected
- **Graph Topology**: Network structure (who talks to whom)
- **Fixed Prior**: Connectome structure is not learnable (edge_index is fixed)

**Not Included**:

- Edge weights (gap/chemical strengths)
- Edge directionality (treated as undirected)
- Multi-hop paths (only 1-hop neighbors in single layer)

### 5.2 Functional Data

**Included**:

- **Current Activation**: Calcium fluorescence at time t for all neurons
- **Synchronized State**: All neurons at same timestep (per worm)
- **Missing Data**: Masked neurons (not recorded) are excluded from loss

**Not Included**:

- Historical activations (only current timestep)
- Temporal patterns (no LSTM/RNN)
- Cross-worm patterns (each worm processed independently)

### 5.3 Learning Scope

**What It Can Learn**:

- How to aggregate neighbor activations
- Which neighbors are most predictive
- Non-linear transformations of spatial patterns
- Local spatial dependencies (1-hop)

**What It Cannot Learn**:

- Temporal dependencies (no history)
- Long-range dependencies (only 1-hop)
- Edge-specific weights (ignores edge attributes)
- Missing links (no link prediction)

---

## 6. What the GCN Will Omit

### 6.1 Temporal Modeling

**Omitted**: LSTM, RNN, or any temporal memory

- **Reason**: Test spatial (graph) structure first
- **Impact**: Model only sees current timestep, no history
- **Future**: Will add LSTM in full model

### 6.2 Edge Attributes

**Omitted**: Gap junction and chemical synapse weights

- **Reason**: Simplest possible test
- **Impact**: All edges treated equally (structure only)
- **Future**: Will add edge-gated attention in GAT model

### 6.3 Node Features

**Omitted**: Neuron type (sensory/inter/motor) as input

- **Reason**: Test with minimal features (just activation)
- **Impact**: Model doesn't know neuron types
- **Future**: Can add as additional input features

### 6.4 Spatial Information

**Omitted**: 3D positions of neurons

- **Reason**: Focus on connectivity, not geometry
- **Impact**: Model doesn't use spatial proximity
- **Future**: Can add positional encodings if needed

### 6.5 Link Prediction

**Omitted**: Predicting missing edges in connectome

- **Reason**: Single-task focus (activation prediction only)
- **Impact**: Cannot discover new connections
- **Future**: Will add link prediction head in full model

### 6.6 Multi-layer Depth

**Omitted**: Multiple GCN layers (only 1 layer)

- **Reason**: Minimal test, faster training
- **Impact**: Only 1-hop neighbors influence predictions
- **Future**: Will stack multiple layers in full model

---

## 7. Input/Output Specification

### 7.1 Input Format

**Single Sample**:

```python
Data(
    x=torch.tensor([N, 1]),           # Current activation
    edge_index=torch.tensor([2, E]),  # Connectome edges
    mask=torch.tensor([N, 1]),        # Valid neurons
    y=torch.tensor([N, 1])            # Target (next timestep)
)
```

**Batch**:

- List of B `Data` objects
- PyTorch Geometric `DataLoader` handles batching

### 7.2 Output Format

**Model Forward Pass**:

```python
x̂_{t+1} = model(x_t, edge_index)  # [N, 1]
```

**Training Output**:

- Predictions: `[N, 1]` per sample
- Loss: Scalar (masked MSE)

**Evaluation Output**:

- Metrics: MSE, MAE, RMSE, R², Correlation
- Per-neuron and aggregate statistics

### 7.3 Data Flow

```
Functional DataFrame
  ↓ (group by worm)
Worm-specific DataFrames
  ↓ (pivot neurons → columns)
Synchronized Matrices [T, N]
  ↓ (create pairs)
(X_t, X_{t+1}) Pairs
  ↓ (convert to PyG)
List[Data] objects
  ↓ (DataLoader)
Batched training samples
  ↓ (GCN forward)
Predictions [B, N, 1]
  ↓ (loss computation)
Scalar loss
```

---

## 8. Whether It Accomplishes Our Goal

### 8.1 Pipeline Validation ✅

**Will Accomplish**:

- Data alignment by worm: ✅ (groupby + pivot)
- Neuron synchronization: ✅ (interpolation/resampling)
- Graph integration: ✅ (PyG Data objects)
- Training loop: ✅ (standard PyTorch training)

**Success Indicator**: Model trains without errors

### 8.2 Graph Structure Validation ✅

**Will Accomplish**:

- Connectome integration: ✅ (edge_index in Data objects)
- Message passing: ✅ (GCN layer aggregates neighbors)
- Graph format: ✅ (PyTorch Geometric compatible)

**Success Indicator**: Forward pass completes, gradients flow

### 8.3 Learning Capability Test ⚠️

**Expected Outcome**:

- Loss decreases: ✅ (should see training loss drop)
- Beats baseline: ⚠️ (may or may not beat R²=0.87)
    - **If beats baseline**: Pipeline works, ready for complex models
    - **If doesn't beat baseline**: Need to investigate data alignment or model capacity

**Success Criteria**:

- Training loss decreases over epochs
- Validation R² > 0.87 (beats naive baseline)
- Model learns spatial patterns (neighbors matter)

### 8.4 Foundation for Complex Models ✅

**Will Accomplish**:

- Working data format: ✅ (reusable for GAT-LSTM)
- Training infrastructure: ✅ (can extend to complex models)
- Evaluation framework: ✅ (metrics and visualization)

**Success Indicator**: Can easily extend to GAT-LSTM without rewriting data pipeline

---

## 9. Limitations and Future Extensions

### 9.1 Current Limitations

1. **No Temporal Modeling**: Only uses current timestep
2. **No Edge Attributes**: Ignores gap/chemical weights
3. **Shallow**: Single layer = 1-hop only
4. **Undirected**: Ignores edge directionality
5. **No Link Prediction**: Cannot discover missing edges

### 9.2 Future Extensions (After Validation)

1. **Add LSTM**: Temporal modeling over windows
2. **Add Edge Attributes**: Gap/chemical weights in GAT
3. **Add Attention**: GAT with learnable edge weights
4. **Add Link Prediction**: Discover missing connections
5. **Add Multi-layer**: Deeper GCN for long-range dependencies

---

## 10. Implementation Files

### 10.1 Files to Create

1. **`models/test_gcn.py`** (~150 lines)

      - `SimpleTestGCN` model class
      - Single GCN layer + linear predictor

2. **`data/preprocessing.py`** (~200 lines)

      - `create_graph_temporal_dataset()` - Main data transformation
      - `align_worm_timesteps()` - Synchronize neurons per worm
      - `WormGraphDataset` - PyTorch Dataset wrapper

3. **`src/train_test_gcn.py`** (~150 lines)

      - Training loop
      - Evaluation metrics
      - Device setup (MPS/CPU)

4. **`tests/test_gcn_vis.py`** (~100 lines)

      - Visualization of predictions
      - Comparison with baseline
      - Loss curves

### 10.2 Files to Modify

1. **`configs/model.yaml`** - Add test GCN config
2. **`.gitignore`** - Add cache directories

---

## 11. Expected Results

### 11.1 Training

- **Epochs**: 50-100 (should converge quickly)
- **Time**: ~5-10 minutes on MPS
- **Loss**: Should decrease from ~0.1 to ~0.05 (MSE)
- **R²**: Should achieve >0.87 (beats baseline)

### 11.2 Validation

- **Metrics**: MSE, MAE, RMSE, R², Correlation
- **Comparison**: vs Naive baseline (X̂(t+1) = X(t))
- **Visualization**: Prediction vs ground truth plots

### 11.3 Success Indicators

✅ Model trains without errors

✅ Loss decreases over epochs

✅ Validation R² > 0.87

✅ Data pipeline is validated

✅ Ready for complex models

---

## Summary

This simple GCN test model serves as a **minimal validation** of the entire pipeline. It uses only:

- Graph structure (connectome edges)
- Current activation (single timestep)
- Basic message passing (1-layer GCN)

It omits:

- Temporal modeling (LSTM)
- Edge attributes (weights)
- Node features (types)
- Link prediction

**Goal**: Validate that data processing works and models can learn, before building complex architectures.

**Success**: If this works, we know the foundation is solid and can confidently add GAT, LSTM, and link prediction.
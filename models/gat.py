"""
GAT-LSTM Model for C. elegans Neural Activity Prediction

A Graph Attention Network combined with LSTM for spatio-temporal prediction.
Uses the connectome as a structural prior and learns functional connectivity.

Optimized for MPS (Apple Silicon) acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.data import Data
import math


class EdgeGATConv(MessagePassing):
    """
    Graph Attention Convolution with Edge Features.
    
    Incorporates edge attributes (gap junction + chemical synapse weights)
    into the attention mechanism. This allows the model to use the
    anatomical connectome as a prior while learning functional weights.
    
    Attention mechanism:
        alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || We * edge_attr_ij]))
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension (per head)
        heads: Number of attention heads
        edge_dim: Edge feature dimension (2 for gap + chem)
        dropout: Dropout rate for attention weights
        negative_slope: LeakyReLU negative slope
        add_self_loops: Whether to add self-loops
    """
    
    def __init__(self, in_channels, out_channels, heads=1, edge_dim=2,
                 dropout=0.0, negative_slope=0.2, add_self_loops=True):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        self._add_self_loops = add_self_loops
        
        # Linear transformations for node features
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Edge feature transformation
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        
        # Attention mechanism
        # Attention vector for [src || dst || edge] concatenation
        self.att = nn.Parameter(torch.empty(1, heads, 3 * out_channels))
        
        # Bias
        self.bias = nn.Parameter(torch.empty(heads * out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
        """
        Forward pass.
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge features
            return_attention_weights: Whether to return attention weights
        
        Returns:
            out: [N, heads * out_channels] updated node features
            attention_weights: (optional) [E, heads] attention weights
        """
        N = x.size(0)
        
        # Add self-loops
        if self._add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, 
                fill_value=0.0,  # Self-loop edge features are zero
                num_nodes=N
            )
        
        E = edge_index.size(1)
        
        # Transform node features: [N, heads * out_channels]
        x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)  # [N, heads, out]
        x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)  # [N, heads, out]
        
        # Transform edge features: [E, heads * out_channels]
        edge_attr_transformed = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)  # [E, heads, out]
        
        # Propagate messages
        out = self.propagate(
            edge_index, 
            x=(x_src, x_dst), 
            edge_attr=edge_attr_transformed,
            size=None
        )
        
        # Reshape and add bias
        out = out.view(-1, self.heads * self.out_channels)  # [N, heads * out]
        out = out + self.bias
        
        if return_attention_weights:
            # Return attention weights for visualization
            alpha = self._compute_attention_weights(x_src, x_dst, edge_index, edge_attr_transformed)
            return out, (edge_index, alpha)
        
        return out
    
    def _compute_attention_weights(self, x_src, x_dst, edge_index, edge_attr):
        """Compute attention weights for all edges."""
        src_idx, dst_idx = edge_index
        
        # Get source and destination features for each edge
        x_src_e = x_src[src_idx]  # [E, heads, out]
        x_dst_e = x_dst[dst_idx]  # [E, heads, out]
        
        # Concatenate: [E, heads, 3 * out]
        alpha_input = torch.cat([x_src_e, x_dst_e, edge_attr], dim=-1)
        
        # Compute attention scores
        alpha = (alpha_input * self.att).sum(dim=-1)  # [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, dst_idx, num_nodes=x_src.size(0))
        
        return alpha
    
    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        """
        Compute messages from source nodes j to target nodes i.
        
        Args:
            x_j: [E, heads, out] source node features
            x_i: [E, heads, out] target node features  
            edge_attr: [E, heads, out] edge features
            index: Target node indices
            ptr: Pointer for segment operations
            size_i: Number of target nodes
        
        Returns:
            messages: [E, heads, out]
        """
        # Concatenate source, target, and edge features
        alpha_input = torch.cat([x_j, x_i, edge_attr], dim=-1)  # [E, heads, 3*out]
        
        # Compute attention scores
        alpha = (alpha_input * self.att).sum(dim=-1)  # [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)  # [E, heads, out]
    
    def update(self, aggr_out):
        """Update node embeddings after aggregation."""
        return aggr_out  # [N, heads, out]


class GATLSTM(nn.Module):
    """
    GAT-LSTM Hybrid Model for Spatio-Temporal Neural Activity Prediction.
    
    Architecture:
        1. Multi-layer GAT: Processes spatial graph structure at each timestep
        2. LSTM: Captures temporal dynamics across window
        3. MLP Predictor: Outputs next-timestep prediction
    
    Uses residual connection: output = input + delta
    
    OPTIMIZED for MPS with vectorized processing.
    
    Args:
        config: Model configuration dict (from model.yaml)
    """
    
    def __init__(self, config=None, 
                 in_channels=1,
                 gat_hidden=[32],       # SINGLE layer default
                 gat_heads=[2],         # Minimal heads
                 edge_dim=2,
                 lstm_hidden=32,        # Minimal
                 lstm_layers=1,         # Single layer
                 dropout=0.0,           # No dropout
                 residual=True,
                 use_vectorized=True):
        super().__init__()
        
        # Parse config if provided
        if config is not None:
            gat_cfg = config.get('gat', {})
            lstm_cfg = config.get('lstm', {})
            pred_cfg = config.get('predictor', {})
            
            in_channels = gat_cfg.get('in_channels', 1)
            gat_hidden = gat_cfg.get('hidden_channels', [32])
            gat_heads = gat_cfg.get('heads', [2])
            edge_dim = gat_cfg.get('edge_dim', 2)
            dropout = gat_cfg.get('dropout', 0.0)
            
            lstm_hidden = lstm_cfg.get('hidden_size', 32)
            lstm_layers = lstm_cfg.get('num_layers', 1)
            
            residual = config.get('residual', True)
            use_vectorized = config.get('use_vectorized', True)
        
        self.in_channels = in_channels
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.edge_dim = edge_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.residual = residual
        self.use_vectorized = use_vectorized
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        prev_dim = in_channels
        for i, (hidden_dim, heads) in enumerate(zip(gat_hidden, gat_heads)):
            self.gat_layers.append(
                EdgeGATConv(
                    in_channels=prev_dim,
                    out_channels=hidden_dim // heads,  # Per-head dimension
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=gat_hidden[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=False  # [seq, batch, features]
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data, return_attention=False):
        """
        Forward pass for graph-temporal data.
        Uses vectorized processing for MPS efficiency.
        
        Args:
            data: PyG Data object with:
                - x: [N, window_size] temporal window of activations
                - edge_index: [2, E] connectome edges
                - edge_attr: [E, 2] edge features (gap + chem)
                - mask: [N, 1] valid neurons mask
            return_attention: Whether to return attention weights
        
        Returns:
            predictions: [N, 1] predicted next-timestep activation
            attention_weights: (optional) dict of attention weights per layer
        """
        # Use vectorized version by default for speed
        if self.use_vectorized and not return_attention:
            return self.forward_vectorized(data, return_attention=False)
        
        # Sequential version (slower, but supports attention extraction)
        x_window = data.x  # [N, window_size]
        edge_index = data.edge_index  # [2, E]
        edge_attr = data.edge_attr  # [E, 2]
        
        N = x_window.shape[0]
        window_size = x_window.shape[1]
        
        # Get last timestep for residual
        x_t = x_window[:, -1:]  # [N, 1]
        
        # Process each timestep through GAT
        attention_weights = {} if return_attention else None
        gat_outputs = []
        
        for t in range(window_size):
            x_t_input = x_window[:, t:t+1]  # [N, 1]
            
            # Pass through GAT layers
            h = x_t_input
            for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
                if return_attention and t == window_size - 1:
                    # Only get attention for last timestep
                    h, (edge_idx, alpha) = gat(h, edge_index, edge_attr, return_attention_weights=True)
                    attention_weights[f'layer_{i}'] = alpha
                else:
                    h = gat(h, edge_index, edge_attr)
                
                h = norm(h)
                if i < len(self.gat_layers) - 1:
                    h = F.elu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            
            gat_outputs.append(h)  # [N, gat_hidden[-1]]
        
        # Stack: [window_size, N, gat_hidden[-1]]
        gat_sequence = torch.stack(gat_outputs, dim=0)
        
        # LSTM: Process temporal sequence
        lstm_out, (h_n, c_n) = self.lstm(gat_sequence)  # [window_size, N, lstm_hidden]
        
        # Take last timestep output
        lstm_final = lstm_out[-1]  # [N, lstm_hidden]
        
        # Predict delta
        delta = self.predictor(lstm_final)  # [N, 1]
        
        # Residual connection
        if self.residual:
            out = x_t + delta
        else:
            out = delta
        
        if return_attention:
            return out, attention_weights
        return out
    
    def forward_vectorized(self, data, return_attention=False):
        """
        Vectorized forward pass for better MPS performance.
        Processes all timesteps in parallel through GAT.
        
        Args:
            data: PyG Data object (same as forward)
            return_attention: Whether to return attention weights
        
        Returns:
            predictions: [N, 1]
            attention_weights: (optional)
        """
        x_window = data.x  # [N, window_size]
        edge_index = data.edge_index  # [2, E]
        edge_attr = data.edge_attr  # [E, 2]
        
        N = x_window.shape[0]
        window_size = x_window.shape[1]
        E = edge_index.shape[1]
        
        x_t = x_window[:, -1:]  # [N, 1] for residual
        
        # Vectorize: reshape to [N * window_size, 1]
        x_batched = x_window.T.reshape(-1, 1)  # [window_size * N, 1]
        
        # Create batched edge_index: shift indices for each timestep
        offsets = torch.arange(window_size, device=edge_index.device, dtype=edge_index.dtype) * N
        offsets = offsets.view(-1, 1, 1)  # [window_size, 1, 1]
        edge_expanded = edge_index.unsqueeze(0).expand(window_size, -1, -1)  # [window_size, 2, E]
        edge_index_batched = (edge_expanded + offsets).reshape(2, -1)  # [2, E * window_size]
        
        # Replicate edge_attr
        edge_attr_batched = edge_attr.unsqueeze(0).expand(window_size, -1, -1).reshape(-1, self.edge_dim)
        
        # Pass through GAT layers (all timesteps at once)
        h = x_batched
        attention_weights = {} if return_attention else None
        
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            if return_attention:
                h, (_, alpha) = gat(h, edge_index_batched, edge_attr_batched, return_attention_weights=True)
                # Only keep last timestep's attention
                attention_weights[f'layer_{i}'] = alpha[-E:]  # Last E edges
            else:
                h = gat(h, edge_index_batched, edge_attr_batched)
            
            h = norm(h)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Reshape: [window_size * N, hidden] -> [window_size, N, hidden]
        gat_sequence = h.reshape(window_size, N, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(gat_sequence)
        lstm_final = lstm_out[-1]  # [N, lstm_hidden]
        
        # Predict
        delta = self.predictor(lstm_final)
        
        if self.residual:
            out = x_t + delta
        else:
            out = delta
        
        if return_attention:
            return out, attention_weights
        return out


def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss only on valid (unmasked) neurons.
    
    Args:
        pred: [N, 1] predictions
        target: [N, 1] targets
        mask: [N, 1] mask (1=valid, 0=ignore)
    
    Returns:
        scalar loss
    """
    pred = pred.view(-1)
    target = target.view(-1)
    mask = mask.view(-1)
    
    valid_pred = pred[mask == 1]
    valid_target = target[mask == 1]
    
    if len(valid_pred) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return F.mse_loss(valid_pred, valid_target)


def create_gat_lstm_model(config=None, device='cpu'):
    """
    Factory function to create GAT-LSTM model.
    
    Args:
        config: Model configuration dict
        device: Target device ('mps', 'cuda', 'cpu')
    
    Returns:
        model: GATLSTM model on specified device
    """
    model = GATLSTM(config=config)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing GAT-LSTM Model...")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create dummy data
    N = 236  # Number of neurons
    E = 4085  # Number of edges
    window_size = 50
    
    x = torch.randn(N, window_size)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 2)  # gap + chem
    mask = torch.ones(N, 1)
    y = torch.randn(N, 1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask, y=y)
    data = data.to(device)
    
    # Create model
    model = GATLSTM()
    model = model.to(device)
    print(f"Model parameters: {model.num_params:,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        out = model(data)
        print(f"Output shape: {out.shape}")
        print(f"Output device: {out.device}")
    
    # Test with attention
    with torch.no_grad():
        out, attn = model(data, return_attention=True)
        print(f"Attention layers: {list(attn.keys())}")
        for k, v in attn.items():
            print(f"  {k}: {v.shape}")
    
    # Test vectorized forward
    with torch.no_grad():
        out_vec = model.forward_vectorized(data)
        print(f"Vectorized output shape: {out_vec.shape}")
    
    # Test loss
    loss = masked_mse_loss(out, data.y, data.mask)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")


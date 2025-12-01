"""
Advanced GCN-LSTM Hybrid Model
Multi-layer GCN with edge-gated attention + Bidirectional LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops


class EdgeGatedGCNConv(MessagePassing):
    """
    Edge-gated GCN layer that uses edge attributes (gap junction + chemical synapse weights).
    
    Incorporates connectome edge weights as a prior and learns corrections.
    """
    def __init__(self, in_channels, out_channels, edge_dim=2):
        super().__init__(aggr='add', flow='source_to_target')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge gate: learns how to weight edges based on edge attributes
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim + 2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()  # Gate between 0 and 1
        )
        
        # Edge correction: learns adjustments to edge weights
        self.edge_correction = nn.Sequential(
            nn.Linear(edge_dim + 2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        for layer in self.edge_gate:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.edge_correction:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge attributes (gap, chem weights)
        
        Returns:
            out: [N, out_channels] updated node features
        """
        # Add self-loops for residual connections
        # For self-loops, edge_attr should be zeros
        edge_index_with_loops, edge_attr_with_loops = add_self_loops(
            edge_index, edge_attr, num_nodes=x.size(0), fill_value='mean'
        )
        
        # Ensure edge_attr_with_loops has correct shape
        if edge_attr_with_loops is None:
            # If no edge_attr provided, create zeros
            edge_attr_with_loops = torch.zeros(edge_index_with_loops.shape[1], self.edge_dim, 
                                               device=x.device, dtype=x.dtype)
        elif edge_attr_with_loops.shape[0] < edge_index_with_loops.shape[1]:
            # Pad with zeros for self-loops
            num_loops = edge_index_with_loops.shape[1] - edge_attr.shape[0]
            zeros = torch.zeros(num_loops, self.edge_dim, device=edge_attr.device, dtype=edge_attr.dtype)
            edge_attr_with_loops = torch.cat([edge_attr, zeros], dim=0)
        
        # Start propagating messages
        return self.propagate(edge_index_with_loops, x=x, edge_attr=edge_attr_with_loops)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages from source nodes j to target nodes i.
        
        Args:
            x_i: [E, in_channels] target node features
            x_j: [E, in_channels] source node features
            edge_attr: [E, edge_dim] edge attributes
        
        Returns:
            messages: [E, out_channels]
        """
        # Concatenate edge attributes with node features
        edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)  # [E, edge_dim + 2*in_channels]
        
        # Compute edge gate (attention-like weighting)
        gate = self.edge_gate(edge_input)  # [E, 1]
        
        # Compute edge correction (learnable adjustment)
        correction = self.edge_correction(edge_input)  # [E, 1]
        
        # Combine: gate * (original edge weight + correction)
        # For self-loops, edge_attr is zero, so correction acts as residual
        edge_weight = gate * (edge_attr.sum(dim=-1, keepdim=True) + correction)  # [E, 1]
        
        # Transform source node features
        x_j_transformed = self.lin(x_j)  # [E, out_channels]
        
        # Weighted message
        return edge_weight * x_j_transformed


class AdvancedGCNLSTM(nn.Module):
    """
    Advanced GCN-LSTM Hybrid Model with:
    - Multi-layer GCN (2-3 layers) for multi-hop spatial patterns
    - Edge-gated GCN using gap junction/chemical synapse weights
    - Bidirectional LSTM for past and future context
    - Residual connections
    """
    
    def __init__(self, 
                 num_gcn_layers=3,
                 hidden_dim=64, 
                 lstm_hidden=128, 
                 num_lstm_layers=2, 
                 window_size=10,
                 dropout=0.1,
                 edge_dim=2):
        """
        Args:
            num_gcn_layers: Number of GCN layers (2-3 recommended)
            hidden_dim: Hidden dimension for GCN layers
            lstm_hidden: Hidden dimension for LSTM
            num_lstm_layers: Number of LSTM layers
            window_size: Number of timesteps in input window (10-20 recommended)
            dropout: Dropout rate
            edge_dim: Dimension of edge attributes (2: gap + chem)
        """
        super().__init__()
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.window_size = window_size
        self.edge_dim = edge_dim
        
        # Multi-layer GCN with edge gating
        self.gcn_layers = nn.ModuleList()
        
        # First layer: 1 -> hidden_dim
        self.gcn_layers.append(EdgeGatedGCNConv(1, hidden_dim, edge_dim=edge_dim))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(EdgeGatedGCNConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
        
        # Last layer: hidden_dim -> hidden_dim (if more than 1 layer)
        if num_gcn_layers > 1:
            self.gcn_layers.append(EdgeGatedGCNConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=False,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True  # BIDIRECTIONAL!
        )
        
        # Prediction head: LSTM output → delta
        # Bidirectional LSTM outputs 2*lstm_hidden
        self.predictor = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),  # 2x because bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data):
        """
        Forward pass with VECTORIZED multi-layer GCN + Bidirectional LSTM.
        
        Args:
            data: PyG Data object with:
                - x: [N, window_size] temporal window
                - edge_index: [2, E] connectome edges
                - edge_attr: [E, edge_dim] edge attributes (gap, chem)
                - mask: [N, 1] valid neurons
        
        Returns:
            predictions: [N, 1] predicted next-timestep activation
        """
        x_window = data.x  # [N, window_size]
        edge_index = data.edge_index  # [2, E]
        edge_attr = data.edge_attr  # [E, edge_dim]
        
        # Get last timestep for residual
        x_t = x_window[:, -1:] if x_window.shape[1] > 1 else x_window  # [N, 1]
        
        window_size = x_window.shape[1]
        N = x_window.shape[0]
        E = edge_index.shape[1]
        
        # VECTORIZED: Process all timesteps in parallel
        # Reshape input: [N, window_size] -> [N*window_size, 1]
        x_batched = x_window.T.reshape(-1, 1)  # [window_size*N, 1]
        
        # Create batched edge_index and edge_attr
        offsets = torch.arange(window_size, device=edge_index.device, dtype=edge_index.dtype) * N
        offsets = offsets.view(-1, 1, 1)
        
        edge_expanded = edge_index.unsqueeze(0).expand(window_size, -1, -1)
        edge_index_batched = (edge_expanded + offsets).reshape(2, -1)  # [2, E*window_size]
        
        # Replicate edge_attr for each timestep
        edge_attr_batched = edge_attr.unsqueeze(0).expand(window_size, -1, -1).reshape(-1, self.edge_dim)  # [E*window_size, edge_dim]
        
        # Multi-layer GCN processing (all timesteps in parallel)
        h = x_batched
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, edge_index_batched, edge_attr_batched)
            if i < len(self.gcn_layers) - 1:  # ReLU between layers (not after last)
                h = F.relu(h)
                h = F.dropout(h, p=0.1, training=self.training)
        
        # Reshape back: [N*window_size, hidden_dim] -> [window_size, N, hidden_dim]
        gcn_sequence = h.reshape(window_size, N, self.hidden_dim)  # [T, N, hidden_dim]
        
        # Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(gcn_sequence)  # lstm_out: [T, N, 2*lstm_hidden]
        
        # Use last output (most recent timestep)
        lstm_final = lstm_out[-1]  # [N, 2*lstm_hidden] (bidirectional)
        
        # Predict delta
        delta = self.predictor(lstm_final)  # [N, 1]
        
        # RESIDUAL: Output = Last input + Delta
        out = x_t + delta
        
        return out


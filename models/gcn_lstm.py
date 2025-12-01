"""
GCN-LSTM Hybrid Model
Combines Graph Convolutional Networks (spatial) with LSTM (temporal) for neural activity prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLSTM(nn.Module):
    """
    GCN-LSTM Hybrid Model with RESIDUAL connection.
    
    Architecture:
        Input: Temporal window [N, T] where T = window_size
        For each timestep t:
            X_t [N, 1] → GCN → h_t [N, hidden_dim]
        Stack: [T, N, hidden_dim]
        LSTM: Process temporal sequence → [N, lstm_hidden]
        Predict delta: [N, 1]
        Output: X_t + delta (residual)
    
    Captures both:
        - Spatial patterns (via GCN on connectome)
        - Temporal patterns (via LSTM over window)
    """
    
    def __init__(self, hidden_dim=64, lstm_hidden=128, num_lstm_layers=2, window_size=5, dropout=0.1):
        """
        Args:
            hidden_dim: Hidden dimension for GCN layers
            lstm_hidden: Hidden dimension for LSTM
            num_lstm_layers: Number of LSTM layers
            window_size: Number of timesteps in input window
            dropout: Dropout rate for LSTM
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.window_size = window_size
        
        # GCN layer: processes spatial structure at each timestep
        self.gcn = GCNConv(1, hidden_dim)
        
        # LSTM: processes temporal sequence of GCN outputs
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=False,  # Input format: [seq_len, batch, features]
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Prediction head: LSTM output → delta
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data):
        """
        Forward pass with VECTORIZED temporal window processing (MPS optimized).
        
        Args:
            data: PyG Data object with:
                - x: [N, window_size] temporal window of activations
                - edge_index: [2, E] connectome edges
                - mask: [N, 1] valid neurons
        
        Returns:
            predictions: [N, 1] predicted next-timestep activation
                         = x_t + delta (residual prediction)
        """
        x_window = data.x  # [N, window_size]
        edge_index = data.edge_index  # [2, E]
        
        # Get the last timestep for residual connection
        x_t = x_window[:, -1:] if x_window.shape[1] > 1 else x_window  # [N, 1]
        
        window_size = x_window.shape[1]
        
        # VECTORIZED: Process all timesteps in parallel (MPS optimized)
        # Strategy: Create a batched graph where each "node" is (neuron, timestep)
        # This allows processing all timesteps in a single GCN call
        
        N = x_window.shape[0]
        E = edge_index.shape[1]
        
        # Reshape input: [N, window_size] -> [N*window_size, 1]
        x_batched = x_window.T.reshape(-1, 1)  # [window_size*N, 1]
        
        # VECTORIZED edge_index construction (no Python loops!)
        # Create offsets for each timestep: [0, N, 2*N, ..., (window_size-1)*N]
        offsets = torch.arange(window_size, device=edge_index.device, dtype=edge_index.dtype) * N
        offsets = offsets.view(-1, 1, 1)  # [window_size, 1, 1]
        
        # Expand edge_index: [2, E] -> [window_size, 2, E]
        edge_expanded = edge_index.unsqueeze(0).expand(window_size, -1, -1)
        
        # Add offsets: [window_size, 2, E]
        edge_index_batched = (edge_expanded + offsets).reshape(2, -1)  # [2, E*window_size]
        
        # Process all timesteps in ONE GCN call (much faster!)
        h_batched = self.gcn(x_batched, edge_index_batched)  # [N*window_size, hidden_dim]
        h_batched = torch.relu(h_batched)
        
        # Reshape back: [N*window_size, hidden_dim] -> [window_size, N, hidden_dim]
        gcn_sequence = h_batched.reshape(window_size, N, self.hidden_dim)  # [T, N, hidden_dim]
        
        # LSTM expects [seq_len, batch, features]
        # We have [T, N, hidden_dim] which is [seq_len, batch=N, features]
        # Ensure all tensors stay on same device (MPS optimization)
        lstm_out, (h_n, c_n) = self.lstm(gcn_sequence)  # lstm_out: [T, N, lstm_hidden]
        
        # Use the last LSTM output (most recent timestep)
        # Use indexing instead of slicing for better MPS performance
        lstm_final = lstm_out[-1]  # [N, lstm_hidden]
        
        # Predict delta
        delta = self.predictor(lstm_final)  # [N, 1]
        
        # RESIDUAL: Output = Last input + Delta
        out = x_t + delta
        
        return out


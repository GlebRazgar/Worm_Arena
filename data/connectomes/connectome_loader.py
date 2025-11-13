"""
Connectome Loader for C. elegans
Config-driven loader for various connectome datasets
"""

from datasets import load_dataset
import torch
from torch_geometric.data import Data
import pandas as pd
import sys
from pathlib import Path

# Add configs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import get_config


def load_connectome(connectome=None, verbose=True):
    """
    Load connectome from HuggingFace dataset.
    
    Args:
        connectome: Which connectome to load. If None, uses config/data.yaml
                   Options: 'cook2019', 'witvliet2020_7', 'white1986', etc.
        verbose: Whether to print loading information
    
    Returns:
        torch_geometric.data.Data: Graph with:
            - x: Node features [num_nodes, 3] (one-hot encoded type)
            - edge_index: Edge connections [2, num_edges]
            - edge_attr: Edge weights [num_edges, 2] (gap junction, chemical synapse)
            - pos: 3D positions [num_nodes, 3]
            - node_label: List of neuron names
            - node_type: List of neuron types
            - node_class: Integer class labels
            - connectome_name: Name of the connectome dataset
    """
    # Load connectome name from config if not specified
    if connectome is None:
        config = get_config()
        connectome = config.connectome
    
    if verbose:
        print(f"Loading {connectome} connectome...")
    
    # Load dataset
    ds = load_dataset("qsimeon/celegans_connectome_data", split='train_test')
    df = pd.DataFrame(ds)
    
    # Filter for specified connectome
    df = df[df['data_sources'].astype(str).str.contains(connectome, na=False)]
    
    if verbose:
        print(f"  Found {len(df)} edges")
    
    # Get unique neurons
    neurons = pd.concat([df['from_neuron'], df['to_neuron']]).unique()
    neurons = sorted(neurons)
    neuron2idx = {n: i for i, n in enumerate(neurons)}
    
    if verbose:
        print(f"  Found {len(neurons)} neurons")
    
    # Build edges
    edge_index = torch.tensor([
        [neuron2idx[row['from_neuron']], neuron2idx[row['to_neuron']]]
        for _, row in df.iterrows()
    ], dtype=torch.long).t()
    
    # Edge attributes: [gap_junction, chemical_synapse]
    edge_attr = torch.tensor([
        [row['mean_gap_weight'] if pd.notna(row['mean_gap_weight']) else 0.0,
         row['mean_chem_weight'] if pd.notna(row['mean_chem_weight']) else 0.0]
        for _, row in df.iterrows()
    ], dtype=torch.float)
    
    # Build node info dictionary
    node_info = {}
    for _, row in df.iterrows():
        for neuron, pos, ntype in [(row['from_neuron'], row['from_pos'], row['from_type']),
                                     (row['to_neuron'], row['to_pos'], row['to_type'])]:
            if neuron not in node_info:
                node_info[neuron] = {'pos': pos, 'type': ntype}
    
    # Node positions and types
    pos_list = []
    type_list = []
    for neuron in neurons:
        info = node_info.get(neuron, {'pos': '(0, 0, 0)', 'type': 'inter'})
        pos_list.append(eval(info['pos']) if isinstance(info['pos'], str) else info['pos'])
        type_list.append(info['type'])
    
    pos = torch.tensor(pos_list, dtype=torch.float)
    
    # Node features: one-hot encoded type
    type2class = {'sensory': 0, 'inter': 1, 'motor': 2}
    node_class = torch.tensor([type2class.get(t, 1) for t in type_list], dtype=torch.long)
    x = torch.nn.functional.one_hot(node_class, num_classes=3).float()
    
    # Create graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        node_label=neurons,
        node_type=type_list,
        node_class=node_class,
        num_classes=3,
        connectome_name=connectome  # Store connectome name
    )
    
    if verbose:
        print(f"\nGraph Summary:")
        print(f"  Connectome: {connectome}")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.num_edges}")
        print(f"  Node features: {graph.x.shape}")
        print(f"  Edge features: {graph.edge_attr.shape}")
    
    return graph


def save_graph(graph, filename):
    """Save graph to disk for fast loading"""
    torch.save(graph, filename)
    print(f"Saved graph to {filename}")


def load_graph(filename):
    """Load graph from disk"""
    return torch.load(filename)


if __name__ == "__main__":
    # Test the loader
    print("Testing config-driven loader:")
    print("=" * 60)
    
    # Load from config
    graph = load_connectome()
    
    print("\nNode types distribution:")
    from collections import Counter
    type_counts = Counter(graph.node_type)
    for ntype, count in type_counts.items():
        print(f"  {ntype}: {count}")
    
    print("\nEdge weight statistics:")
    print(f"  Gap junction: {graph.edge_attr[:, 0].sum():.1f} total")
    print(f"  Chemical: {graph.edge_attr[:, 1].sum():.1f} total")
"""
Connectome Visualization Utilities
Simple visualization for C. elegans connectomes
"""

import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
from pathlib import Path


def plot_connectome(graph, plot_3d=False, output_dir="plots", filename=None):
    """
    Visualize the connectome graph.
    
    Args:
        graph: PyTorch Geometric Data object containing the connectome
        plot_3d: If True, create 3D visualization; otherwise 2D
        output_dir: Directory to save the plot
        filename: Name of the output file (auto-generated if None)
    """
    # Get connectome name from graph if available
    connectome_name = getattr(graph, 'connectome_name', 'connectome')
    
    # Auto-generate filename if not provided
    if filename is None:
        filename = f"connectome-{connectome_name}.png"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {'3D' if plot_3d else '2D'} connectome visualization...")
    
    # Convert to NetworkX for visualization
    G = nx.DiGraph()
    
    # Add nodes
    num_nodes = graph.num_nodes
    for i in range(num_nodes):
        node_label = graph.node_label[i] if hasattr(graph, 'node_label') else str(i)
        node_type = graph.node_type[i] if hasattr(graph, 'node_type') else 'unknown'
        G.add_node(i, label=node_label, type=node_type)
    
    # Add edges
    edge_index = graph.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i]
        target = edge_index[1, i]
        
        # Get edge weights if available
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            gap_weight = graph.edge_attr[i, 0].item()
            chem_weight = graph.edge_attr[i, 1].item()
            total_weight = gap_weight + chem_weight
            G.add_edge(source, target, weight=total_weight, gap=gap_weight, chem=chem_weight)
        else:
            G.add_edge(source, target)
    
    # Color nodes by type
    node_colors = []
    color_map = {'sensory': '#FF6B6B', 'inter': '#4ECDC4', 'motor': '#45B7D1', 'unknown': '#95A5A6'}
    
    for i in range(num_nodes):
        node_type = graph.node_type[i] if hasattr(graph, 'node_type') else 'unknown'
        node_colors.append(color_map.get(node_type, color_map['unknown']))
    
    # Create the plot
    if plot_3d:
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use actual 3D positions
        pos_3d = graph.pos.cpu().numpy()
        
        # Plot edges
        for edge in G.edges():
            source, target = edge
            xs = [pos_3d[source, 0], pos_3d[target, 0]]
            ys = [pos_3d[source, 1], pos_3d[target, 1]]
            zs = [pos_3d[source, 2], pos_3d[target, 2]]
            ax.plot(xs, ys, zs, 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot nodes
        ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], 
                  c=node_colors, s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('X Position (Left-Right)', fontsize=10)
        ax.set_ylabel('Y Position (Anterior-Posterior)', fontsize=10)
        ax.set_zlabel('Z Position (Dorsal-Ventral)', fontsize=10)
        ax.set_title(f'C. elegans Connectome - {connectome_name.upper()} (3D)\nNodes: {num_nodes}, Edges: {G.number_of_edges()}', 
                    fontsize=14, fontweight='bold')
        
        # Set aspect ratio to show worm-like elongated shape
        x_range = pos_3d[:, 0].max() - pos_3d[:, 0].min()
        y_range = pos_3d[:, 1].max() - pos_3d[:, 1].min()
        z_range = pos_3d[:, 2].max() - pos_3d[:, 2].min()
        
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
        
    else:
        # Calculate aspect ratio based on actual data
        pos_array = graph.pos.cpu().numpy()
        y_range = pos_array[:, 1].max() - pos_array[:, 1].min()
        x_range = pos_array[:, 0].max() - pos_array[:, 0].min()
        
        # Create figure with aspect ratio matching the worm's elongated shape
        aspect_ratio = y_range / x_range if x_range > 0 else 1
        fig_width = 20
        fig_height = max(6, min(30, fig_width * aspect_ratio / 3))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Use 2D projection (X and Y coordinates from 3D position)
        pos_2d = {}
        for i in range(num_nodes):
            pos_2d[i] = (pos_array[i, 0], pos_array[i, 1])
        
        # Draw the graph
        nx.draw_networkx_edges(G, pos_2d, alpha=0.2, width=0.5, 
                              arrows=True, arrowsize=5, arrowstyle='->', 
                              edge_color='gray', ax=ax)
        
        nx.draw_networkx_nodes(G, pos_2d, node_color=node_colors, 
                              node_size=100, alpha=0.8, 
                              edgecolors='black', linewidths=0.5, ax=ax)
        
        # Add labels for a subset of important neurons (to avoid clutter)
        if hasattr(graph, 'node_label'):
            label_dict = {i: graph.node_label[i] for i in range(min(50, num_nodes))}
            nx.draw_networkx_labels(G, pos_2d, label_dict, font_size=6, ax=ax)
        
        ax.set_title(f'C. elegans Connectome - {connectome_name.upper()} (2D Lateral View)\nNodes: {num_nodes}, Edges: {G.number_of_edges()}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position (Left-Right, μm)', fontsize=12)
        ax.set_ylabel('Y Position (Anterior-Posterior, μm)', fontsize=12)
        ax.axis('on')
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio to preserve true shape
        ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['sensory'], 
                   markersize=10, label='Sensory'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['inter'], 
                   markersize=10, label='Interneuron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['motor'], 
                   markersize=10, label='Motor')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = output_path / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    # Test visualization
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.connectomes.connectome_loader import load_connectome
    
    print("Loading connectome...")
    graph = load_connectome()
    
    print("\nCreating visualizations...")
    plot_connectome(graph, plot_3d=False, output_dir="plots")
    plot_connectome(graph, plot_3d=True, output_dir="plots")
    
    print("\n✓ Visualizations complete!")

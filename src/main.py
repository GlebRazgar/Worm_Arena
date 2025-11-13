"""
Main entry point for Worm Arena
Simple one-liner visualizations - comment out what you don't need
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from tests.connectome_vis import plot_connectome
from tests.functional_vis import plot_sample_traces


if __name__ == "__main__":
    # Load connectome from config
    graph = load_connectome()
    
    # Connectome visualization (comment out to disable)
    plot_connectome(graph, plot_3d=False, output_dir="tests/plots")
    
    # Functional visualization (comment out to disable)
    df = load_functional_data(connectome=graph, verbose=False)
    plot_sample_traces(df, num_neurons=6, output_dir="tests/plots")
    
    print("\n✓ Complete! Check tests/plots/ for visualizations.")

"""
Main entry point for Worm Arena
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from tests.visualisation import plot_connectome


if __name__ == "__main__":
    # Load connectome from config
    graph = load_connectome()
    
    # Visualize (comment out if not needed)
    plot_connectome(graph, plot_3d=False, output_dir="tests/plots")

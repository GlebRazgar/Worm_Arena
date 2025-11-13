"""
Test functional data loading and create calcium trace visualizations
Uses source datasets specified in configs/data.yaml
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.functional.functional_loader import load_functional_data, dataframe_to_tensors
from data.connectomes.connectome_loader import load_connectome
from configs.config import get_config


def test_and_visualize():
    """Load functional data from config and create visualizations"""
    
    config = get_config()
    print("="*70)
    print(f"FUNCTIONAL DATA LOADING")
    print("="*70)
    print(f"Source datasets: {config.functional_sources}")
    print(f"Match connectome: {config.match_connectome}\n")
    
    # Load connectome for neuron matching
    print("Loading connectome...")
    connectome = load_connectome(verbose=False)
    print(f"  Connectome has {connectome.num_nodes} neurons\n")
    
    # Load functional data
    df = load_functional_data(connectome=connectome)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    plot_sample_traces(df, num_neurons=6)
    plot_cross_dataset_comparison(df)
    plot_same_neuron_different_recordings(df)
    plot_neuron_coverage(df, connectome)
    plot_recording_statistics(df)
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"  Check plots/ directory for functional_*.png files")
    
    # Test tensor conversion
    print("\n" + "="*70)
    print("TESTING TENSOR CONVERSION")
    print("="*70)
    test_tensor_conversion(df)


def plot_sample_traces(df, num_neurons=6, output_dir="plots"):
    """Plot sample calcium traces from DIVERSE recordings (different worms/datasets)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select diverse samples from different worms and datasets
    diverse_samples = []
    
    # Try to get samples from different source datasets
    for source in df['source_dataset'].unique():
        source_df = df[df['source_dataset'] == source]
        
        # Get samples from different worms within this source
        for worm in source_df['worm'].unique()[:2]:  # Max 2 worms per source
            worm_df = source_df[source_df['worm'] == worm]
            if len(worm_df) > 0:
                diverse_samples.append(worm_df.iloc[0])
                if len(diverse_samples) >= num_neurons:
                    break
        if len(diverse_samples) >= num_neurons:
            break
    
    # If we don't have enough, just sample randomly
    if len(diverse_samples) < num_neurons:
        diverse_samples = df.sample(n=min(num_neurons, len(df))).to_dict('records')
    
    fig, axes = plt.subplots(len(diverse_samples), 1, figsize=(14, 2.5*len(diverse_samples)))
    if len(diverse_samples) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(diverse_samples)))
    
    for idx, (sample, ax) in enumerate(zip(diverse_samples, axes)):
        calcium = sample['original_calcium_data']
        time = sample['original_time_in_seconds'] if 'original_time_in_seconds' in sample else np.arange(len(calcium))
        
        ax.plot(time, calcium, color=colors[idx], linewidth=1.5, alpha=0.8)
        ax.set_ylabel('ΔF/F', fontsize=10)
        ax.set_title(f'{sample["neuron"]} ({sample["source_dataset"]}, {sample["worm"]})', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add info box with more metadata
        info_text = f"Length: {len(calcium)} pts\nFile: {sample['raw_data_file'][:20]}..."
        ax.text(0.98, 0.95, info_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=7)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    
    plt.suptitle('Sample Calcium Traces from DIVERSE Recordings\n(Different worms and datasets)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Get source names for filename
    sources = '_'.join(sorted(df['source_dataset'].unique())[:3])
    output_file = output_path / f"functional_{sources}_diverse_traces.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved diverse calcium traces to: {output_file}")
    plt.close()


def plot_cross_dataset_comparison(df, output_dir="plots"):
    """Compare activity patterns across different source datasets"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pick a neuron that appears in all datasets
    neuron_counts = df.groupby('neuron')['source_dataset'].nunique()
    multi_dataset_neurons = neuron_counts[neuron_counts >= 2].index.tolist()
    
    if len(multi_dataset_neurons) == 0:
        print("No neurons found in multiple datasets, skipping cross-dataset comparison")
        return
    
    # Pick a common neuron
    neuron = multi_dataset_neurons[0]
    neuron_df = df[df['neuron'] == neuron]
    
    # Get one sample from each dataset
    fig, axes = plt.subplots(len(neuron_df['source_dataset'].unique()), 1, 
                            figsize=(14, 3*len(neuron_df['source_dataset'].unique())))
    if len(neuron_df['source_dataset'].unique()) == 1:
        axes = [axes]
    
    colors = {'Venkatachalam2024': '#FF6B6B', 'Flavell2023': '#4ECDC4', 'Leifer2023': '#45B7D1'}
    
    for idx, (source, ax) in enumerate(zip(sorted(neuron_df['source_dataset'].unique()), axes)):
        source_data = neuron_df[neuron_df['source_dataset'] == source].iloc[0]
        
        calcium = source_data['original_calcium_data']
        time = source_data['original_time_in_seconds'] if 'original_time_in_seconds' in source_data else np.arange(len(calcium))
        
        color = colors.get(source, '#95A5A6')
        ax.plot(time, calcium, color=color, linewidth=1.5, alpha=0.8)
        ax.set_ylabel('ΔF/F', fontsize=11)
        ax.set_title(f'{neuron} - {source} (worm: {source_data["worm"]}, file: {source_data["raw_data_file"][:30]}...)', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Stats
        mean_activity = np.mean(calcium)
        std_activity = np.std(calcium)
        ax.axhline(mean_activity, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.02, 0.95, f'Mean: {mean_activity:.2f}±{std_activity:.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=9)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    plt.suptitle(f'Cross-Dataset Comparison: Same Neuron ({neuron}) in Different Studies', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    sources = '_'.join(sorted(df['source_dataset'].unique())[:3])
    output_file = output_path / f"functional_{sources}_cross_dataset.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved cross-dataset comparison to: {output_file}")
    plt.close()


def plot_same_neuron_different_recordings(df, output_dir="plots"):
    """Show same neuron from different recordings within same dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find a neuron with multiple recordings in same dataset
    for source in df['source_dataset'].unique():
        source_df = df[df['source_dataset'] == source]
        neuron_counts = source_df.groupby('neuron')['worm'].nunique()
        multi_recording_neurons = neuron_counts[neuron_counts >= 3].index.tolist()
        
        if len(multi_recording_neurons) > 0:
            neuron = multi_recording_neurons[0]
            neuron_df = source_df[source_df['neuron'] == neuron]
            
            # Get up to 4 different worms
            worms = neuron_df['worm'].unique()[:4]
            
            fig, axes = plt.subplots(len(worms), 1, figsize=(14, 2.5*len(worms)))
            if len(worms) == 1:
                axes = [axes]
            
            for idx, (worm, ax) in enumerate(zip(worms, axes)):
                worm_data = neuron_df[neuron_df['worm'] == worm].iloc[0]
                
                calcium = worm_data['original_calcium_data']
                time = worm_data['original_time_in_seconds'] if 'original_time_in_seconds' in worm_data else np.arange(len(calcium))
                
                ax.plot(time, calcium, color=f'C{idx}', linewidth=1.5, alpha=0.8)
                ax.set_ylabel('ΔF/F', fontsize=10)
                ax.set_title(f'{neuron} - Worm {worm} (Recording: {worm_data["raw_data_file"][:40]}...)', 
                            fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = np.mean(calcium)
                max_val = np.max(calcium)
                ax.text(0.02, 0.95, f'Mean: {mean_val:.1f}, Max: {max_val:.1f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=8)
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=11)
            plt.suptitle(f'Same Neuron ({neuron}), Different Worms - {source}\nShowing recording variability', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            output_file = output_path / f"functional_{source}_same_neuron_variability.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved same-neuron variability plot to: {output_file}")
            plt.close()
            break  # Only do this for first dataset with enough samples


def plot_neuron_coverage(df, connectome, output_dir="plots"):
    """Plot which connectome neurons have functional data"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get neurons with functional data
    functional_neurons = set(df['neuron'].unique())
    connectome_neurons = set(connectome.node_label)
    
    # Categorize
    matched = functional_neurons & connectome_neurons
    functional_only = functional_neurons - connectome_neurons
    connectome_only = connectome_neurons - functional_neurons
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Venn diagram style bar chart
    categories = ['Connectome\nOnly', 'Both\n(Matched)', 'Functional\nOnly']
    counts = [len(connectome_only), len(matched), len(functional_only)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Neurons', fontsize=12)
    ax1.set_title('Neuron Coverage: Connectome vs Functional Data', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Neuron type distribution for matched neurons
    matched_df = df[df['neuron'].isin(matched)]
    
    # Map neurons to types from connectome
    neuron_type_map = {label: ntype for label, ntype in zip(connectome.node_label, connectome.node_type)}
    matched_types = [neuron_type_map.get(n, 'unknown') for n in matched_df['neuron'].unique()]
    
    from collections import Counter
    type_counts = Counter(matched_types)
    
    types = list(type_counts.keys())
    type_values = [type_counts[t] for t in types]
    type_colors = {'sensory': '#FF6B6B', 'inter': '#4ECDC4', 'motor': '#45B7D1', 
                   'pharyngeal': '#95A5A6', 'unknown': '#BDC3C7'}
    colors_for_types = [type_colors.get(t, '#95A5A6') for t in types]
    
    ax2.bar(types, type_values, color=colors_for_types, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Neurons', fontsize=12)
    ax2.set_xlabel('Neuron Type', fontsize=12)
    ax2.set_title('Matched Neurons by Type', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (t, v) in enumerate(zip(types, type_values)):
        ax2.text(i, v, f'{v}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    sources = '_'.join(sorted(df['source_dataset'].unique())[:3])
    output_file = output_path / f"functional_{sources}_coverage.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved coverage analysis to: {output_file}")
    plt.close()


def plot_recording_statistics(df, output_dir="plots"):
    """Plot statistics about recording lengths and sources"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate sequence lengths
    df['seq_length'] = df['original_calcium_data'].apply(len)
    df['duration'] = df['original_time_in_seconds'].apply(lambda x: x[-1] if len(x) > 0 else 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sequence length distribution
    axes[0, 0].hist(df['seq_length'], bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Sequence Length (time points)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Recording Lengths', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Duration distribution
    axes[0, 1].hist(df['duration'], bins=50, color='#45B7D1', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Duration (seconds)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Recording Durations', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Samples per source dataset
    source_counts = df['source_dataset'].value_counts()
    axes[1, 0].barh(source_counts.index, source_counts.values, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Number of Samples', fontsize=11)
    axes[1, 0].set_title('Samples per Source Dataset', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Labeled vs unlabeled neurons
    label_counts = df['is_labeled_neuron'].value_counts()
    labels = ['Labeled', 'Unlabeled']
    values = [label_counts.get(True, 0), label_counts.get(False, 0)]
    axes[1, 1].pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                  colors=['#4ECDC4', '#FF6B6B'], explode=(0.05, 0))
    axes[1, 1].set_title('Neuron Labeling Confidence', fontsize=12, fontweight='bold')
    
    plt.suptitle('Functional Data Recording Statistics', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    sources = '_'.join(sorted(df['source_dataset'].unique())[:3])
    output_file = output_path / f"functional_{sources}_statistics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved statistics to: {output_file}")
    plt.close()


def test_tensor_conversion(df):
    """Test converting DataFrame to PyTorch tensors"""
    # Take a subset for testing
    test_df = df.head(100)
    
    print("\nConverting sample to tensors...")
    tensors = dataframe_to_tensors(test_df)
    
    print(f"  Data tensor shape: {tensors['data'].shape}")
    print(f"  Mask tensor shape: {tensors['mask'].shape}")
    print(f"  Time tensor shape: {tensors['time'].shape if 'time' in tensors else 'N/A'}")
    print(f"  Number of unique neurons: {len(tensors['unique_neurons'])}")
    print(f"  Average sequence length: {tensors['lengths'].float().mean():.1f}")
    print(f"  Min/Max sequence length: {tensors['lengths'].min()}/{tensors['lengths'].max()}")
    print("\n✓ Tensor conversion successful!")


if __name__ == "__main__":
    test_and_visualize()


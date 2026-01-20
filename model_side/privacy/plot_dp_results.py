"""
Plot privacy-utility trade-off curves from DP experiment results.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path


def load_results(results_dir='results/stage2'):
    """Load experiment results from JSON file."""
    results_path = os.path.join(results_dir, 'dp_privacy_utility.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_privacy_utility_tradeoff(results, output_dir='results/stage2'):
    """
    Create privacy-utility trade-off plot.
    
    Args:
        results: List of dicts with 'epsilon' and 'test_accuracy' keys
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data, handling 'inf' as a special case
    epsilons = []
    accuracies = []
    
    for r in results:
        eps = r['epsilon']
        if eps == 'inf' or eps == float('inf'):
            continue  # We'll add this separately
        epsilons.append(float(eps))
        accuracies.append(r['test_accuracy'])
    
    # Sort by epsilon
    sorted_pairs = sorted(zip(epsilons, accuracies))
    epsilons = [p[0] for p in sorted_pairs]
    accuracies = [p[1] for p in sorted_pairs]
    
    # Get non-DP baseline if available
    baseline_acc = None
    for r in results:
        if r['epsilon'] == 'inf' or r['epsilon'] == float('inf'):
            baseline_acc = r['test_accuracy']
            break
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot DP results
    ax.plot(epsilons, accuracies, 'bo-', markersize=10, linewidth=2, label='DP Training')
    
    # Add baseline if available
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, 
                   label=f'No DP (ε=∞): {baseline_acc:.4f}')
    
    # Formatting
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Privacy-Utility Trade-off', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add annotations for each point
    for eps, acc in zip(epsilons, accuracies):
        ax.annotate(f'{acc:.3f}', (eps, acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # Save
    output_path = os.path.join(output_dir, 'privacy_utility_tradeoff.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()
    
    return output_path


def plot_training_curves(results_dir='results/stage2', output_dir='results/stage2'):
    """
    Plot training curves for each epsilon value.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find all history files
    history_files = list(Path(results_dir).glob('dp_epsilon_*_history.json'))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(history_files)))
    
    for i, history_path in enumerate(sorted(history_files)):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Extract epsilon from filename
        eps_str = history_path.stem.replace('dp_epsilon_', '').replace('_history', '')
        label = f'ε={eps_str}' if eps_str != 'inf' else 'ε=∞ (No DP)'
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        axes[0].plot(epochs, history['train_loss'], color=colors[i], 
                     linewidth=2, label=label)
        
        # Plot accuracy
        axes[1].plot(epochs, history['train_acc'], color=colors[i], 
                     linewidth=2, label=label)
    
    # Format loss plot
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss by Privacy Budget', fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Format accuracy plot
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy by Privacy Budget', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dp_training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    
    plt.close()
    
    return output_path


def generate_report_table(results, output_dir='results/stage2'):
    """Generate a markdown table for the report."""
    os.makedirs(output_dir, exist_ok=True)
    
    lines = [
        "# Privacy-Utility Trade-off Results",
        "",
        "| Epsilon (ε) | Train Accuracy | Test Accuracy | Privacy Guarantee |",
        "|-------------|----------------|---------------|-------------------|"
    ]
    
    for r in sorted(results, key=lambda x: float('inf') if x['epsilon'] == 'inf' else float(x['epsilon'])):
        eps = r['epsilon']
        eps_display = '∞ (No DP)' if eps == 'inf' else f'{eps}'
        privacy = 'None' if eps == 'inf' else f'({eps}, 1e-5)-DP'
        lines.append(f"| {eps_display} | {r['final_train_acc']:.4f} | {r['test_accuracy']:.4f} | {privacy} |")
    
    table_content = '\n'.join(lines)
    
    output_path = os.path.join(output_dir, 'privacy_utility_table.md')
    with open(output_path, 'w') as f:
        f.write(table_content)
    
    print(f"Report table saved to {output_path}")
    print("\n" + table_content)
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot DP experiment results')
    parser.add_argument('--results_dir', type=str, default='results/stage2',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/stage2',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    try:
        results = load_results(args.results_dir)
        plot_privacy_utility_tradeoff(results, args.output_dir)
        plot_training_curves(args.results_dir, args.output_dir)
        generate_report_table(results, args.output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run dp_experiments.py first to generate results.")

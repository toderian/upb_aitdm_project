"""
Plot FL+DP Simulation Results
Generates visualizations for the report and presentation.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results():
    """Load all simulation results."""
    eps_values = [1.0, 5.0, 10.0]
    results = {}
    for eps in eps_values:
        path = f'results/fl_dp_sim/fl_dp_simulation_eps{eps}.json'
        if os.path.exists(path):
            with open(path) as f:
                results[eps] = json.load(f)
    return results

def plot_all_results(results, output_dir='results/fl_dp_sim'):
    """Generate all visualizations."""
    eps_values = list(results.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Global Accuracy per Round (all epsilons)
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, eps in enumerate(eps_values):
        rounds = results[eps]['history']['rounds']
        acc = results[eps]['history']['global_accuracy']
        ax1.plot(rounds, acc, 'o-', label=f'ε = {eps}', linewidth=2, 
                 markersize=8, color=colors[i])
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Global Test Accuracy', fontsize=12)
    ax1.set_title('Global Model Accuracy Over FL Rounds', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.45, 0.55)
    
    # 2. Client Training Accuracy (for eps=5.0)
    ax2 = axes[0, 1]
    eps = 5.0
    client_metrics = results[eps]['history']['client_metrics']
    client_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    for client_id in range(3):
        client_acc = [round_data[client_id]['acc'] for round_data in client_metrics]
        ax2.plot(range(1, 6), client_acc, 'o-', label=f'Client {client_id}', 
                 linewidth=2, markersize=8, color=client_colors[client_id])
    ax2.set_xlabel('Federated Round', fontsize=12)
    ax2.set_ylabel('Client Training Accuracy', fontsize=12)
    ax2.set_title(f'Per-Client Training Accuracy (ε = {eps})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Client Training Loss (for eps=5.0)
    ax3 = axes[1, 0]
    for client_id in range(3):
        client_loss = [round_data[client_id]['loss'] for round_data in client_metrics]
        ax3.plot(range(1, 6), client_loss, 'o-', label=f'Client {client_id}', 
                 linewidth=2, markersize=8, color=client_colors[client_id])
    ax3.set_xlabel('Federated Round', fontsize=12)
    ax3.set_ylabel('Client Training Loss', fontsize=12)
    ax3.set_title(f'Per-Client Training Loss (ε = {eps})', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Privacy-Utility Trade-off Bar Chart
    ax4 = axes[1, 1]
    final_acc = [results[eps]['final_accuracy'] for eps in eps_values]
    final_eps = [results[eps]['final_max_epsilon'] for eps in eps_values]
    
    x = np.arange(len(eps_values))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, final_acc, width, label='Final Accuracy', color='steelblue')
    ax4.set_ylabel('Accuracy', color='steelblue', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4.set_ylim(0, 1)
    
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, final_eps, width, label='Epsilon Spent', color='coral')
    ax4_twin.set_ylabel('Privacy Budget (ε)', color='coral', fontsize=12)
    ax4_twin.tick_params(axis='y', labelcolor='coral')
    
    ax4.set_xlabel('Target Epsilon', fontsize=12)
    ax4.set_title('Privacy-Utility Trade-off Summary', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'ε = {e}' for e in eps_values])
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fl_dp_visualization.png', dpi=150, bbox_inches='tight')
    print(f'✅ Saved: {output_dir}/fl_dp_visualization.png')
    
    # Training progress plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    for i, eps in enumerate(eps_values):
        client_metrics = results[eps]['history']['client_metrics']
        avg_acc = []
        for round_data in client_metrics:
            avg = np.mean([c['acc'] for c in round_data])
            avg_acc.append(avg)
        ax.plot(range(1, 6), avg_acc, 'o-', label=f'ε = {eps}', 
                linewidth=2, markersize=10, color=colors[i])
    
    ax.set_xlabel('Federated Round', fontsize=14)
    ax.set_ylabel('Average Client Training Accuracy', fontsize=14)
    ax.set_title('FL + Differential Privacy: Training Progress', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fl_dp_training_progress.png', dpi=150, bbox_inches='tight')
    print(f'✅ Saved: {output_dir}/fl_dp_training_progress.png')
    
    plt.close('all')

if __name__ == '__main__':
    results = load_results()
    if results:
        plot_all_results(results)
        print("\n📊 All visualizations generated successfully!")
    else:
        print("No results found to plot.")

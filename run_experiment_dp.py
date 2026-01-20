#!/usr/bin/env python
"""
Unified DP Experiment Runner with Results Output.

Usage:
    python run_experiment_dp.py --epochs 100
    python run_experiment_dp.py --epochs 50 --batch_size 16 --epsilons 1.0 5.0 10.0
    python run_experiment_dp.py --epochs 20 --cpu  # Force CPU
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_side.data.data_loader_enhanced import *
from model_side.models.cnn_model import SimpleCNN
from model_side.privacy.dp_training import train_with_dp
from opacus.validators import ModuleValidator

from tqdm import tqdm
# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")


def train_standard(model, train_loader, device, epochs=10, lr=1e-3):
    """Standard training without DP (epsilon = infinity)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    history = {'train_loss': [], 'train_acc': [], 'epsilon': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['epsilon'].append(float('inf'))
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ε: ∞")
    
    return model.state_dict(), history


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total


def plot_results(results, output_dir):
    """Generate and save plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    # Extract data
    epsilons = []
    accuracies = []
    baseline_acc = None
    
    for r in results:
        eps = r['epsilon']
        if eps == 'inf' or eps == float('inf'):
            baseline_acc = r['test_accuracy']
        else:
            epsilons.append(float(eps))
            accuracies.append(r['test_accuracy'])
    
    # Sort by epsilon
    sorted_pairs = sorted(zip(epsilons, accuracies))
    epsilons = [p[0] for p in sorted_pairs]
    accuracies = [p[1] for p in sorted_pairs]
    
    # Create trade-off plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilons, accuracies, 'bo-', markersize=10, linewidth=2, label='DP Training')
    
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, 
                   label=f'No DP (ε=∞): {baseline_acc:.4f}')
    
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Privacy-Utility Trade-off', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    for eps, acc in zip(epsilons, accuracies):
        ax.annotate(f'{acc:.3f}', (eps, acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'privacy_utility_tradeoff.png'), dpi=150)
    plt.close()
    print(f"Plot saved: {output_dir}/privacy_utility_tradeoff.png")


def plot_training_curves(output_dir):
    """Plot training curves from history files."""
    if not HAS_MATPLOTLIB:
        return
    
    history_files = list(Path(output_dir).glob('dp_epsilon_*_history.json'))
    if not history_files:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(history_files)))
    
    for i, history_path in enumerate(sorted(history_files)):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        eps_str = history_path.stem.replace('dp_epsilon_', '').replace('_history', '')
        label = f'ε={eps_str}' if eps_str != 'inf' else 'ε=∞ (No DP)'
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], color=colors[i], linewidth=2, label=label)
        axes[1].plot(epochs, history['train_acc'], color=colors[i], linewidth=2, label=label)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss by Privacy Budget')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Accuracy')
    axes[1].set_title('Training Accuracy by Privacy Budget')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dp_training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved: {output_dir}/dp_training_curves.png")


def print_results_table(results):
    """Print results as a formatted table."""
    print("\n" + "=" * 70)
    print("PRIVACY-UTILITY TRADE-OFF RESULTS")
    print("=" * 70)
    print(f"{'Epsilon':<12} {'Train Acc':<12} {'Test Acc':<12} {'Privacy Guarantee':<20}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: float('inf') if x['epsilon'] == 'inf' else float(x['epsilon'])):
        eps = r['epsilon']
        eps_display = '∞ (No DP)' if eps == 'inf' else f'{eps}'
        privacy = 'None' if eps == 'inf' else f'({eps}, 1e-5)-DP'
        print(f"{eps_display:<12} {r['final_train_acc']:<12.4f} {r['test_accuracy']:<12.4f} {privacy:<20}")
    
    print("=" * 70)


def save_markdown_table(results, output_dir):
    """Save results as markdown table."""
    lines = [
        "# Privacy-Utility Trade-off Results\n",
        "| Epsilon (ε) | Train Accuracy | Test Accuracy | Privacy Guarantee |",
        "|-------------|----------------|---------------|-------------------|"
    ]
    
    for r in sorted(results, key=lambda x: float('inf') if x['epsilon'] == 'inf' else float(x['epsilon'])):
        eps = r['epsilon']
        eps_display = '∞ (No DP)' if eps == 'inf' else f'{eps}'
        privacy = 'None' if eps == 'inf' else f'({eps}, 1e-5)-DP'
        lines.append(f"| {eps_display} | {r['final_train_acc']:.4f} | {r['test_accuracy']:.4f} | {privacy} |")
    
    with open(os.path.join(output_dir, 'privacy_utility_table.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"Markdown table saved: {output_dir}/privacy_utility_table.md")


def main():
    parser = argparse.ArgumentParser(description='Run DP Experiments with Results Output')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes (default: 4)')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1.0, 5.0, 10.0],
                        help='Epsilon values to test (default: 1.0 5.0 10.0)')
    parser.add_argument('--include_baseline', action='store_true', default=True,
                        help='Include non-DP baseline (default: True)')
    parser.add_argument('--no_baseline', action='store_true', help='Skip non-DP baseline')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data (optional)')
    parser.add_argument('--output_dir', type=str, default='results/stage2', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--image_size', type=int, default=64, help='Image size for dummy data (default: 64)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of training samples for dummy data')
    args = parser.parse_args()
    
    # Handle baseline flag
    if args.no_baseline:
        args.include_baseline = False
    
    # Device selection
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load real data from COVIDx dataset
    from model_side.data.data_loader import COVIDxZipDataset
    from torchvision import transforms
    
    # Dataset paths - adjust if needed
    ZIP_FILE_PATH = "Dani/Dani/archive.zip"
    
    # Check if data exists
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"ERROR: {ZIP_FILE_PATH} not found!")
        print("Falling back to dummy data for testing...")
        X_train = torch.randn(args.num_samples, 3, args.image_size, args.image_size)
        y_train = torch.randint(0, args.num_classes, (args.num_samples,))
        X_test = torch.randn(args.num_samples // 5, 3, args.image_size, args.image_size)
        y_test = torch.randint(0, args.num_classes, (args.num_samples // 5,))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        print(f"Loading real COVIDx data from {ZIP_FILE_PATH}")
        
        # Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load full training dataset (all sources combined)
        # train_dataset = COVIDxZipDataset(
        #     zip_path=ZIP_FILE_PATH,
        #     txt_file="train.txt",
        #     source_filter=None,  # Use all sources for centralized training
        #     transform=transform
        # )
        
        # # Load test dataset
        # test_dataset = COVIDxZipDataset(
        #     zip_path=ZIP_FILE_PATH,
        #     txt_file="test.txt",
        #     source_filter=None,
        #     transform=transform
        # )
        
        # print(f"Train dataset size: {len(train_dataset)}")
        # print(f"Test dataset size: {len(test_dataset)}")
        
        # Update num_classes based on actual data (binary: positive/negative)
        args.num_classes = 2
        print(f"Using {args.num_classes} classes (binary: positive/negative)")
        
        train_loader = get_federated_client(0, batch_size=args.batch_size)
        test_loader = get_global_test_loader(batch_size=args.batch_size)
    
    # Build epsilon list
    epsilons = args.epsilons.copy()
    if args.include_baseline:
        epsilons.append(float('inf'))
    
    print(f"\nRunning experiments with:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epsilons: {[e if e != float('inf') else '∞' for e in epsilons]}")
    print()
    
    results = []
    
    for epsilon in epsilons:
        print(f"\n{'='*60}")
        print(f"Training with epsilon = {'∞' if epsilon == float('inf') else epsilon}")
        print('='*60)
        
        model = SimpleCNN(num_classes=args.num_classes)
        
        if epsilon == float('inf'):
            model_state, history = train_standard(model, train_loader, device, epochs=args.epochs)
        else:
            model_state, history = train_with_dp(
                model, train_loader, device,
                target_epsilon=epsilon,
                target_delta=1e-5,
                epochs=args.epochs,
                max_grad_norm=1.0
            )
        
        # Save model
        epsilon_str = 'inf' if epsilon == float('inf') else str(epsilon)
        model_path = f'models/dp_epsilon_{epsilon_str}.pth'
        torch.save(model_state, model_path)
        print(f"Model saved: {model_path}")
        
        # Evaluate
        eval_model = SimpleCNN(num_classes=args.num_classes)
        if epsilon != float('inf'):
            eval_model = ModuleValidator.fix(eval_model)
        eval_model.load_state_dict(model_state)
        eval_model = eval_model.to(device)
        test_acc = evaluate(eval_model, test_loader, device)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Record results
        results.append({
            'epsilon': epsilon if epsilon != float('inf') else 'inf',
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'test_accuracy': test_acc,
        })
        
        # Save history
        history_json = {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'epsilon': [e if e != float('inf') else 'inf' for e in history['epsilon']]
        }
        with open(f"{args.output_dir}/dp_epsilon_{epsilon_str}_history.json", 'w') as f:
            json.dump(history_json, f, indent=2)
    
    # Save results JSON
    with open(f"{args.output_dir}/dp_privacy_utility.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Output results
    print_results_table(results)
    save_markdown_table(results, args.output_dir)
    plot_results(results, args.output_dir)
    plot_training_curves(args.output_dir)
    
    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

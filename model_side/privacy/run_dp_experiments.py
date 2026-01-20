"""
Run Differential Privacy experiments with different epsilon values.
Trains models and saves results for privacy-utility trade-off analysis.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_side.models.cnn_model import SimpleCNN
from model_side.privacy.dp_training import train_with_dp


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
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ε: ∞")
    
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


def run_experiments(train_loader, test_loader, num_classes=4, epochs=10, 
                    epsilons=[1.0, 5.0, 10.0, float('inf')], output_dir='results/stage2'):
    """
    Run DP experiments with different epsilon values.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_classes: Number of output classes
        epochs: Number of training epochs
        epsilons: List of epsilon values to test (inf = no DP)
        output_dir: Directory to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    results = []
    
    for epsilon in epsilons:
        print(f"\n{'='*50}")
        print(f"Training with epsilon = {epsilon}")
        print('='*50)
        
        # Create fresh model
        model = SimpleCNN(num_classes=num_classes)
        
        if epsilon == float('inf'):
            # Standard training without DP
            model_state, history = train_standard(
                model, train_loader, device, epochs=epochs
            )
        else:
            # DP training
            model_state, history = train_with_dp(
                model, train_loader, device,
                target_epsilon=epsilon,
                target_delta=1e-5,
                epochs=epochs,
                max_grad_norm=1.0
            )
        
        # Save model
        epsilon_str = 'inf' if epsilon == float('inf') else str(epsilon)
        model_path = f'models/dp_epsilon_{epsilon_str}.pth'
        torch.save(model_state, model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate - need to use ModuleValidator.fix() for DP models to match architecture
        model = SimpleCNN(num_classes=num_classes)
        if epsilon != float('inf'):
            from opacus.validators import ModuleValidator
            model = ModuleValidator.fix(model)
        model.load_state_dict(model_state)
        model = model.to(device)
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Record results
        results.append({
            'epsilon': epsilon if epsilon != float('inf') else 'inf',
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'test_accuracy': test_acc,
            'final_epsilon': history['epsilon'][-1] if epsilon != float('inf') else 'inf'
        })
        
        # Save individual history
        history_path = f'{output_dir}/dp_epsilon_{epsilon_str}_history.json'
        # Convert inf to string for JSON serialization
        history_json = {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'epsilon': [e if e != float('inf') else 'inf' for e in history['epsilon']]
        }
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
    
    # Save summary results
    results_path = f'{output_dir}/dp_privacy_utility.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Privacy-Utility Trade-off Summary")
    print('='*60)
    print(f"{'Epsilon':<12} {'Train Acc':<12} {'Test Acc':<12}")
    print('-'*36)
    for r in results:
        eps_str = '∞' if r['epsilon'] == 'inf' else f"{r['epsilon']}"
        print(f"{eps_str:<12} {r['final_train_acc']:.4f}       {r['test_accuracy']:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    from torch.utils.data import TensorDataset
    
    parser = argparse.ArgumentParser(description='Run DP experiments')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (2 for binary: positive/negative)')
    parser.add_argument('--use_real_data', action='store_true', help='Use real COVIDx data from zip file')
    parser.add_argument('--client_id', type=int, default=0, help='Client ID for federated data (0, 1, or 2)')
    parser.add_argument('--output_dir', type=str, default='results/stage2', help='Output directory')
    args = parser.parse_args()
    
    if args.use_real_data:
        print("Loading real COVIDx data...")
        try:
            from model_side.data.data_loader_enhanced import get_federated_client, get_global_test_loader
            
            # Get training data for specified client
            train_loader = get_federated_client(args.client_id, batch_size=args.batch_size)
            
            # Get global test set for evaluation
            test_loader = get_global_test_loader(batch_size=args.batch_size)
            
            print(f"Data loaded successfully!")
            print(f"  Training: Client {args.client_id}")
            print(f"  Testing: Global test set")
            
        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Make sure the archive.zip file exists at Dani/Dani/archive.zip")
            sys.exit(1)
    else:
        print("Using dummy data for testing (use --use_real_data for real experiments)...")
        print("WARNING: Dummy data will not produce meaningful results!")
        # Use smaller images (64x64) to reduce memory usage
        X_train = torch.randn(500, 3, 64, 64)  # Increased from 100
        y_train = torch.randint(0, args.num_classes, (500,))
        X_test = torch.randn(100, 3, 64, 64)   # Increased from 20
        y_test = torch.randint(0, args.num_classes, (100,))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    run_experiments(
        train_loader, test_loader,
        num_classes=args.num_classes,
        epochs=args.epochs,
        epsilons=[1.0, 5.0, 10.0, float('inf')],
        output_dir=args.output_dir
    )

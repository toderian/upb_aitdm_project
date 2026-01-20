"""
Simulated Federated Learning with Differential Privacy Test.
Uses synthetic data for quick testing of the FL+DP pipeline.

This demonstrates the concept without waiting for full dataset loading.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import accuracy_score, classification_report, f1_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_side.models.cnn_model import SimpleCNN


def create_synthetic_client_data(client_id, num_samples=1000, img_size=64):
    """Create synthetic data simulating non-IID distribution."""
    np.random.seed(42 + client_id)
    
    # Simulate non-IID: different clients have different class distributions
    if client_id == 0:
        # Client 0: More negative samples (hospital with many healthy patients)
        pos_ratio = 0.3
    elif client_id == 1:
        # Client 1: More positive samples (COVID specialty hospital)
        pos_ratio = 0.7
    else:
        # Client 2: Balanced
        pos_ratio = 0.5
    
    num_pos = int(num_samples * pos_ratio)
    num_neg = num_samples - num_pos
    
    # Create synthetic images (random noise with slight pattern based on label)
    X_pos = torch.randn(num_pos, 3, img_size, img_size) * 0.5 + 0.3
    X_neg = torch.randn(num_neg, 3, img_size, img_size) * 0.5 - 0.3
    
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).long()
    
    # Shuffle
    perm = torch.randperm(num_samples)
    X, y = X[perm], y[perm]
    
    return X, y


class FLDPSimulator:
    """Simulates Federated Learning with Differential Privacy."""
    
    def __init__(self, num_clients=3, epsilon=5.0, delta=1e-5, 
                 samples_per_client=1000, batch_size=32, output_dir='results/fl_dp_sim'):
        self.num_clients = num_clients
        self.epsilon = epsilon
        self.delta = delta
        self.samples_per_client = samples_per_client
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models/fl_dp_sim', exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize global model
        self.global_model = SimpleCNN(num_classes=2)
        self.global_model = ModuleValidator.fix(self.global_model)
        self.global_model = self.global_model.to(self.device)
        
        # Prepare client data
        self.client_loaders = []
        for c in range(num_clients):
            X, y = create_synthetic_client_data(c, samples_per_client)
            dataset = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.client_loaders.append(loader)
            print(f"Client {c}: {len(dataset)} samples, pos_ratio: {(y==1).float().mean():.2f}")
        
        # Create test set (balanced)
        X_test, y_test = create_synthetic_client_data(999, 500)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        self.history = {
            'rounds': [],
            'client_metrics': [],
            'global_accuracy': [],
            'epsilons': []
        }
    
    def train_client_with_dp(self, client_id, epochs=1):
        """Train a single client with DP."""
        # Create local model copy
        local_model = SimpleCNN(num_classes=2)
        local_model = ModuleValidator.fix(local_model)
        local_model.load_state_dict(self.global_model.state_dict())
        local_model = local_model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        
        # Attach DP
        privacy_engine = PrivacyEngine()
        local_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=local_model,
            optimizer=optimizer,
            data_loader=self.client_loaders[client_id],
            epochs=epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=1.0
        )
        
        local_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epsilon_spent = privacy_engine.get_epsilon(delta=self.delta)
        
        # Get model state dict (unwrap if needed)
        model_state = local_model._module.state_dict() if hasattr(local_model, '_module') else local_model.state_dict()
        
        return {
            'state_dict': model_state,
            'num_samples': total,
            'loss': total_loss / total,
            'accuracy': correct / total,
            'epsilon': epsilon_spent
        }
    
    def fedavg_aggregate(self, client_results):
        """FedAvg aggregation."""
        total_samples = sum(r['num_samples'] for r in client_results)
        
        # Weighted average of parameters
        avg_state_dict = {}
        for key in client_results[0]['state_dict'].keys():
            avg_state_dict[key] = sum(
                r['state_dict'][key] * (r['num_samples'] / total_samples)
                for r in client_results
            )
        
        self.global_model.load_state_dict(avg_state_dict)
    
    def evaluate_global(self):
        """Evaluate global model on test set."""
        self.global_model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.global_model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_labels.extend(labels.numpy())
                all_preds.extend(preds)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {'accuracy': accuracy, 'f1': f1}
    
    def run_simulation(self, num_rounds=5, local_epochs=1):
        """Run the FL+DP simulation."""
        print("\n" + "="*60)
        print("FEDERATED LEARNING + DIFFERENTIAL PRIVACY SIMULATION")
        print("="*60)
        print(f"  Clients: {self.num_clients}")
        print(f"  Rounds: {num_rounds}")
        print(f"  Target Epsilon: {self.epsilon}")
        print(f"  Delta: {self.delta}")
        print("="*60)
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n📍 Round {round_num}/{num_rounds}")
            
            # Train each client with DP
            client_results = []
            round_epsilons = []
            
            for client_id in range(self.num_clients):
                print(f"  Training Client {client_id}...", end=" ")
                result = self.train_client_with_dp(client_id, epochs=local_epochs)
                client_results.append(result)
                round_epsilons.append(result['epsilon'])
                print(f"Loss: {result['loss']:.4f}, Acc: {result['accuracy']:.4f}, ε: {result['epsilon']:.2f}")
            
            # Aggregate
            self.fedavg_aggregate(client_results)
            
            # Evaluate
            eval_metrics = self.evaluate_global()
            
            print(f"\n  📊 Global Model - Accuracy: {eval_metrics['accuracy']:.4f}, F1: {eval_metrics['f1']:.4f}")
            print(f"  🔒 Max Client ε: {max(round_epsilons):.2f}")
            
            # Save history
            self.history['rounds'].append(round_num)
            self.history['global_accuracy'].append(eval_metrics['accuracy'])
            self.history['epsilons'].append(max(round_epsilons))
            self.history['client_metrics'].append([
                {'client': c, 'loss': r['loss'], 'acc': r['accuracy'], 'epsilon': r['epsilon']}
                for c, r in enumerate(client_results)
            ])
        
        # Save final model
        model_path = f'models/fl_dp_sim/fl_dp_eps{self.epsilon}_rounds{num_rounds}.pth'
        torch.save(self.global_model.state_dict(), model_path)
        print(f"\n💾 Model saved: {model_path}")
        
        # Save results
        self.save_results(num_rounds)
        
        return self.history
    
    def save_results(self, num_rounds):
        """Save simulation results."""
        results = {
            'config': {
                'num_clients': self.num_clients,
                'num_rounds': num_rounds,
                'epsilon': self.epsilon,
                'delta': self.delta,
                'samples_per_client': self.samples_per_client
            },
            'history': self.history,
            'final_accuracy': self.history['global_accuracy'][-1],
            'final_max_epsilon': self.history['epsilons'][-1],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = f'{self.output_dir}/fl_dp_simulation_eps{self.epsilon}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved: {results_path}")
        
        # Generate summary report
        self.generate_report(num_rounds)
    
    def generate_report(self, num_rounds):
        """Generate markdown report."""
        report = [
            "# FL + DP Simulation Results\n",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Configuration\n",
            f"- **Number of Clients:** {self.num_clients}",
            f"- **Federated Rounds:** {num_rounds}",
            f"- **Target Epsilon (ε):** {self.epsilon}",
            f"- **Delta (δ):** {self.delta}",
            f"- **Samples per Client:** {self.samples_per_client}\n",
            "## Results Summary\n",
            f"- **Final Global Accuracy:** {self.history['global_accuracy'][-1]:.4f}",
            f"- **Final Max Client ε:** {self.history['epsilons'][-1]:.2f}\n",
            "## Round-by-Round Progress\n",
            "| Round | Global Accuracy | Max ε |",
            "|-------|-----------------|-------|"
        ]
        
        for i, (acc, eps) in enumerate(zip(self.history['global_accuracy'], self.history['epsilons'])):
            report.append(f"| {i+1} | {acc:.4f} | {eps:.2f} |")
        
        report.append("\n## Privacy Analysis\n")
        report.append(f"Each client achieved (ε={self.epsilon}, δ={self.delta})-differential privacy per round.")
        report.append("With composition across rounds, the total privacy budget increases.")
        
        report_path = f'{self.output_dir}/fl_dp_report_eps{self.epsilon}.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Report saved: {report_path}")


def run_multiple_epsilons():
    """Run simulations with different epsilon values."""
    epsilons = [1.0, 5.0, 10.0]
    results_summary = []
    
    print("\n" + "="*70)
    print("RUNNING FL+DP EXPERIMENTS WITH MULTIPLE EPSILON VALUES")
    print("="*70)
    
    for epsilon in epsilons:
        print(f"\n\n{'#'*70}")
        print(f"# EXPERIMENT: ε = {epsilon}")
        print(f"{'#'*70}")
        
        simulator = FLDPSimulator(
            num_clients=3,
            epsilon=epsilon,
            delta=1e-5,
            samples_per_client=1000,
            batch_size=32
        )
        
        history = simulator.run_simulation(num_rounds=5, local_epochs=1)
        
        results_summary.append({
            'epsilon': epsilon,
            'final_accuracy': history['global_accuracy'][-1],
            'final_epsilon': history['epsilons'][-1]
        })
    
    # Print comparison
    print("\n\n" + "="*70)
    print("PRIVACY-UTILITY TRADE-OFF SUMMARY")
    print("="*70)
    print(f"\n{'Epsilon (ε)':<15} {'Final Accuracy':<18} {'Actual ε Spent':<15}")
    print("-"*48)
    for r in results_summary:
        print(f"{r['epsilon']:<15} {r['final_accuracy']:<18.4f} {r['final_epsilon']:<15.2f}")
    
    # Save comparison
    with open('results/fl_dp_sim/epsilon_comparison.json', 'w') as f:
        json.dump({
            'results': results_summary,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📁 Comparison saved: results/fl_dp_sim/epsilon_comparison.json")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="FL+DP Simulation")
    parser.add_argument('--epsilon', type=float, default=5.0, help='Target epsilon')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--samples_per_client', type=int, default=1000, help='Samples per client')
    parser.add_argument('--compare', action='store_true', help='Run comparison with multiple epsilons')
    
    args = parser.parse_args()
    
    if args.compare:
        run_multiple_epsilons()
    else:
        simulator = FLDPSimulator(
            num_clients=args.num_clients,
            epsilon=args.epsilon,
            samples_per_client=args.samples_per_client
        )
        simulator.run_simulation(num_rounds=args.num_rounds)

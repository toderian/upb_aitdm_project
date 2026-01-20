"""
Flower server with Differential Privacy tracking.
Aggregates models and tracks cumulative privacy budget.
"""
import flwr as fl
import torch
import numpy as np
import argparse
import json
import os
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

from model_side.models.cnn_model import SimpleCNN


class DPFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy with DP tracking and model saving.
    """
    def __init__(self, output_dir='results/fl_dp', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models/fl_dp', exist_ok=True)
        
        self.round_metrics = []
        self.client_epsilons = {}  # Track epsilon per client
        
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate with DP metrics tracking."""
        # Standard FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert parameters to numpy arrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Rebuild model and save
            model = SimpleCNN(num_classes=2)
            # Fix model for DP compatibility
            from opacus.validators import ModuleValidator
            model = ModuleValidator.fix(model)
            
            state_dict_keys = list(model.state_dict().keys())
            new_state_dict = {
                key: torch.tensor(array) for key, array in zip(state_dict_keys, ndarrays)
            }
            model.load_state_dict(new_state_dict, strict=True)
            
            # Save model
            save_path = f"models/fl_dp/fl_dp_round_{server_round}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model saved: {save_path}")
            
            # Extract and track epsilon from client metrics
            for client_id, fit_res in results:
                client_metrics = fit_res.metrics
                if 'epsilon' in client_metrics:
                    eps = client_metrics['epsilon']
                    if client_id not in self.client_epsilons:
                        self.client_epsilons[client_id] = []
                    self.client_epsilons[client_id].append(eps)
                    print(f"  Client {client_id}: ε = {eps:.2f}")
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation with metrics logging."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated:
            loss, metrics = aggregated
            
            # Get max epsilon across clients for this round
            max_eps = 0
            for client_id, eps_list in self.client_epsilons.items():
                if eps_list:
                    max_eps = max(max_eps, eps_list[-1])
            
            round_data = {
                'round': server_round,
                'loss': float(loss),
                'accuracy': metrics.get('accuracy', 0),
                'max_epsilon': max_eps
            }
            self.round_metrics.append(round_data)
            
            print(f"\n📊 Round {server_round} Summary:")
            print(f"   Loss: {loss:.4f}")
            print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"   Max ε: {max_eps:.2f}")
            
            # Save metrics after each round
            self._save_metrics()
        
        return aggregated
    
    def _save_metrics(self):
        """Save accumulated metrics to file."""
        metrics_path = f"{self.output_dir}/fl_dp_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'rounds': self.round_metrics,
                'client_epsilons': {str(k): v for k, v in self.client_epsilons.items()},
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of metrics from clients."""
    if not metrics:
        return {}
    
    accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
    epsilons = [m.get("epsilon", 0) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    total_examples = sum(examples)
    
    result = {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0,
        "loss": sum(losses) / total_examples if total_examples > 0 else 0,
    }
    
    # Track max epsilon
    if epsilons:
        result["max_epsilon"] = max(epsilons)
    
    return result


def get_on_fit_config(local_epochs: int):
    """Return function that creates fit config for each round."""
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "current_round": server_round
        }
    return fit_config


def get_initial_parameters():
    """Get initial model parameters (DP-compatible)."""
    from opacus.validators import ModuleValidator
    model = SimpleCNN(num_classes=2)
    model = ModuleValidator.fix(model)
    return [val.cpu().numpy() for val in model.state_dict().values()]


def main(args):
    print("="*60)
    print("FEDERATED LEARNING SERVER WITH DIFFERENTIAL PRIVACY")
    print("="*60)
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Output: {args.output_dir}")
    print("="*60)
    
    # Define strategy
    strategy = DPFedAvgStrategy(
        output_dir=args.output_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config(args.local_epochs),
        initial_parameters=ndarrays_to_parameters(get_initial_parameters()),
    )
    
    # Server config
    config = fl.server.ServerConfig(num_rounds=args.num_rounds)
    
    # Start server
    print(f"\n🚀 Starting server on {args.server_address}")
    print(f"⏳ Waiting for {args.num_clients} clients...\n")
    
    fl.server.start_server(
        server_address=args.server_address,
        config=config,
        strategy=strategy
    )
    
    print("\n✅ Federated learning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower DP Server")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/fl_dp")
    args = parser.parse_args()
    main(args)

import flwr as fl
from client import FlowerClient # Import the class directly
from server_utils import ServerManager
import argparse
import torch
import os

def run_simulation(strategy_type="fedavg", num_rounds=5, mu=0.1, lr=0.001, freeze_backbone=False):
    # Create a descriptive strategy name for folders (e.g., "fedavg_frozen")
    freeze_str = "frozen" if freeze_backbone else "unfrozen"
    full_strategy_name = f"{strategy_type}_{freeze_str}"
    
    print(f"--- Starting Simulation: {full_strategy_name.upper()} ---")
    print(f"Learning Rate: {lr}, Frozen Backbone: {freeze_backbone}")
    
    num_gpus_available = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus_available}")

    # Initialize Server Manager with the detailed name
    manager = ServerManager(strategy_name=full_strategy_name)
    
    # --- Define Client Factory Locally ---
    # Fix: Use (cid: str) signature for compatibility with older Flower versions
    def client_fn(cid: str):
        # Pass the freeze flag to the client
        return FlowerClient(int(cid), freeze_backbone=freeze_backbone).to_client()

    def fit_config(server_round: int):
        return {
            "local_epochs": 1,
            "proximal_mu": mu if strategy_type == "fedprox" else 0.0,
            "lr": lr,
        }

    if strategy_type == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            evaluate_fn=manager.get_evaluate_fn(),
            on_fit_config_fn=fit_config,
        )
    elif strategy_type == "fedprox":
        strategy = fl.server.strategy.FedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            proximal_mu=mu,
            evaluate_fn=manager.get_evaluate_fn(),
            on_fit_config_fn=fit_config,
        )
    else:
        raise ValueError("Unknown strategy")

    # Resource Allocation
    # Reduce slightly to prevent edge-case OOM if system overhead is high
    gpu_per_client = 0.5
    cpu_per_client = 4 

    client_resources = {"num_cpus": cpu_per_client, "num_gpus": gpu_per_client} 
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.1, help="Proximal mu for FedProx")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--freeze", action="store_true", help="Freeze ResNet backbone layers")
    
    args = parser.parse_args()

    run_simulation(args.strategy, args.rounds, args.mu, args.lr, args.freeze)
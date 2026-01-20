import flwr as fl
import argparse
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, ndarrays_to_parameters

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    """
    # Collect accuracies and losses
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples) if sum(losses) > 0 else 0
    }

def get_on_fit_config(local_epochs: int):
    """
    Return function that creates fit config for each round.
    """
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "current_round": server_round
        }
    return fit_config

import flwr as fl
import torch
import numpy as np
from flwr.common import parameters_to_ndarrays
from model_side.models.cnn_model import COVIDxCNN

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        # Face agregarea normală
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convertim parametrii Flower -> numpy arrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Reconstruim modelul PyTorch EXACT ca pe client
            model = COVIDxCNN(num_classes=2, pretrained=True)

            # Structurăm parametrii în state_dict
            state_dict_keys = list(model.state_dict().keys())
            new_state_dict = {
                key: torch.tensor(array) for key, array in zip(state_dict_keys, ndarrays)
            }

            # Încărcăm parametrii în model
            model.load_state_dict(new_state_dict, strict=True)

            # Salvăm modelul PyTorch
            save_path = f"server_model_round_{server_round}.pth"
            torch.save(model.state_dict(), save_path)

            print(f"💾 Model PyTorch salvat la: {save_path}")

        return aggregated_parameters, metrics

def get_initial_parameters():
    """Get initial model parameters."""
    model = COVIDxCNN(num_classes=2, pretrained=True)
    return [val.cpu().numpy() for val in model.state_dict().values()]

def main(args):
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config(args.local_epochs),
        initial_parameters=ndarrays_to_parameters(get_initial_parameters()),  # Add this

    )

    # Server config
    config = fl.server.ServerConfig(num_rounds=args.num_rounds)

    # Start server
    print(f"Starting Flower server on {args.server_address}")
    print(f"Waiting for {args.num_clients} clients...")
    print(f"Running {args.num_rounds} rounds with {args.local_epochs} local epochs each")

    fl.server.start_server(
        server_address=args.server_address,
        config=config,
        strategy=strategy
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
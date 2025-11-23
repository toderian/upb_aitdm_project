import flwr as fl
import argparse
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

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

def main(args):
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all clients for evaluation
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config(args.local_epochs)
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
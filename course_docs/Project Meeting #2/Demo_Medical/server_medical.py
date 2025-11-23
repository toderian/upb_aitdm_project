import flwr as fl
import argparse

# -------------------------------
# Metric aggregation functions
# -------------------------------
def fit_metrics_aggregation(metrics_list):
    """
    Aggregate training metrics from all clients.
    Computes weighted average of loss and accuracy.
    """
    total_examples = 0
    weighted_loss = 0.0
    weighted_acc = 0.0
    for m in metrics_list:
        if m is None:
            continue
        num_examples, metrics = m
        if metrics:
            if "loss" in metrics:
                weighted_loss += metrics["loss"] * num_examples
            if "accuracy" in metrics:
                weighted_acc += metrics["accuracy"] * num_examples
            total_examples += num_examples
    return {
        "loss": float(weighted_loss / total_examples) if total_examples > 0 else 0.0,
        "accuracy": float(weighted_acc / total_examples) if total_examples > 0 else 0.0
    }

def evaluate_metrics_aggregation(metrics_list):
    """
    Aggregate evaluation metrics from all clients.
    Computes weighted average of accuracy only.
    """
    total_examples = 0
    weighted_acc = 0.0
    for m in metrics_list:
        if m is None:
            continue
        num_examples, metrics = m
        if metrics and "accuracy" in metrics:
            weighted_acc += metrics["accuracy"] * num_examples
            total_examples += num_examples
    return {"accuracy": float(weighted_acc / total_examples) if total_examples > 0 else 0.0}


def get_arguments():
    parser = argparse.ArgumentParser(description="Flower Template Server")
    parser.add_argument(
        "--federation_rounds",
        type=int,
        default=3,
        help="Number of training rounds",
    )
    parser.add_argument(
        "--train_rounds",
        type=int,
        default=5,
        help="Number of local training epochs per round",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of clients",
    )
    return parser.parse_args()



if __name__ == "__main__":

    args = get_arguments()

    # Flower FedAvg strategy with custom metric aggregation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        on_fit_config_fn=lambda rnd: {"local_epochs": args.train_rounds},  # Number of local epochs per round
    )

    fl.server.start_server( 
        server_address="0.0.0.0:8080" #127.0.0.1,
        config=fl.server.ServerConfig(num_rounds=args.federation_rounds),
        strategy=strategy
        # strategy=fl.server.strategy.FedAvg()
        )

        

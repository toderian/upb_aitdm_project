import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from model import get_model, train, test
import data_loader 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT_BATCH_SIZE = 32

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, freeze_backbone=False):
        self.client_id = client_id
        # Initialize model with the specific freeze setting
        self.net = get_model(DEVICE, freeze_backbone=freeze_backbone)
        
        try:
            self.trainloader = data_loader.get_federated_client(client_id, batch_size=CLIENT_BATCH_SIZE)
            self.valloader = data_loader.get_client_validation(client_id, batch_size=CLIENT_BATCH_SIZE)
        except Exception as e:
            print(f"Client {client_id} Data Load Error: {e}")
            self.trainloader = None
            self.valloader = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Read config from server
        proximal_mu = float(config.get("proximal_mu", 0.0))
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))
        
        if self.trainloader:
            train(self.net, self.trainloader, epochs=epochs, device=DEVICE, 
                  proximal_mu=proximal_mu, lr=lr)
            
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.valloader:
            loss, accuracy = test(self.net, self.valloader, DEVICE)
            return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
        else:
            return 0.0, 0, {"accuracy": 0.0}

# Fix: Use standard signature (cid: str) instead of context
def client_fn(cid: str):
    return FlowerClient(int(cid)).to_client()
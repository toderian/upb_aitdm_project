import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from model import get_model, test, get_predictions, save_model_safe
import data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SERVER_BATCH_SIZE = 256 

class ServerManager:
    def __init__(self, strategy_name, save_dir="results"):
        self.strategy_name = strategy_name
        self.save_dir = os.path.join(save_dir, strategy_name)
        self.ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        self.plots_dir = os.path.join(self.save_dir, "plots")
        self.history_file = os.path.join(self.save_dir, "history.json")
        self.history = {"rounds": [], "accuracy": [], "loss": []}
        
        # Track the best model metrics
        self.best_loss = float("inf")
        self.best_round = -1
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Load Global Test Loader
        self.test_loader = data_loader.get_global_test_loader(batch_size=SERVER_BATCH_SIZE)

    def save_model(self, suffix, parameters):
        """
        Save model checkpoint.
        suffix can be a round number or 'best'.
        """
        # Load parameters into a model structure for saving
        net = get_model(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        
        filename = f"model_{suffix}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        
        save_model_safe(net, path)
        print(f"Saved checkpoint: {path}")

    def save_history(self):
        """Save metrics to JSON."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=4)

    def save_confusion_matrix(self, net, suffix):
        """
        Generates and saves a confusion matrix heatmap.
        suffix can be a round number or 'best'.
        """
        y_true, y_pred = get_predictions(net, self.test_loader, DEVICE)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {suffix}')
        
        filename = f"conf_matrix_{suffix}.png"
        path = os.path.join(self.plots_dir, filename)
        plt.savefig(path)
        plt.close()

    def get_evaluate_fn(self):
        def evaluate(server_round, parameters, config):
            # 1. Instantiate model with current parameters
            net = get_model(DEVICE)
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # 2. Evaluate on Global Test Set
            loss, accuracy = test(net, self.test_loader, DEVICE)
            
            # 3. Log History
            self.history["rounds"].append(server_round)
            self.history["accuracy"].append(accuracy)
            self.history["loss"].append(loss)
            self.save_history()

            # 4. Save Regular Checkpoint (History)
            self.save_model(f"round_{server_round}", parameters)
            self.save_confusion_matrix(net, f"round_{server_round}")
            
            # 5. Check and Save BEST model (Lowest Loss Strategy)
            if loss < self.best_loss:
                print(f"\n>>> NEW BEST MODEL FOUND at Round {server_round}! (Loss: {loss:.4f} < {self.best_loss:.4f})")
                self.best_loss = loss
                self.best_round = server_round
                # Save as "model_best.pth" and "conf_matrix_best.png"
                self.save_model("best", parameters)
                self.save_confusion_matrix(net, "best")
            
            print(f"Round {server_round} | Global Acc: {accuracy:.4f} | Loss: {loss:.4f} | (Best: {self.best_loss:.4f})")

            return loss, {"accuracy": accuracy}
        
        return evaluate
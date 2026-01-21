import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import data_loader
from model import get_model, train, test, get_predictions, save_model_safe
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

DEFAULT_BATCH_SIZE = 256  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

def get_all_training_data(batch_size):
    datasets = []
    for cid in [0, 1, 2]:
        loader = data_loader.get_federated_client(cid, batch_size=batch_size)
        datasets.append(loader.dataset)
    
    full_dataset = ConcatDataset(datasets)
    print(f"Total Centralized Training Samples: {len(full_dataset)}")
    
    return DataLoader(full_dataset, batch_size=batch_size, shuffle=True, 
                      num_workers=8, pin_memory=True, persistent_workers=True)

def check_test_balance(loader):
    print("\n--- Checking Test Set Balance ---")
    try:
        df = loader.dataset.df
        counts = df['label'].value_counts()
        total = len(df)
        print(f"Total Test Images: {total}")
        for label, count in counts.items():
            percentage = (count / total) * 100
            print(f"  - {label}: {count} ({percentage:.2f}%)")
        print("---------------------------------\n")
    except Exception as e:
        print(f"Could not check balance: {e}")

def debug_data_loading(loader):
    print("\n--- DEBUG: Checking Data Loader ---")
    try:
        images, labels = next(iter(loader))
        if images.std() < 0.001:
            print("\n!!! CRITICAL ERROR: Loaded images appear to be empty/flat! Check archive.zip !!!\n")
        else:
            print(f"Data seems valid. Batch Shape: {images.shape}")
            print(f"Image Mean: {images.mean().item():.4f}, Std: {images.std().item():.4f}")
            print("-----------------------------------\n")
    except Exception as e:
        print(f"CRITICAL: Data loading failed completely: {e}")

def save_confusion_matrix(net, test_loader, epoch, plots_dir):
    y_true, y_pred = get_predictions(net, test_loader, DEVICE)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    
    filename = f"conf_matrix_epoch_{epoch}.png"
    path = os.path.join(plots_dir, filename)
    plt.savefig(path)
    plt.close()

def run_centralized(model_name="resnet", optimizer_name="adamw", lr=None, batch_size=DEFAULT_BATCH_SIZE, freeze=False, dry_run=False):
    lr_str = f"lr{lr}" if lr else "lrDefault"
    # Dynamic folder naming for Frozen vs NoFreeze
    freeze_str = "frozen" if freeze else "nofreeze"
    save_dir_name = f"results/centralized_{model_name}_{optimizer_name}_{freeze_str}_sched_bs{batch_size}_{lr_str}"
    
    plots_dir = os.path.join(save_dir_name, "plots")
    os.makedirs(save_dir_name, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"--- Starting Centralized Training on {DEVICE} ---")
    print(f"Model: {model_name}, Freeze: {freeze}, Optimizer: {optimizer_name}, LR: {lr}, Batch Size: {batch_size}")
    print(f"Saving results to: {save_dir_name}")

    if dry_run:
        print("!!! DRY RUN MODE: Training for limited batches only !!!")

    train_loader = get_all_training_data(batch_size)
    test_loader = data_loader.get_global_test_loader(batch_size=batch_size)
    
    check_test_balance(test_loader)
    debug_data_loading(train_loader)
    
    # Pass the freeze argument here
    model = get_model(DEVICE, model_name=model_name, freeze_backbone=freeze)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU or CPU.")

    # --- SETUP OPTIMIZER & SCHEDULER ---
    if lr is None:
        lr = 0.01 if optimizer_name == "sgd" else 1e-4

    weight_decay = 1e-4
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    else:
        raise ValueError("Unknown optimizer")

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    history = {"epochs": [], "accuracy": [], "loss": [], "train_accuracy": [], "train_loss": []}
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} ...")
        
        max_batches = 10 if dry_run else None
        
        train_loss, train_acc = train(model, train_loader, epochs=1, device=DEVICE, 
                                      max_batches=max_batches, optimizer=optimizer)
        
        scheduler.step()
        
        loss, accuracy = test(model, test_loader, DEVICE)
        
        history["epochs"].append(epoch)
        history["accuracy"].append(accuracy)
        history["loss"].append(loss)
        history["train_accuracy"].append(train_acc)
        history["train_loss"].append(train_loss)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {loss:.4f}, Test Acc: {accuracy:.4f}")

        save_model_safe(model, os.path.join(save_dir_name, f"model_epoch_{epoch}.pth"))
        save_confusion_matrix(model, test_loader, epoch, plots_dir)

        if dry_run:
            print("Dry run finished successfully.")
            return

    with open(os.path.join(save_dir_name, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    print("Centralized training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a short test to check speed")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "convnext", "convnext_base"], help="Model architecture")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer type")
    parser.add_argument("--lr", type=float, default=None, help="Learning Rate override (optional)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    # Added --freeze argument
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone layers")
    
    args = parser.parse_args()
    
    run_centralized(
        model_name=args.model, 
        optimizer_name=args.optimizer, 
        lr=args.lr, 
        batch_size=args.batch_size,
        freeze=args.freeze,
        dry_run=args.dry_run
    )
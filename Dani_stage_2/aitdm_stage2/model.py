import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

def get_model(device, model_name="resnet", freeze_backbone=False):
    """
    Returns a model modified for binary classification.
    Args:
        model_name (str): "resnet" (ResNet50), "convnext" (Tiny), or "convnext_base"
        freeze_backbone (bool): If True, locks the convolutional layers.
    """
    if model_name == "resnet":
        print(f"[Model] Loading ResNet50 (Freeze={freeze_backbone})...")
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except (AttributeError, TypeError):
            model = models.resnet50(pretrained=True)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    elif model_name == "convnext":
        print(f"[Model] Loading ConvNeXt Tiny (Freeze={freeze_backbone})...")
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            model = models.convnext_tiny(weights=weights)
        except (AttributeError, TypeError):
            model = models.convnext_tiny(pretrained=True)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    elif model_name == "convnext_base":
        print(f"[Model] Loading ConvNeXt Base (Freeze={freeze_backbone})...")
        try:
            # ConvNeXt Base is larger and more powerful than Tiny
            weights = models.ConvNeXt_Base_Weights.DEFAULT
            model = models.convnext_base(weights=weights)
        except (AttributeError, TypeError):
            model = models.convnext_base(pretrained=True)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Adjust classifier head
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    return model

def get_predictions(net, loader, device):
    net.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Generating Predictions", leave=False):
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    return all_targets, all_preds

def test(net, testloader, device):
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

def train(net, trainloader, epochs, device, proximal_mu=0.0, max_batches=None, optimizer_name="adamw", lr=None, optimizer=None):
    """
    Train the network.
    """
    # Using Label Smoothing to help with generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 1. OPTIMIZER LOGIC
    if optimizer is None:
        if lr is None:
            if optimizer_name == "sgd":
                lr = 0.01 
            elif optimizer_name == "adamw":
                lr = 1e-4

        weight_decay = 1e-4
        # Only update parameters that require gradients (handles frozen layers)
        params_to_update = [p for p in net.parameters() if p.requires_grad]

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params_to_update, lr=lr, weight_decay=1e-2)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    global_params = [p.clone().detach() for p in net.parameters()] if proximal_mu > 0.0 else None
    
    running_loss = 0.0
    correct = 0
    total = 0

    net.train()
    for epoch in range(epochs):
        pbar = tqdm(trainloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                pbar.close()
                print(f"\nDry run limit reached ({max_batches} batches). Stopping epoch.")
                break 

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            if proximal_mu > 0.0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc

def save_model_safe(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
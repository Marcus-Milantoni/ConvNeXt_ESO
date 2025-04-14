import numpy as np
import torch


def train_one_epoch(model, optimizer, loss, training_loader, device) -> None:
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        loss (torch.nn.Module): The loss function.
        training_loader (torch.utils.data.DataLoader): The training data loader.
        device (torch.device): The device to use for training (CPU or GPU).
        
    Returns:
        None
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    all_outcomes = []
    all_outputs = []
    
    for batch in training_loader:
        inputs, labels = batch["Data"].to(device), batch["outcome"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss_value = loss(outputs, labels)
        loss_value.backward()
        
        optimizer.step()
        
        running_loss += loss_value.item() * inputs.size(0)
        total_samples += inputs.size(0)
    
        all_outcomes.append(labels.detatch().cpu().numpy())
        all_outputs.append(outputs.detatch().cpu().numpy())

    epoch_loss = running_loss / total_samples
    
    all_outcomes = np.concatenate(all_outcomes, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    return epoch_loss, all_outputs, all_outcomes


def val_one_epoch(model, loss, validation_loader, device) -> None:
    """
    Validate the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to validate.
        loss (torch.nn.Module): The loss function.
        validation_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to use for validation (CPU or GPU).
        
    Returns:
        None
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_outcomes = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in validation_loader:
            inputs, labels = batch["Data"].to(device), batch["outcome"].to(device)
            
            outputs = model(inputs)
            
            loss_value = loss(outputs, labels)
            
            running_loss += loss_value.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_outcomes.append(labels.detach().cpu().numpy())
            all_outputs.append(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / total_samples
    
    all_outcomes = np.concatenate(all_outcomes, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    return epoch_loss, all_outputs, all_outcomes

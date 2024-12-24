import os
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs, accumulation_steps=1, checkpoint_dir=None):
    """
    Train the model and validate after each epoch with mixed precision and gradient accumulation.
    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to use ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        accumulation_steps (int): Number of steps to accumulate gradients.
        checkpoint_dir (str): Directory to save model checkpoints.
    Returns:
        torch.nn.Module: The best-performing model on validation data.
    """
    scaler = GradScaler()  # Initialize gradient scaler for mixed precision
    best_model = None
    best_val_loss = float('inf')

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it doesn't exist

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs, task_labels, clothing_types) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(device, non_blocking=True)

            with autocast():
                losses = []
                for i in range(len(clothing_types)):
                    clothing_type = clothing_types[i]
                    sample_task_labels = {task: task_labels[task][i] for task in task_labels}

                    for task, label in sample_task_labels.items():
                        task_name = f"{clothing_type}_{task}"
                        label = label.to(device, non_blocking=True)
                        if label.dim() == 0:
                            label = label.unsqueeze(0)

                        output = model(inputs[i].unsqueeze(0), task_name)
                        losses.append(criterion(output, label))

                total_loss = sum(losses) / len(losses)

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += total_loss.item()

        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, task_labels, clothing_types) in enumerate(val_dataloader):
                inputs = inputs.to(device, non_blocking=True)
                with autocast():  # Use mixed precision for validation
                    losses = []
                    for i in range(len(clothing_types)):
                        clothing_type = clothing_types[i]
                        sample_task_labels = {task: task_labels[task][i] for task in task_labels}

                        for task, label in sample_task_labels.items():
                            task_name = f"{clothing_type}_{task}"
                            label = label.to(device, non_blocking=True)
                            label = label.unsqueeze(0)

                            output = model(inputs[i].unsqueeze(0), task_name)
                            losses.append(criterion(output, label))

                    val_loss += sum(losses).item() / len(losses)

        val_loss /= len(val_dataloader)

        # Save checkpoint
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return best_model

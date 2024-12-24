import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset.
    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for test data.
        device (str): Device to use ('cuda' or 'cpu').
    """
    model.eval()
    all_predictions = {task: [] for task in dataloader.dataset.label_mappings['아우터']}
    all_targets = {task: [] for task in dataloader.dataset.label_mappings['아우터']}

    with torch.no_grad():
        for inputs, task_labels, clothing_types in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            for i, clothing_type in enumerate(clothing_types):
                for task, labels in task_labels.items():
                    task_name = f"{clothing_type}_{task}"  # Generate task name dynamically
                    labels = labels.to(device, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(inputs[i].unsqueeze(0), task_name)
                    _, preds = torch.max(outputs, 1)

                    all_predictions[task].extend(preds.cpu().numpy())
                    all_targets[task].extend(labels.cpu().numpy())

    # Compute accuracy for each task
    for task in all_predictions:
        accuracy = accuracy_score(all_targets[task], all_predictions[task])
        print(f"Task {task} Accuracy: {accuracy:.4f}")

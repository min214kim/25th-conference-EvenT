import torch.nn as nn
import unicodedata
from torchvision import models
from transformers import AutoModel, AutoProcessor

class EfficientNetMultiTask(nn.Module):
    def __init__(self, num_classes_per_task):
        """
        EfficientNetB7 with separate heads for clothing types and tasks.
        Args:
            num_classes_per_task (dict): Dict of dicts mapping clothing types and tasks to number of classes.
        """
        super(EfficientNetMultiTask, self).__init__()
        self.base_model = models.efficientnet_b3(pretrained=True)

        # Remove default classifier and adapt for mixed precision
        self.base_model.classifier = nn.Identity()

        # Add task-specific heads for each clothing type
        self.task_heads = nn.ModuleDict()
        for clothing_type, tasks in num_classes_per_task.items():
            for task, num_classes in tasks.items():
                head_name = f"{clothing_type}_{task}"
                self.task_heads[head_name] = nn.Sequential(
                    nn.Linear(1536, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )

    def forward(self, x, task):
        """
        Forward pass with task-specific head.
        Args:
            x (Tensor): Input tensor.
            task (str): Task-specific head name (e.g., '하의_task1').
        Returns:
            Tensor: Output tensor for the given task.
        """

        task = unicodedata.normalize('NFC', task)
        features = self.base_model(x)  # EfficientNetB7 feature extractor
        return self.task_heads[task](features)
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModel, AutoProcessor

class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        embedding_dim = 128
        self.model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)

        self.embeddingLayer1 = nn.Linear(768, 256)
        self.embeddingLayer2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        processed = self.processor(images=x, padding='max_length', return_tensors="pt")
        pixel_values = processed['pixel_values'].to(next(self.model.parameters()).device)
        x = self.model.get_image_features(pixel_values, normalize=True)
        x = x.clone().detach().requires_grad_(True)

        embedding = self.embeddingLayer1(x)
        embedding = self.embeddingLayer2(embedding)
        return embedding
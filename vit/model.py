from transformers import ViTModel, AutoImageProcessor, ViTConfig
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T

class ViTPosterClassifier(nn.Module):

    def __init__(self, num_classes: int=4709) -> None:
        super().__init__()
        self.vitConfig = ViTConfig()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fullyConnected = nn.Sequential(nn.ModuleList([
            nn.LazyLinear(num_classes),
            nn.GELU(),
            nn.Dropout(0.2)
        ]))
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Data.
            y (torch.Tensor, optional): Targets. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of logits and loss.
        """
        batch_size = x.shape[0]
        x = self.vit.forward(x,
                            return_dict=False,
                            output_attentions=False,
                            output_hidden_states=False)[0]
        logits = self.fullyConnected(x.view(batch_size, -1))

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            logits, _ = self.forward(x)
            prediction = logits.softmax(dim=-1).argmax(dim=-1)
        self.train()
        return prediction
    
    @classmethod
    def from_pretrained(cls, path: str, num_classes: int=4709) -> nn.Module:
        """
        Loads a pretrained model from a state dict.

        Args:
            path (str): Path to state dict.
            num_classes (int, optional): Number of classes. Defaults to 4709.

        Returns:
            nn.Module: Model with loaded weights.
        """
        model = cls(num_classes)
        model.load_state_dict(torch.load(path))
        return model
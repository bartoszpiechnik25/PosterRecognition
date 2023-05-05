from transformers import ViTModel, ViTConfig
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

class ViTPosterClassifier(nn.Module):

    def __init__(self, last_hidden_size: int = 2048, num_classes: int=4709) -> None:
        super().__init__()
        self.vitConfig = ViTConfig()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fullyConnected = nn.Sequential(
            nn.Linear(self.vitConfig.hidden_size, last_hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(last_hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Data.
            y (torch.Tensor, optional): Targets. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of logits and loss.
        """
        batch = x.shape[0]
        x: torch.Tensor = self.vit.forward(x,
                            return_dict=False,
                            output_attentions=False,
                            output_hidden_states=False)[0]
        
        x = self.avgPool(x.permute(0, 2, 1)).view(batch, -1)
        logits = self.fullyConnected(x)

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
    def from_pretrained(cls, path: str) -> nn.Module:
        """
        Loads a pretrained model from a state dict.

        Args:
            path (str): Path to state dict.
            num_classes (int, optional): Number of classes. Defaults to 4709.

        Returns:
            nn.Module: Model with loaded weights.
        """
        model = torch.load(path)
        return model
    
    def saveWeights(self, path: str=None) -> None:
        """
        Saves the model weights.

        Args:
            path (str): Path to save weights.
        """
        if path is None:
            path = f"/home/barti/PosterRecognition/vit/checkpoints/{self._get_name()}_state_dict.pth"
        print(f"Saving model to ---> {path}")
        torch.save(self, path)


if __name__ == '__main__':
    model = ViTPosterClassifier()
    print(f"Model has --> {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    print(model._get_name())
    dummy_batch = torch.randn(8, 3, 224, 224, device='cuda')
    dummy_target = torch.randint(0, 4709, (8,), device='cuda')
    model.to('cuda')
    logits, loss = model.forward(dummy_batch, dummy_target)
    loss.backward()
    model.saveWeights()
    print(logits.shape)
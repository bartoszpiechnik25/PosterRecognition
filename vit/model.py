from transformers import ViTModel, ViTConfig
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

class ViTPosterClassifier(nn.Module):

    def __init__(self, last_hidden_size: int = 2048,
                       num_classes: int = 4709,
                       checkpoint: str = 'google/vit-base-patch16-224-in21k') -> None:
        super().__init__()
        self.vitConfig = ViTConfig()
        self.vit = ViTModel.from_pretrained(checkpoint)
        for param in self.vit.parameters():
            param.requires_grad = False
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
        
        logits = self.fullyConnected(x[:, 0, :])

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
    
    @staticmethod
    def validation_loop(model: torch.nn.Module, data: torch.utils.data.DataLoader) -> float:
        """
        Calculates the validation loss over the whole dataset

        Args:
            model (torch.Module): Model to validate.
            data (torch.utils.data.DataLoader): Data to validate on.

        Returns:
            float: Validation loss.
        """
        model.eval()
        dataset_loss: float = .0
        dataset_accuracy: float = .0

        for j, (X, Y) in enumerate(data, 1):
            X = X.to('cuda')
            Y = Y.to('cuda')
            with torch.inference_mode():
                logits, loss = model(X, Y)
            dataset_accuracy += ViTPosterClassifier.calculateAccuracy(logits, Y, return_percent=True)
            dataset_loss += loss.item()
            del X, Y
        model.train()
        return dataset_loss / j, dataset_accuracy / j

    @staticmethod
    def calculateAccuracy(logits: torch.Tensor,
                          targets: torch.Tensor,
                          return_percent: bool = True) -> float:
        """
        Calculates the accuracy of the model.

        Args:
            logits (torch.Tensor): Logits of the model.
            targets (torch.Tensor): Targets.

        Returns:
            float: Accuracy.
        """
        predictions = logits.softmax(dim=-1).argmax(dim=-1)
        if return_percent:
            return (predictions == targets).sum().item() / targets.shape[0]
        else:
            return (predictions == targets).sum().item()


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
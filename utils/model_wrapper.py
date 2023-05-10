from torch.utils.data import DataLoader
from typing import Callable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class ViTPosterClassifier(nn.Module):

    def __init__(self, last_hidden_size: int = 4096,
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
            nn.Dropout(0.5),
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
        x: torch.Tensor = self.vit.forward(x,
                                           return_dict=False,
                                           output_attentions=False,
                                           output_hidden_states=False)[0]
        
        logits = self.fullyConnected(x[:, 0, :])

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits, y)
        return logits, loss

class DinoV2Classifier(nn.Module):
    
    def __init__(self,
                 num_embeddings: int = 768,
                 num_classes: int = 1000,
                 feature_extraction: bool = True) -> None:
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        if feature_extraction:
            for parmeter in self.dinov2.parameters():
                parmeter.requires_grad = False
        self.linear = nn.Linear(num_embeddings * 2, num_classes)
        self.linear.bias.data.zero_()
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.dinov2.forward_features(x)
        linear_inpt = torch.cat([features['x_norm_clstoken'],
                                 features['x_norm_patchtokens'].mean(dim=1)], dim=-1)
        logits = self.linear(linear_inpt)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

#wondering if this is a good idea
class DinoV2ClassifierMLP(DinoV2Classifier):

    def __init__(self,
                 num_embeddings: int = 768,
                 num_classes: int = 1000,
                 feature_extraction: bool = True,
                 hidden_size: int = 2048) -> None:
        super().__init__(num_embeddings, hidden_size, feature_extraction)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        linear, _ = super().forward(x)
        logits = self.mlp(linear)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

@torch.inference_mode()
def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        logits, _ = model.forward(x)
        prediction = logits.softmax(dim=-1).argmax(dim=-1)
    model.train()
    return prediction

def validation_loop(model: torch.nn.Module, data: DataLoader) -> float:
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
        dataset_accuracy += calculateAccuracy(logits, Y, return_percent=True)
        dataset_loss += loss.item()
        del X, Y
    model.train()
    return dataset_loss / j, dataset_accuracy / j

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

def from_pretrained(path: str, class_obj: torch.nn.Module) -> torch.nn.Module:
    """
    Loads a pretrained model from a state dict.

    Args:
        path (str): Path to state dict.
        num_classes (int, optional): Number of classes. Defaults to 4709.

    Returns:
        nn.Module: Model with loaded weights.
    """
    model = class_obj.__init__()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

@staticmethod
def saveWeights(model: torch.nn.Module, path: str=None) -> None:
    """
    Saves the model weights.

    Args:
        path (str): Path to save weights.
    """
    if path is None:
        path = f"/home/barti/PosterRecognition/vit/checkpoints/{model._get_name()}_state_dict.pth"
    print(f"Saving model to ---> {path}")
    torch.save(model.state_dict(), path)
    
def training_loop(model: Union[DinoV2Classifier,
                               ViTPosterClassifier,
                               DinoV2ClassifierMLP,
                               torch.nn.Module],
                    data: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    val_info: Tuple[float, float],
                    progress_bar: Callable = None) -> float:
    """
    Calculates the training loss over the whole dataset

    Args:
        model (torch.Module): Model to train.
        data (torch.utils.data.DataLoader): Data to train on.

    Returns:
        float: Training loss.
    """
    running_loss: float = .0
    running_accuracy: float = .0

    for j, (X, Y) in enumerate(data, 1):
        X = X.to('cuda')
        Y = Y.to('cuda')
        logits, loss = model.forward(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_accuracy += calculateAccuracy(logits, Y, return_percent=True)
        running_loss += loss.item()
        if progress_bar is not None:
            progress_bar.text = f"""Batch {j}/{len(data)} |
Training loss --> {running_loss / j:.4f} |
Training accuracy: {running_accuracy / j:.4f} |
Validation loss: {val_info[0]:.4f} |
Validation accuracy: {val_info[1]:.4f}"""
        del X, Y
    return running_loss / j, running_accuracy / j
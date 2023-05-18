import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PrototypicalNetwork(nn.Module):

    def __init__(self,
                 num_classes: int = 4000,
                 num_embeddings: int = 768,
                 feature_extractor: nn.Module = None) -> None:
        super().__init__()
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2',
                                                    'dinov2_vitb14')
        self.prototypes = nn.Parameter(torch.zeros((num_classes, num_embeddings)),
                                       requires_grad=False)

    def compute_prototypes(self,
                           X: torch.Tensor,
                           cls_idx: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # X --> (n_shot, 3, 224, 224)
        # cls_idx --> (cls_idx, )
        x = self.feature_extractor(X)

        self.prototypes.data[cls_idx] = x.mean(dim=0)
    
    
    def forward(self,
                X: torch.Tensor,
                y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate distances to prototypes.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).
            y (torch.Tensor, optional): Target tensor of shape (batch_size, 1). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of predictions and accuracy.
        """
        # X --> (batch_size, 3, 224, 224)
        # y --> (batch_size, 1)
        x = self.feature_extractor(X).mean(dim=0)
        preds = self.euclidean_distance(x, self.prototypes).argmin(dim=-1)

        accuracy = None
        if y is not None:
            accuracy = self.calculate_accuracy(preds, y)
        return preds, accuracy
    
    @staticmethod
    def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (predictions == targets).sum().float() / targets.shape[0]
        
        
    
    @staticmethod
    def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the euclidean distance between two tensors.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Euclidean distance between the two tensors of shape (batch_size).
        """
        # assert x1.shape == x2.shape, 'x1 and x2 must have the same shape'
        if x1.shape == x2.shape or any([x1.shape[0] == 1, x2.shape[0] == 1]):
            return (x1 - x2).pow_(2).sum(dim=-1).sqrt_()
        # Expand tensor x1 to match the shape of x2
        # Shape: (32, 1, 10)
        expanded_x1 = x1.unsqueeze(1).expand(-1, x2.shape[0], -1)

        # Expand tensor x2 to match the shape of x1
        # Shape: (1, 64, 10)
        expanded_x2 = x2.unsqueeze(0).expand(x1.shape[0], -1, -1)

        return torch.sqrt(torch.sum((expanded_x1 - expanded_x2) ** 2, dim=-1))
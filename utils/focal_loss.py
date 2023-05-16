import torch
import torch.nn.functional as F

def focal_loss(inputs: torch.Tensor,
               targets: torch.Tensor,
               gamma: float = 2,
               alpha: float = 0.25,
               weights: torch.Tensor = None,
               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute focal loss for multi-class classification.

    If weights is None then alpha is used to weight the losses,
    otherwise we use the weights given.

    For more details and informations on focal loss, see the paper
    https://arxiv.org/pdf/1708.02002.pdf

    Args:
        inputs (torch.Tensor): Input tensor with shape (B, C)

        targets (torch.Tensor): Target tensor with shape (B, )

        gamma (float, optional): Tunable focusing parameter. Defaults to 2.

        alpha (float, optional): Hyperparameter from paper. Defaults to 0.25.

        weights (torch.Tensor, optional): Tensor of weights to be applied. Defaults to None.

        reduction (str, optional): Whether to return tensor of shape (B, ), sum of losses or mean. Defaults to 'mean'.

    Returns:
        torch.Tensor: Focal loss.
    """
    assert reduction.lower() in ('mean', 'sum', 'none'),\
          ValueError("Reduction shall be: sum, mean or none")
    assert inputs.shape[0] == targets.shape[0], "No inputs != N.o targets"

    # negative log-likelihood
    cross_entropy = F.cross_entropy(inputs, targets, reduction='none')

    #inverse of nll i.e. e^x -> get probability
    p_t = torch.exp(-cross_entropy)

    # compute the loss
    if weights is not None:
        cross_entropy = cross_entropy * weights
    else:
        cross_entropy = cross_entropy * alpha
    loss = (1 - p_t) ** gamma * cross_entropy

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoClassifier(nn.Module):
    
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
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        features = self.dinov2.forward_features(x)
        linear_inpt = torch.cat([features['x_norm_clstoken'],
                                 features['x_norm_patchtokens'].mean(dim=1)], dim=-1)
        logits = self.linear(linear_inpt)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            logits, _ = self.forward(x)
            prediction = logits.softmax(dim=-1).argmax(dim=-1)
        self.train()
        return prediction
    
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

if __name__ == '__main__':
    dino = DinoClassifier(num_classes=9000)
    print(f'Number of parameters: {sum(p.numel() for p in dino.parameters()):,}')
import torch, pickle
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Union
from PIL import Image


class PosterCLIP(nn.Module):
    def __init__(self,
                 checkpoint: str = "openai/clip-vit-base-patch32",
                 device: str ='cpu',
                 inference: bool = True) -> None:
        super().__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(checkpoint)
        self.clip_model = CLIPModel.from_pretrained(checkpoint)
        self.device = device
        self.text_embeddings = None
        self.idx2class = None
        # self.load_embeddings("./text_embeddings.pt")
        if inference:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.to(self.device)
    
    
    def cache_text_embeddings(self, classes: List[str], batch_size: int = 512) -> torch.Tensor:
        """
        Caches the text embeddings for the classes.

        Args:
            classes (List[str]): List of classes with prompts.
            batch_size (int, optional): Size of the batch to process. Defaults to 512.

        Returns:
            torch.Tensor: Tensor of shape (len(classes), 512) containing the text embeddings.
        """
        text_embedd = []
        self.idx2class = {i: name for i, name in enumerate(classes)}
        for i in range(0, len(classes), batch_size):
            text_embedd.append(self.clip_model.get_text_features(**self.clip_processor(text=classes[i:i+batch_size],
                                            return_tensors="pt",
                                            padding=True).to(self.device)).detach())
        self.text_embeddings = torch.cat(text_embedd, dim=0)
        self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(p=2, dim=-1, keepdim=True)
        return torch.cat(text_embedd, dim=0)
    
    @torch.inference_mode()
    def predict(self,
                image: Union[torch.Tensor, Image.Image],
                temperature: float = 0.1,
                top_k_vals: int = 5) -> Dict[str, float]:
        """
        Predicts the classes for the given image.

        Args:
            image (Union[torch.Tensor, Image.Image]): Preprocessed image or PIL image.
            temperature (float, optional): Parameter which use to scale logits. Defaults to 0.1.
            top_k_vals (int, optional): Number of top probabilities. Defaults to 5.

        Raises:
            RuntimeError: If text embeddings are not cached.

        Returns:
            Dict[str, float]: Dictionary of classes and probabilities.
        """
        if isinstance(image, Image.Image):
            image = self.clip_processor(images=image,
                                        return_tensors="pt",
                                        padding=True).to(self.device)

        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings not cached. Call cache_text_embeddings() first.")
        image_features = self.clip_model.get_image_features(**image).to(self.device)
        text_embeds = self.text_embeddings
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        scale_param = torch.exp(torch.tensor(temperature, device=self.device))
        logits = (torch.matmul(image_features, text_embeds.T) * scale_param).view(-1)

        values, indices = torch.topk(logits, k=top_k_vals, dim=-1)
        return {self.idx2class[idx.item()]: values[i].item() for i, idx in enumerate(indices)}

    def save_embeddings(self, path: str) -> None:
        """
        Saves the text embeddings to the given path.

        Args:
            path (str): Path to save the embeddings.
        """
        torch.save(self.text_embeddings, path)
        with open(path + ".idx2class", "wb") as f:
            pickle.dump(self.idx2class, f)
    
    def load_embeddings(self, path: str) -> None:
        """
        Loads the text embeddings from the given path.

        Args:
            path (str): Path to load the embeddings.
        """
        self.text_embeddings = torch.load(path)
        with open(path + ".idx2class", "rb") as f:
            self.idx2class = pickle.load(f)
    

if __name__ == "__main__":
    import pandas as pd

    movies = pd.read_csv('../scraper/data/movies_with_posters_and_rich_desc.csv')

    poster_clip = PosterCLIP(device="cuda")
    id_to_name = {idx: movies.loc[movies['imdb_id'] == idx]['title'].values[0] for idx in movies['imdb_id']}
    prompts = [f"Poster of a movie: {name}" for name in id_to_name.values()]

    poster_clip.cache_text_embeddings(prompts)

    res = poster_clip.predict(Image.open("/home/barti/PosterRecognition/scraper/data/posters/tt0013442/test/eGUBhumQqHQqciJZJnpzEYA8LWT.jpg"))
    print(res)
        

    
import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, BatchEncoding
from typing import List, Dict, Union, Tuple
from PIL import Image


class PosterCLIP(nn.Module):
    def __init__(self,
                 checkpoint: str = "openai/clip-vit-base-patch32",
                 device: str ='cpu',
                 inference: bool = True,
                 default_path: str = './') -> None:
        super().__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(checkpoint)
        self.clip_model = CLIPModel.from_pretrained(checkpoint)
        self.device = device
        if inference:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.to(self.device)
        self.text_embeddings, self.idx2class = self.find_embeddings_file(default_path)
        if self.text_embeddings and self.idx2class:
            self.load_embeddings(self.text_embeddings, self.idx2class)
        else:
            self.text_embeddings = None
            self.idx2class = None
    
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
        return torch.cat(text_embedd, dim=0)
    
    def forward(self, 
                images: Union[BatchEncoding, List[Image.Image]],
                texts: Union[BatchEncoding, List[str]],
                temperature: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, list):
            images = self.clip_processor(images=images,
                                         return_tensors="pt",
                                         padding=True).to(self.device)
        if isinstance(texts, list):
            texts = self.clip_processor(text=texts,
                                        return_tensors="pt",
                                        padding=True).to(self.device)
        image_embeddings = self.clip_model.get_image_features(**images)
        text_embeddings = self.clip_model.get_text_features(**texts)

        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        scale = torch.exp(torch.tensor(temperature, device=self.device))
        logits = torch.matmul(text_embeddings, image_embeddings.T) * scale

        image_loss = F.cross_entropy(logits.t(), torch.arange(len(logits)))
        text_loss = F.cross_entropy(logits, torch.arange(len(logits)))
        return logits, (image_loss + text_loss) / 2

    
    @torch.inference_mode()
    def predict(self,
                image: Union[BatchEncoding, Image.Image],
                temperature: float = 0.1,
                top_k_vals: int = 5,
                use_cosine_simmilarities: bool = True) -> Dict[str, float]:
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
        scale_param = torch.exp(torch.tensor(temperature, device=self.device))
        
        if use_cosine_simmilarities:
            text_embeds = self.text_embeddings / self.text_embeddings.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            logits = (torch.matmul(text_embeds, image_features.T) * scale_param).t()
            logits = logits.softmax(dim=-1)
        else:
            logits = (torch.matmul(text_embeds, image_features.T) * scale_param).t()

        # print(logits.shape)
        values, indices = torch.topk(logits, k=top_k_vals, dim=-1)
        values, indices = values.view(-1), indices.view(-1)
        return {self.idx2class[idx.item()]: values[i].item() for i, idx in enumerate(indices)}

    def save_embeddings(self,
                        path: str) -> None:
        """
        Saves the text embeddings to the given path.

        Args:
            path (str): Path to save the embeddings.
        """
        torch.save(self.text_embeddings, path)
        with open(path + ".idx2class", "wb") as f:
            pickle.dump(self.idx2class, f)
    
    def load_embeddings(self,
                        embeddings_path: str,
                        idx2class_path: str) -> None:
        """
        Loads the text embeddings from the given path.

        Args:
            path (str): Path to load the embeddings.
        """
        self.text_embeddings = torch.load(embeddings_path)
        with open(idx2class_path, "rb") as f:
            self.idx2class = pickle.load(f)
        
    @staticmethod
    def find_embeddings_file(path: str = None):
        """
        Finds the embeddings file in the given path.

        Args:
            path (str, optional): Path to search. Defaults to None.

        Returns:
            str: Path to the embeddings file.
        """
        if path is None:
            path = "./"
        embeddings, idx2class = None, None
        for file in os.listdir(path):
            if file.endswith(".pt"):
                embeddings = file
            elif file.endswith(".idx2class"):
                idx2class = file
        return embeddings, idx2class
    

if __name__ == "__main__":
    import pandas as pd
    import os, alive_progress

    POSTERS_PATH = "/home/barti/PosterRecognition/scraper/data/posters/"
    posters = os.listdir(POSTERS_PATH)

    print("Loading data...")
    movies = pd.read_csv('../scraper/data/movies_with_posters_and_rich_desc.csv')

    print("Creating prompts...")
    id_to_name = {idx: movies.loc[movies['imdb_id'] == idx]['title'].values[0] for idx in movies['imdb_id']}
    name_to_id = {name: idx for idx, name in id_to_name.items()}
    prompts = [f"Poster of a movie: {name}" for name in id_to_name.values()]

    print("Loading model...")
    poster_clip = PosterCLIP(device="cuda")

    print("Caching text embeddings...")
    # poster_clip.cache_text_embeddings(prompts)
    # poster_clip.save_embeddings("./text_embeddings.pt")

    # poster_clip.cache_text_embeddings(prompts)
    num_classes = len(prompts)
    count = 0

    with alive_progress.alive_bar(len(posters)) as bar:
        for i, poster_id in enumerate(posters):
            path = os.path.join(POSTERS_PATH, poster_id, "test")
            images = os.listdir(path)
            image = Image.open(os.path.join(path, images[0]))
            res = poster_clip.predict(image, top_k_vals=1, temperature=0)
            if id_to_name[poster_id] in list(map(lambda val: val.replace("Poster of a movie: ", ""),res.keys())):
                count += 1
            bar()
    
    print(f"Top-1 Accuracy: {(count / len(posters)*100):.4f}")
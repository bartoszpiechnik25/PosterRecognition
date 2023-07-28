import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, BatchEncoding
from typing import List, Dict, Union, Tuple
from collections import OrderedDict
from PIL import Image
import os


class PosterCLIP(nn.Module):
    def __init__(self,
                 checkpoint: str = "openai/clip-vit-base-patch32",
                 device: str ='cpu',
                 inference: bool = True,
                 path: str='./model') -> None:
        super().__init__()
        self.text_embeddings = None
        self.idx2class = None
        if os.path.isdir(path) and os.listdir(path):
            self.clip_processor = CLIPProcessor.from_pretrained(path + "/clip_processor")
            self.clip_model = CLIPModel.from_pretrained(path + "/clip_model")
            self.text_embeddings = torch.load(f'{path}/text_embeddings.pt', map_location=device)
            self.idx2class = torch.load(f'{path}/idx2class.pt', map_location=device)
        else:
            self.clip_processor = CLIPProcessor.from_pretrained(checkpoint)
            self.clip_model = CLIPModel.from_pretrained(checkpoint)
        if inference:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.device = device
        self.to(device)
    
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
                text_embeddings: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, list):
            images = self.clip_processor(images=images,
                                         return_tensors="pt",
                                         padding=True).to(self.device)
        if isinstance(texts, list):
            texts = self.clip_processor(text=texts,
                                        return_tensors="pt",
                                        padding=True).to(self.device)
        image_embeddings = self.clip_model.get_image_features(**images)
        if text_embeddings is None:
            text_embeddings = self.clip_model.get_text_features(**texts)

        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        scale = self.clip_model.logit_scale.exp()
        logits = torch.matmul(text_embeddings, image_embeddings.T) * scale

        image_loss = F.cross_entropy(logits.t(), torch.arange(len(logits), device=self.device))
        text_loss = F.cross_entropy(logits, torch.arange(len(logits), device=self.device))

        return logits, (image_loss + text_loss) / 2

    
    @torch.inference_mode()
    def predict(self,
                image: Union[BatchEncoding, Image.Image],
                top_k_vals: int = 5,
                use_cosine_simmilarities: bool = True) -> List[Dict[str, float]]:
        """
        Predicts the classes for the given image.

        Args:
            image (Union[torch.Tensor, Image.Image]): Preprocessed image or PIL image.
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
        
        # image_features = self.clip_model.get_image_features(**image).to(self.device)
        # text_embeds = self.text_embeddings
        # scale_param = self.clip_model.logit_scale.exp()
        
        # if use_cosine_simmilarities:
        #     text_embeds = self.text_embeddings / self.text_embeddings.norm(p=2, dim=-1, keepdim=True)
        #     image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        #     logits = (torch.matmul(text_embeds, image_features.T) * scale_param).t()
        #     logits = logits.softmax(dim=-1)
        # else:
        #     logits = (torch.matmul(text_embeds, image_features.T) * scale_param).t()
        #     logits = logits.softmax(dim=-1)
        logits, loss = self.forward(image, text_embeddings=self.text_embeddings)

        values, indices = torch.topk(logits, k=top_k_vals, dim=-1)
        res = []

        for i, index in enumerate(indices):
            res.append(OrderedDict((self.idx2class[ids.item()], values[i][j].item()) for j, ids in enumerate(index)))
        return res, loss
    
    def save(self, save_directory):
        self.clip_model.save_pretrained(save_directory+ "/clip_model")
        self.clip_processor.save_pretrained(save_directory + "/clip_processor")
        torch.save(self.text_embeddings, f'{save_directory}/text_embeddings.pt', map_location='cpu')
        torch.save(self.idx2class, f'{save_directory}/idx2class.pt', map_location='cpu')

    # @staticmethod
    # def load(path: str) -> 'PosterCLIP':
    #     """
    #     Loads the model.

    #     Args:
    #         path (str): Path to load the model.

    #     Returns:
    #         torch.nn.Module: Model with loaded state dict.
    #     """
    #     return PosterCLIP.from_pretrained(path)
    
    # @staticmethod
    # def load_pretrained(path: str="./model/clip.pt") -> 'PosterCLIP':
    #     """
    #     Loads the state dict of the model.

    #     Args:
    #         path (str): Path to load the state dict.

    #     Returns:
    #         torch.nn.Module: Model with loaded state dict.
    #     """
    #     return PosterCLIP.from_pretrained(path)
    

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
    # poster_clip.save('./model')

    # poster_clip.cache_text_embeddings(prompts)
    num_classes = len(prompts)
    top_1 = top_3 = top_5 = 0

    with alive_progress.alive_bar(len(posters)) as bar:
        for i, poster_id in enumerate(posters):
            path = os.path.join(POSTERS_PATH, poster_id, "test")
            images = os.listdir(path)
            image = Image.open(os.path.join(path, images[0]))
            res = poster_clip.predict(image, top_k_vals=5, temperature=0)
            result_classes = list(map(lambda val: val.replace("Poster of a movie: ", ""),res.keys()))
            if id_to_name[poster_id] in result_classes[:1]:
                top_1 += 1
            if id_to_name[poster_id] in result_classes[:3]:
                top_3 += 1
            if id_to_name[poster_id] in result_classes:
                top_5 += 1
            bar()
    
    print(f"Top-1 Accuracy: {(top_1 / len(posters)*100):.2f}%")
    print(f"Top-3 Accuracy: {(top_3 / len(posters)*100):.2f}%")
    print(f"Top-5 Accuracy: {(top_5 / len(posters)*100):.2f}%")
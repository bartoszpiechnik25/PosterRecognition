from clip.poster_clip import PosterCLIP
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_manipulation import PosterDataset
from transformers import CLIPProcessor
from typing import List, Dict

def validation_loop(model: PosterCLIP,
                    validation_loader: DataLoader,
                    idxToPrompt: Dict[int, str],
                    text_backbone: List[str]) -> float:
    model.eval()
    model.cache_text_embeddings(classes=text_backbone)
    with torch.no_grad():
        val_accuracy = .0
        val_loss = .0
        for count, batch in enumerate(validation_loader, 1):                      
            image_features = {
                "pixel_values": batch[2].to('cuda'),
            }
            results, loss = model.predict(
                image=image_features,
                top_k_vals=1
            )
            classes = batch[3].tolist()
            val_loss += loss.item()
            for i, result in enumerate(results):
                if idxToPrompt[classes[i]] == list(result.keys())[0]:
                    val_accuracy += 1
    model.train()
    return val_accuracy / count, val_loss / count

if __name__ == "__main__":
    path = '/home/barti/PosterRecognition'
    dataset_path = path + '/scraper/data/posters'
    model = PosterCLIP(device="cuda")
    train=PosterDataset(dataset_path, processor=model.clip_processor)
    train_loader = DataLoader(
        train,
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )
    for batch in train_loader:
        text_features = {
            "input_ids": batch[0].to('cuda'),
            "attention_mask": batch[1].to('cuda'),
        }                        
        image_features = {
            "pixel_values": batch[2][:2].to('cuda'),
        }
        _, loss = model.forward(
            images=image_features,
            texts=text_features,
        )
    
    validation_loop(
        model,
        train_loader,
        train.idToPrompt,
        list(train.idToPrompt.values())
    )

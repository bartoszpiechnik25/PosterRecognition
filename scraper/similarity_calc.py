import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import hashlib, numpy as np, os
from bing_image_downloader import downloader

def getTransform() -> T.Compose:
    """
    Returns a composition of transforms to be applied to the image.

    Returns:
        T.Compose: Composition of transforms.
    """
    return T.Compose([T.Resize(256),
                      T.CenterCrop(224),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    ])

def getResNet50() -> torch.nn.Module:
    """
    Returns a pretrained ResNet50 model.

    Returns:
        torch.nn.Module: Model.
    """
    similarity_resnet = torch.nn.Sequential(
        *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        )
    similarity_resnet.eval()
    return similarity_resnet

def getSimilarity(img1: Image,
                  img2: Image,
                  model: torch.nn.Module,
                  transform: T.Compose) -> float:
    """
    Returns the similarity between two images.

    Args:
        img1 (Image): First image.
        img2 (Image): Second image.

    Returns:
        float: Similarity between the two images.
    """
    #add batch dimension
    img1 = torch.unsqueeze(transform(img1), 0)
    img2 = torch.unsqueeze(transform(img2), 0)

    #disable gradient calculation
    with torch.inference_mode():

        #get embeddings
        img1 = model(img1).squeeze()
        img2 = model(img2).squeeze()

        #calculate similarity
        similarity = torch.cosine_similarity(img1, img2, dim=-1)

    return similarity.item()


def dhash(img):
    img = img.cpu().detach().numpy()[0]
    diff = img[:, 1:] > img[:, :-1]
    return hashlib.sha256(np.packbits(diff)).hexdigest()

if __name__ == "__main__":
    pp = "/home/barti/Downloads/asdf"

    imgs = {title: Image.open(os.path.join(pp, title)) for title in os.listdir("/home/barti/bing_image_downloader/bing_image_downloader/dog")}

    model = getResNet50()
    transform = getTransform()
    
    #compare all images from dictionary
    for title1, img1 in imgs.items():
        for title2, img2 in imgs.items():
            similarity = getSimilarity(img1, img2, model, transform)
            print(f"{title1} - {title2}: {similarity}")
    
    # #download images
    # downloader.download("the frozen ground various movie posters",
    #                     limit=20,
    #                     output_dir='/home/barti/Downloads',
    #                     adult_filter_off=True,
    #                     force_replace=False,
    #                     timeout=60)
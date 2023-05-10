import os, sys, shutil, random
from alive_progress import alive_bar
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import ViTImageProcessor

def split_train_val_test(path: str) -> None:
    """
    Split each class into subdirectories for train, val, and test.

    Args:
        path (str): Path to directory containing images.

    Raises:
        AttributeError: If path is not a directory.
    """
    if not os.path.isdir(path):
        print(f"{path} is not a directory!", file=sys.stderr)
        return

    #get list of posters and shuffle
    posters = os.listdir(path)

    #if train, val, or test is in the posters return
    if any([split in posters for split in ('train', 'val', 'test')]):
        print(f"{path} already split!", file=sys.stderr)
        return

    random.shuffle(posters)
    num_posters = len(posters)
    
    #create train, val, test splits
    train_end_idx = int(num_posters * 0.7)
    val_end_idx = int(num_posters * 0.9)

    split_paths = [os.path.join(path, split)\
                   for split in ('train', 'val', 'test')]
    
    #move posters into split directories
    for poster_imgs, split in zip([posters[:train_end_idx],
                            posters[train_end_idx:val_end_idx],
                            posters[val_end_idx:]], split_paths):
        os.makedirs(split)
        for poster in poster_imgs:
            shutil.move(os.path.join(path, poster),
                        os.path.join(split, poster))
            

class PosterDataset(Dataset):

    def __init__(self, datasetRootPath: str,
                 transform: Callable=None,
                 split: str='train') -> None:
        """
        Initializes a PosterDataset object.

        Args:
            datasetRootPath (str): Path to directory containing classes.

            transform (Callable, optional): Augumentation function. Defaults to None.

            split (str, optional): Type of split for training. Defaults to 'train'.

        Raises:
            ValueError: If split is do not indicate train, val, or test split
            of the dataset.
        """
        
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"{split} not in (train', val, test)!")
        
        self.transform = transform

        self.classToIdx = {cls: idx \
            for idx, cls in enumerate(sorted(os.listdir(datasetRootPath)))}
        
        self.idxToClass = {idx: cls for cls, idx in self.classToIdx.items()}

        self._postersPaths = []
        for className in self.classToIdx.keys():
            clsPath: str = os.path.join(datasetRootPath, className, split) 
            dirnames: list =\
            [[os.path.join(clsPath, poster), self.classToIdx[className]]\
             for poster in os.listdir(clsPath)]
            self._postersPaths.extend(dirnames)
        
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a tuple of image and label.

        Args:
            index (int): Index from which to return image and label.

        Returns:
            Tuple[torch.Tensor, int]: Tuple of image and label.
        """
        x, y = self._postersPaths[index]

        image = Image.open(x).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(y, dtype=torch.long)
    
    def __len__(self) -> int:
        """
        Returns the number of all posters in the dataset.

        Returns:
            int: Number of all posters in the dataset.
        """
        return len(self._postersPaths)

    def idxToClassName(self, idx: int) -> str:
        """
        Returns the class name for a given index.

        Args:
            idx (index): Index of the class.

        Returns:
            str: Class name.
        """
        return self.idxToClass[idx]

    @property
    def getNumClasses(self) -> int:
        """
        Returns the number of classes in the dataset.

        Returns:
            int: Number of classes in the dataset.
        """
        return len(self.classToIdx)
    
    @staticmethod
    def getTrainTransforms(img_size: int = 224) -> torch.Tensor:
        """
        Returns a composition of transforms for the dataset.

        Returns:
            T.Compose: Composition of transforms.
        """
        return T.Compose([
                    T.Resize((img_size, img_size), antialias=True),
                    T.CenterCrop(img_size),
                    T.RandomRotation(degrees=30),
                    T.RandomPerspective(distortion_scale=0.2),
                    T.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3, saturation=0.3),
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
    
    @staticmethod
    def getValTransforms(img_size: int = 224) -> torch.Tensor:
        """
        Returns a composition of transforms for the dataset.

        Returns:
            T.Compose: Composition of transforms.
        """
        return T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], 
                        std=[0.5, 0.5, 0.5])
        ])

class Transform:
    def __init__(self, image_size: int=224, transform: Callable=None) -> None:
        self.image_size = image_size
        self.transform = transform
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.transform is not None:
            imgs = self.transform(imgs)
        return self.vit_processor(imgs,
                                  return_tensors='pt')['pixel_values'][0]

def plot_learning(title: str, x_label: str, y_label: str, epochs: List[int], **kwargs) -> None:
    plt.figure(figsize=(11,7))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(epochs, kwargs['train_loss'], label="Training loss")
    plt.plot(epochs, kwargs['validation_loss'], label="Validation loss")
    plt.legend()
    plt.savefig(path + f"/vit/{title}.png")

            
if __name__ == '__main__':
    import pandas as pd
    path = "/home/barti/PosterRecognition/scraper/data"
    data = pd.read_csv(path + "/movies_with_posters_and_rich_desc.csv")
    csv_dirs = set(data.loc[data['poster_count'] > 4]['imdb_id'].values)
    print(len(csv_dirs))

    path = "/home/barti/PosterRecognition/scraper/data/posters"
    dirs = set(os.listdir(path)).intersection(csv_dirs)

    with alive_bar(len(dirs)) as bar, open(path + "logs.txt", 'w+') as f:
        for directory in dirs:
            try:
                split_train_val_test(os.path.join(path, directory))
            except:
                f.write(f"Error occured while processing -> {directory}\n")
            finally:
                bar()
    # from torch.utils.data import DataLoader
    # path = '/home/barti/PosterRecognition'
    # train_transform = Transform(transform=PosterDataset.getTrainTransforms())
    # val_transform = Transform(transform=PosterDataset.getValTransforms())

    # train_dataset = PosterDataset(path + '/scraper/data/images',
    #                             transform=train_transform)
    # val_dataset = PosterDataset(path + '/scraper/data/images',
    #                             transform=val_transform,
    #                             split='val')
    # test_dataset = PosterDataset(path + '/scraper/data/images',
    #                             transform=val_transform,
    #                             split='test')
    # EPOCHS = 5

    # train_loader = DataLoader(train_dataset,
    #                         batch_size=16,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         num_workers=4)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=16,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         num_workers=4)
    # test_loader = DataLoader(test_dataset,
    #                         batch_size=16,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         num_workers=4)

    # for epoch in range(EPOCHS):
    #     for batch, (X, Y) in enumerate(train_loader):
    #         X = X.to('cuda')
    #         Y = Y.to('cuda')
    #         print(X.shape, Y.shape)
    #         print(f"Batch {batch+1}/{len(train_loader)}")
    #         del X, Y

            



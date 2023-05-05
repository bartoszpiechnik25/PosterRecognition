import os, sys, shutil, random
from alive_progress import alive_bar
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Callable, Tuple
import torch
from PIL import Image

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

        image = Image.open(x)

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

    def getNumClasses(self) -> int:
        """
        Returns the number of classes in the dataset.

        Returns:
            int: Number of classes in the dataset.
        """
        return len(self.classToIdx)
    
    @staticmethod
    def getTrainTransforms() -> T.Compose:
        """
        Returns a composition of transforms for the dataset.

        Returns:
            T.Compose: Composition of transforms.
        """
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
        ])
    
    @staticmethod
    def getValTransforms() -> T.Compose:
        """
        Returns a composition of transforms for the dataset.

        Returns:
            T.Compose: Composition of transforms.
        """
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
        ])

            
if __name__ == '__main__':
    path = "/home/barti/PosterRecognition/scraper/data/images"
    dirs = os.listdir(path)
    with alive_bar(len(dirs)) as bar, open(path + "logs.txt", 'w+') as f:
        for directory in dirs:
            try:
                split_train_val_test(os.path.join(path, directory))
            except:
                f.write(f"Error occured while processing -> {directory}\n")
            finally:
                bar()

            



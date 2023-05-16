import os, pandas as pd
import tmdbsimple as tmdb
from similarity_calc import getResNet50, getSimilarity, getTransform
from PIL import Image
from io import BytesIO
from urllib.request import urlopen, Request
from collections import defaultdict
import torch, sys
import torchvision.transforms as T
from alive_progress import alive_bar
import concurrent.futures
from typing import List, Callable



class TMDBPosterDownloader:
    def __init__(self,
                 model: torch.nn.Module,
                 transform: T.Compose,
                 imdb_id: str,
                 poster_urls: List[str],
                 filenames: List[str]) -> None:
        self.downloaded_images: defaultdict[str, Image.open] = defaultdict()
        self.model = model
        self.transforms = transform
        self.path = os.path.join('/home/barti/PosterRecognition/scraper', 'data', 'posters', imdb_id)
        self.poster_urls = poster_urls
        self.filenames = filenames
    
    def download_and_process_posters(self, bar: Callable) -> None:
        """
        Download and process posters from TMDB API.

        Args:
            bar (Callable): Progress bar.
        """
        os.makedirs(self.path, exist_ok=True)
        for i, (url, filename) in enumerate(zip(self.poster_urls, self.filenames), 1):
            try:
                self.save_image(url, filename)
            except Exception as e:
                print(f'Error downloading poster {url}: {e}', file=sys.stderr)
                continue
            if i == 40:
                break
        bar()

    def save_image(self, link: str, filename: str) -> None:
        """
        Save image from given link.

        Args:
            link (str): Link to the image.
            filename (str): Filename of the image.
        """
        request = Request(link, None)
        image = urlopen(request, timeout=10).read()
        img = Image.open(BytesIO(image)).convert('RGB')
        img = self.checkSimilarities(img, filename).save(self.path + filename, 'JPEG')

    def checkSimilarities(self, image: Image, name: str) -> Image:
        """
        Check if image is too similar to already downloaded images.

        Args:
            image (Image): Candidate image.
            name (str): Name of the candidate image.

        Raises:
            ValueError: If image is too similar to already downloaded images.

        Returns:
            Image: Image to be saved.
        """
        for img_name, img in self.downloaded_images.items():
            if (similarity := getSimilarity(image, img, self.model, self.transforms)) > 0.8:
                raise ValueError(f'Image is too similar to {img_name} --> {similarity}')
        self.downloaded_images[name] = image
        return image
    

if __name__ == '__main__':
    key = os.environ.get('TMDB_API_KEY')
    tmdb.API_KEY = key
    imdb_ids = list(pd.read_csv('data/movies_with_posters_and_rich_desc.csv')['imdb_id'].values)
    poster_url = 'https://image.tmdb.org/t/p/w342/{}'
    resnet = getResNet50()
    transform = getTransform()

    with alive_bar(len(imdb_ids[:1000]), dual_line=True) as bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for imdb_id in imdb_ids[:1000]:
                movie = tmdb.Movies(imdb_id)
                images = movie.images(language='en')['posters']
                filenames = list(map(lambda x: x['file_path'], images))
                posters = TMDBPosterDownloader(resnet, transform, imdb_id, [poster_url.format(f) for f in filenames], filenames)
                futures.append(executor.submit(posters.download_and_process_posters, bar))
            for future in concurrent.futures.as_completed(futures):
                future.result()

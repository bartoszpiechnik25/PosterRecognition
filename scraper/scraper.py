from urllib.request import urlopen, urlretrieve
from bs4 import BeautifulSoup
import re, pandas as pd, os, time, random, sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from alive_progress import alive_bar

def download(url: str, movie_id: str) -> None:
    """
    Download and save .jpg file from specified URL.

    Args:
        url (str): Image url.
        movie_name (str): Name of the movie.
    """
    filename = f"./data/images/{movie_id}/{url.split('/')[-1]}"
    os.makedirs(f"./data/images/{movie_id}", exist_ok=True)
    urlretrieve(url, filename)
    with Image.open(filename) as image:
        image.save(filename, "JPEG")

def scrape(movie_specs: dict) -> None:
    """
    Scrape images from movieposterdb.com.

    Args:
        movie_specs (dict): Dictionary containing movie title and imdb id.
    """
    title, id_ = movie_specs['title'], movie_specs['imdb_id'].lstrip("0")

    # Get all images from movieposterdb.com
    url = f'https://www.movieposterdb.com/{title}-i{id_}?v=posters&type_id=1&country_code=US&min_width=&min_height=&min_size='
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    regex = rf'data-src="https://.+/{id_}/.+\.jpg"'
    pattern = re.compile(regex)
    images: list[str] = re.findall(pattern, soup.prettify())

    # Shuffle images to get different images for each movie
    random.shuffle(images)
    num_img: int = 10 if len(images) > 9 else len(images)

    # Wait for 2 seconds to avoid getting blocked
    time.sleep(2)

    # Download images
    for i in range(num_img):
        string_url = images[i].replace('"', "")
        string_url = string_url.replace("data-src=", "")
        download(string_url, id_)

if __name__ == '__main__':

    # Get movie titles and imdb ids
    movies = pd.read_csv("./data/movies_metadata.csv")
    movies = movies.loc[:, ['title', 'imdb_id']].drop_duplicates(keep=False).dropna()
    movies['title'] = movies['title'].apply(lambda x: x.lower().replace(" ", "-"))
    movies['imdb_id'] = movies['imdb_id'].apply(lambda x: x.replace("tt", ""))
    title_imdb = movies.to_dict('records')

    # with ThreadPoolExecutor() as executor, alive_bar(len(title_imdb)) as progress:
    #     futures = []
    #     for movie in title_imdb:
    #         futures.append(executor.submit(scrape, movie, progress))
    #     for future in as_completed(futures):
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(f"Exception occurred: {e}")

    # Scrape images with progress bar using alive_progress shown in the terminal
    with alive_bar(len(title_imdb)) as progress:
        for movie in title_imdb:
            try:
                scrape(movie)
            except Exception as e:
                print(f"Exception occurred: {e}", file=sys.stderr)
            finally:
                progress()
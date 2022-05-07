from pytorch_lightning import LightningDataModule
from os.path import exists
from os import getcwd
from urllib.request import urlretrieve
from zipfile import ZipFile
from typing import List, Sequence
from torch.utils.data import random_split, Subset
from progressbar import ProgressBar

from ImageTextDataset import ImageTextDataset


class DownloadProgressBar():
    def __init__(self):
        self.progress_bar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.progress_bar:
            self.progress_bar = ProgressBar(maxval=total_size)
            self.progress_bar.start()

        downloaded = block_num * block_size

        if downloaded < total_size:
            self.progress_bar.update(downloaded)
        else:
            self.progress_bar.finish()


class ImageTextDataModule(LightningDataModule):

    def __init__(self) -> None:
        super().__init__()

        self.image_size = 224

    @staticmethod
    def split_dataset(dataset: ImageTextDataset) -> List[Subset[ImageTextDataset]]:
        size = len(dataset)

        # get 70% for the train set
        train_size = int(size // 1.25)

        # 20% for test set
        test_size = int(size // 5)

        # get 10% for val set
        val_size = int(size - train_size - test_size)

        lengths: Sequence = [train_size, test_size, val_size]

        return random_split(dataset, lengths)

    @staticmethod
    def download_if_not_exists(directory: str, url: str) -> None:
        if not exists(directory):
            zipFile = url.split('/')[-1]

            # 2. Make sure zip file is downloaded
            if not exists(zipFile):
                print(f"Downloading {zipFile} from {url}...")
                urlretrieve(url, zipFile, DownloadProgressBar())
                print("Done!\n")

            print(f"Unzipping {zipFile}...")

            # 3. extract zip contents to directory
            with ZipFile(zipFile, 'r') as data:
                data.extractall(getcwd())

            print("Done!\n")

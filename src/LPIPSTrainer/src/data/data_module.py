from typing import Tuple
from random import randint
from pytorch_lightning.core.saving import CHECKPOINT_PAST_HPARAMS_KEYS

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, RandomAffine, Resize, RandomCrop, ToTensor, Normalize
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl


class PatternDataset(Dataset):
    """
    PatternDataset - Custom dataset, the dataset structure should be the same with ImageFolder dataset from torch.utils.data
    """
    def __init__(self, folder: str, random_size: Tuple[int, int]=(150, 512), resize_size: int = 256, len: int = 10000):
        """
        __init__ Create PatternDataset

        Args:
            folder (str): path to dataset folder, the dataset structure should be the same with ImageFolder dataset
            random_size (Tuple[int, int], optional): the random crop size for transforms augmentation. Defaults to (150, 512).
            resize_size (int, optional): input size to the network. Defaults to 256.
            len (int, optional): len of dataset. Defaults to 10000.
        """
        self.dataset = ImageFolder(folder)
        self.tfs = Compose([
            RandomAffine(degrees=15),
            ToTensor(),
            Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        
        self.resize = (resize_size, resize_size)
        self.random_size = random_size
        self.len = len
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        """
        __getitem__ Choosing random 2 image and augment them, 1/5 of the pairs is labled as True

        Args:
            index ([type]): index of data

        Returns:
            dict: dict(img_1: torch.Tensor, img_2: torch.Tensor, label: torch.tensor)
        """
        chosen_index = index%len(self.dataset)
        img0 = self.dataset[chosen_index]

        if index%5 == 0:
            img1 = self.dataset[chosen_index]
            label = 0.
        
        else:
            second_index = randint(0, len(self.dataset) - 1)
            while second_index==chosen_index:
                second_index = randint(0, len(self.dataset) - 1)

            img1 = self.dataset[second_index]
            label = 1.
        
        cache_size = self.random_size
        w, h = img0[0].size
        w1, h1 = img1[0].size
        img_min_size = min(w, h, w1, h1)
        cache_size = (cache_size[0], min(img_min_size, cache_size[1]) - 10)
        
        cache_size = randint(*cache_size)
        
        post_tfs = Compose([
        RandomCrop((cache_size, cache_size)),
        Resize(self.resize)  
        ])

        result = dict(
            img_1=post_tfs(self.tfs(img0[0])),
            img_2=post_tfs(self.tfs(img1[0])),
            label=torch.tensor(label)
        )
        
        return result
        

class PatternDataModule(pl.LightningDataModule):
    """
    PatternDataModule Pytorch Lightning Data Module for training with pytorch lightning
    """

    def __init__(self, folder: str, 
                 train_batch_size: int = 8, 
                 test_batch_size: int = 8, 
                 train_len: int = 1000, 
                 test_len: int = 200, 
                 random_size = (150, 512), 
                 resize_size: int = 256):
        """
        __init__ Create PatternDataModule

        Args:
            folder (str): path to dataset folder, the dataset structure should be the same with ImageFolder dataset
            train_batch_size (int, optional): training batch size, app the the validation step also. Defaults to 8.
            test_batch_size (int, optional): testing batch size. Defaults to 8.
            train_len (int, optional): length of training dataset, validation dataset's length will be 1/10*train_len. Defaults to 1000.
            test_len (int, optional): length of testing dataset. Defaults to 200.
            random_size (tuple, optional): the random crop size for transforms augmentation. Defaults to (150, 512).
            resize_size (int, optional): input size to the network. Defaults to 256.
        """
        
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        self.folder = folder
        
        self.train_len = train_len
        self.val_len = train_len//5
        self.test_len = test_len
        
        self.random_size = random_size
        self.resize_size = resize_size

    def setup(self, stage=None):
        self.train_dataset = PatternDataset(
            self.folder,
            random_size=self.random_size,
            resize_size=self.resize_size,
            len=self.train_len
        )

        self.val_dataset = PatternDataset(
            self.folder,
            random_size=self.random_size,
            resize_size=self.resize_size,
            len=self.val_len
        )

        self.test_dataset = PatternDataset(
            self.folder,
            random_size=self.random_size,
            resize_size=self.resize_size,
            len=self.test_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=28
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=28
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=28
        )


import collections
import os
import sys
from typing import List, Dict

import torch

sys.setrecursionlimit(10**6)

from PIL import Image
import numpy as np

import webcolors

class ColorModel:
    def __init__(self, samples_dir: str) -> None:
        self.samples_dir: str = samples_dir
        self.lib = self._load_samples()
    
    def _load_samples(self):
        imgs = os.listdir(self.samples_dir)
        lib = collections.defaultdict(list)
        for img in imgs:
            img_path = os.path.join(self.samples_dir, img)
            img_name = img.split('.')[0]
            img = Image.open(img_path).convert('RGB')
            color = self.get_color(img)
            lib[img_name] = color

        return lib
    
    def change_samples(self, samples_dir: str):
        self.samples_dir = samples_dir
        self.lib = self._load_samples()
    
    @staticmethod
    def get_central(arr: np.array) -> List:
        """
        get_central Find the central point from an array

        Args:
            arr (np.array): Array colors to process

        Returns:
            List: The central color from array
        """

        central = np.mean(arr, axis=0)
        dis = np.array([np.linalg.norm(p - central) for p in arr])

        mean_dist = np.mean(dis)
        is_not_outlier = [dis < mean_dist * 1.5]
        arr = arr[is_not_outlier]
        central = np.mean(arr, axis=0)

        return central

    @staticmethod
    def get_color(img: Image) -> List[int]:
        """
        get_color Extract the main color of image.

        Args:
            img (Image): image to process

        Returns:
            List[int]: the main color of image
        """

        size = img.size
        img = img.resize((28, 28))
        colors = img.getcolors(28 * 28)
        colors = [list(c[1]) for c in colors]

        return [int(c) for c in ColorModel.get_central(np.array(colors))]

    def get_closest_color(self, color: List) -> str:
        """
        closest_color Find closest predefined color

        Args:
            color (List): The color in RGB

        Returns:
            str: The name of color
        """

        min_colors = {}
        for key, value in self.lib.items():
            r_c, g_c, b_c = value
            rd = (r_c - color[0]) ** 2
            gd = (g_c - color[1]) ** 2
            bd = (b_c - color[2]) ** 2
            min_colors[(rd + gd + bd)] = key

        return min_colors[min(min_colors.keys())]

    def find_color(self, img: Image) -> Dict:
        """
        find_color Find the main color with its code and name.

        Args:
            img (Image): input image to process

        Returns:
            Dict: dict(rgb: [R, G, B], color_name: str)
        """

        rgb = self.get_color(img)
        color_name = self.get_closest_color(rgb)

        return dict(rgb=rgb, hex=webcolors.rgb_to_hex(rgb), color_name=color_name)
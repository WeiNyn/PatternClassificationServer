import lpips

import torch
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder

from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image
import webcolors



class PatternClassifier:
    """
     Main class for pattern classification
    """
    def __init__(
        self,
        sample_dir: str,
        sample_size: Tuple[int, int] = (512, 512),
        model_path: Optional[str] = None,
        device: str = "cuda",
        net: str = "alex",
    ):
        """
        __init__ Create the PatternClassifier class

        Args:
            sample_dir (str): path to pattern samples
            sample_size (Tuple[int, int]): size of stored sample images
            model_path (Optional[str], optional): path to pretrained model. Defaults to None.
            device (str, optional): device to use. Defaults to "cuda".
            net (str, optional): backbone for Perceptual Similarity. Defaults to "alex".
        """
        
        self.dataset = ImageFolder(sample_dir, transform=tfs.Compose([tfs.ToTensor(), tfs.CenterCrop(sample_size)]))

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net, version="0.1", model_path=model_path)
        self.model = self.model.to(self.device)

        self.sample_dir = sample_dir

        self.tfs = tfs.Compose(
            [tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self._get_features()

    def _get_features(self):
        """
        _get_features Create sample features for pattern matching.
        """
        
        images = []
        labels = []

        for i in range(len(self.dataset)):
            image, label = self.dataset[i]
            images.append(image)
            labels.append(label)

        self.images = torch.stack(images).to(self.device)
        self.labels = labels

        del images
        del labels

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

        return [int(c) for c in PatternClassifier.get_central(np.array(colors))]

    @staticmethod
    def closest_color(color: List) -> str:
        """
        closest_color Find closest predefined color

        Args:
            color (List): The color in RGB

        Returns:
            str: The name of color
        """
        
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - color[0]) ** 2
            gd = (g_c - color[1]) ** 2
            bd = (b_c - color[2]) ** 2
            min_colors[(rd + gd + bd)] = name

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
        color_name = self.closest_color(rgb)

        return dict(rgb=rgb, color_name=color_name)

    def find_pattern(self, img: Image) -> Dict:
        """
        find_pattern Find the matching pattern from given samples

        Args:
            img (Image): Image to process

        Returns:
            Dict: dict(query: np.array, result: Image, label: int, sim: float)
        """
        
        size = img.size
        if size[0] > 256:
            size = (256, 256)
            img = tfs.CenterCrop(size)(img)

        img = self.tfs(img)
        img = img.repeat(len(self.images), 1, 1, 1)

        images = [
            tfs.Compose(
                [tfs.CenterCrop(size), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )(image)
            for image in self.images
        ]

        with torch.no_grad():
            scores = self.model(
                img.to(self.device), torch.stack(images).to(self.device)
            )
            scores = scores.detach().cpu().numpy()

        max_index = np.argmin(scores)
        predict_label = self.labels[max_index]
        predict_label = self.dataset.classes[predict_label]

        result = tfs.ToPILImage()(self.images[max_index].detach().cpu())
        sim = scores[max_index]

        return dict(
            query=img[0].detach().cpu().numpy(),
            result=result,
            label=predict_label,
            sim=sim,
        )

    @staticmethod
    def get_mat(mask: Image, grid=(56, 40)) -> List[List[int]]:
        """
        get_mat Get the border of mask in the format of [left, right]

        Args:
            mask (Image): mask image to process
            grid (tuple, optional): the grid size for that mat (should use the same ratio with the image). Defaults to (56, 40).

        Returns:
            List[List[int]]: the matrix for the border of the mask 
        """
        mat = np.array(mask.convert("L").resize(grid))
        marked_mat = []
        for row in mat:
            if np.count_nonzero(row) == 0:
                marked_mat.append([grid[0] + 1, grid[0]])
                continue

            flag = False
            indices = []
            for index, value in enumerate(row):
                if value > 0 and not flag:
                    indices.append(index)
                    flag = True

                if value == 0 and flag is True:
                    indices.append(index - 1)
                    break

            marked_mat.append(indices)

        return marked_mat

    @staticmethod
    def extend_square(i: int, j: int, marked_mat: List[List[int]], max_result: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        extend_square Find the max square inside the mask

        Args:
            i (int): row index
            j (int): column index
            marked_mat (List[List[int]]): the matrix for border 
            max_result (Tuple[int, int, int]): the passing result (i, j, edge_size)

        Returns:
            Tuple[int]: (i, j, edge_size)
        """
        
        if i >= len(marked_mat):
            return max_result

        if j < marked_mat[i][0]:
            return PatternClassifier.extend_square(i, marked_mat[i][0], marked_mat, max_result)

        if j > marked_mat[i][1]:
            return PatternClassifier.extend_square(i + 1, 0, marked_mat, max_result)

        extend = max_result[2]
        while True:
            top = i
            bot = i + extend + 1
            left = j
            right = j + extend + 1

            if marked_mat[top][1] < right:
                break

            if bot >= len(marked_mat):
                break

            if marked_mat[bot][0] > left:
                break

            if marked_mat[bot][1] < right:
                break

            extend += 1

        if extend > max_result[2]:
            max_result = (i, j, extend)

        return PatternClassifier.extend_square(i, j + 1, marked_mat, max_result)

    @staticmethod
    def select_square(mask: Image, grid=(56, 40)) -> List[int]:
        """
        select_square Select the region inside the mask for geting the pattern contain square image

        Args:
            mask (Image): the mask image
            grid (tuple, optional): The grid for algorithm. Defaults to (56, 40).

        Returns:
            List[int, int, int, int]: The border for the select region [left, top, right, bot]
        """
        
        w, h = mask.size

        marked_mat = PatternClassifier.get_mat(mask, grid)
        square = PatternClassifier.extend_square(0, 0, marked_mat, (0, 0, 0))
        square = [
            square[1] + 1,
            square[0] + 1,
            square[1] + square[2],
            square[0] + square[2],
        ]

        square = (
            int(float(square[0] * w) / grid[0]),
            int(float(square[1] * h) / grid[1]),
            int(float(square[2] * w) / grid[0]),
            int(float(square[3] * h) / grid[1]),
        )

        return square
    
    async def __call__(self, image: Image, mask: Image) -> Dict:
        """
        __call__ Process the pattern classification and color detection.

        Args:
            image (Image): Image to process
            mask (Image): The mask for image

        Returns:
            Dict: dict(patter: Dict, color: Dict)
        """
        square = self.select_square(mask)
        image = image.crop(square)
        
        color = self.find_color(image)
        
        w, _ = image.size
        image = image.resize((w, w))
        
        pattern = self.find_pattern(image)
        
        return dict(
            pattern=pattern,
            color=color
        )
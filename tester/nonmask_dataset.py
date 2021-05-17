from torch.utils.data import Dataset

import os
from os import path


class NonMaskDataset(Dataset):
    def __init__(self, root_folder: str):
        folders = [path.join(root_folder, folder) for folder in os.listdir(root_folder)]
        self.labels = [
            int(image.split("_")[0])
            for folder in folders
            for image in os.listdir(folder)
            if not image.startswith("mask")
        ]
        self.images = [
            path.join(folder, image)
            for folder in folders
            for image in os.listdir(folder)
            if not image.startswith("mask")
        ]
        self.masks = [
            ""
            for image in self.images
        ]

    def __getitem__(self, index):
        return self.images[index], self.masks[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = NonMaskDataset("/data/part_detect_smart_factory/result_gen_fabric_shape/")
    for i in range(len(dataset)):
        image, mask, _ = dataset[i]
        if not path.exists(image):
            raise ValueError(f"{image} - {mask}")

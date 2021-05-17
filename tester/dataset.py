from torch.utils.data import Dataset

import os
from os import path


class TesterDataset(Dataset):
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
            path.join(folder, "_".join(["mask_"] + image.split("_")[1:]))
            for folder in folders
            for image in os.listdir(folder)
            if not image.startswith("mask")
        ]

    def __getitem__(self, index):
        return self.images[index], self.masks[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = TesterDataset("/data/part_detect_smart_factory/result_gen_fabric_shape/")
    for i in range(len(dataset)):
        image, mask, _ = dataset[i]
        if not path.exists(image) or not path.exists(mask):
            raise ValueError(f"{image} - {mask}")

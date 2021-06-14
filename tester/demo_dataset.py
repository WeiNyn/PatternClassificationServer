from torch.utils.data import Dataset


from torchvision import transforms
from torchvision.transforms import RandomCrop


from torchvision.datasets import ImageFolder


class TestDataset(Dataset):
    def __init__(self, folder, length):
        self.dataset = ImageFolder(folder)
        self.len = length
        
        self.tfs = RandomCrop(256)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        index = index%len(self.dataset)
        
        image, label = self.dataset[index]
        
        return self.tfs(image), label
    
    
if __name__ == '__main__':
    dataset = TestDataset('/data/container_testing/smart_factory_demo/IMGS_0706_crop', 100)
    
    for image in dataset
    
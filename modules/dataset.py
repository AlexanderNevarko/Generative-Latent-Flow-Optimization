from torch.utils.data import Dataset

class IdxDataset(Dataset):
    """
    Wrap a dataset to map indices to images
    In other words, instead of producing (X, y) it produces (idx, X, y).
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return (idx, img, target)
    

class ImgDataset(Dataset):
    """
    Wrap a dataset to load only images without labels.
    Used to calculate FD with classifier.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img
    
    
class LatentsDataset(Dataset):
    def __init__(self, latents, transform=None):
        self.lat = latents
        self.transform = transform
        
    def __len__(self):
        return len(self.lat)
    
    def __getitem__(self, idx):
        x = self.lat[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x
    
class LatentsContextualDataset(Dataset):
    def __init__(self, latents, contexts, transform=None):
        self.lat = latents
        self.cont = contexts
        self.transform = transform
        
    def __len__(self):
        return len(self.lat)
    
    def __getitem__(self, idx):
        x = self.lat[idx]
        y = self.cont[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
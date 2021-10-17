from torch.utils.data import Dataset

class IdxDataset(Dataset):
    """Wrap a dataset to map indices to images
    In other words, instead of producing (X, y) it produces (idx, X). The label
    y is not relevant for our task.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return (idx, img, target)
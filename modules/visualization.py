import torch

from torchvision.utils import make_grid, save_image
from torchvision import transforms



def visualize_image_grid(glo_model, inputs=None):
    if inputs is not None:
        inputs = inputs.to(glo_model.z.weight.device)
        img = glo_model(inputs=inputs)
    else:    
        idx_num = len(glo_model.z.weight)
        random_idx = torch.randint(low=0, high=idx_num, size=(16,), device=glo_model.z.weight.device)
        img = glo_model(idx=random_idx)
    img = img.detach().cpu()
    grid = make_grid(img, nrow=len(img) // 4, padding=1)
    transform = transforms.ToPILImage()
    return transform(grid)

def visualize_paired_results(glo_model, dataloader, img_num=8):
    '''
    glo_model: model
    dataloader: train dataloader
    img_num: number of images to draw
    '''
    
    idx = torch.randint(low=0, high=len(glo_model.z.weight), size=(img_num, ))
    preds = glo_model(idx=idx.to(glo_model.z.weight.device)).detach().cpu()
    img = []
    for i in idx:
        img.append(dataloader.dataset[i][1].detach().cpu())
    
    img_grid = make_grid(img, nrow=1, padding=1)
    pred_grid = make_grid(preds, nrow=1, padding=1)
    pairs = torch.empty(2, *img_grid.shape, dtype=torch.float32)
    pairs[0] = pred_grid
    pairs[1] = img_grid
    
    grid = make_grid(pairs, nrow=2)
    transform = transforms.ToPILImage()
    return transform(grid)    
    
def img_side_by_side(img1, img2, inv_transform=None, save_file=None):
    '''
    Visualize two pairs of image tensor sets
    img1, img2: tensors with shape B x C x H x W
    inv_transform: inverse transform for tensor -> image transformation
        default: torchvision.transforms.ToPILImage() 
    save_file: str, file name to save images. If None, don't save images.
        default: None
    '''
    if inv_transform is None:
        inv_transform = transforms.ToPILImage()
    img1, img2 = img1.detach().cpu(), img2.detach().cpu()
    grid1, grid2 = make_grid(img1, nrow=1, padding=1), make_grid(img2, nrow=1, padding=1)
    pairs = torch.empty(2, *grid1.shape, dtype=torch.float32)
    pairs[0], pairs[1] = grid1, grid2
    grid = make_grid(pairs, nrow=2, padding=1)
    if save_file is not None:
        save_image(grid, save_file)
    return inv_transform(grid)
    
def img_raw_by_raw(*imgs, inv_transform=None, save_file=None):
    '''
    Visualize two pairs of image tensor sets
    imgs: tensors array with shapes B x C x H x W
    inv_transform: inverse transform for tensor -> image transformation
        default: torchvision.transforms.ToPILImage() 
    save_file: str, file name to save images. If None, don't save images.
        default: None
    '''
    if inv_transform is None:
        inv_transform = transforms.ToPILImage()
    grids = []
    for img in imgs:
        img = img.detach().cpu()
        grid = (make_grid(img, nrow=len(img), padding=1))
        grids.append(grid)
    
    tuples = torch.empty(len(imgs), *grids[0].shape, dtype=torch.float32)
    for i in range(len(tuples)):
        tuples[i] = grids[i]
    grid = make_grid(tuples, nrow=1, padding=1)
    if save_file is not None:
        save_image(grid, save_file)
    return inv_transform(grid)

import torch

from torchvision.utils import make_grid
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
    grid = make_grid(img, nrow=len(img) // 4)
    transform = transforms.Compose([
        transforms.ToPILImage(),
    ])
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
    
    img_grid = make_grid(img, nrow=1)
    pred_grid = make_grid(preds, nrow=1)
    pairs = torch.empty(2, *img_grid.shape, dtype=torch.float32)
    pairs[0] = pred_grid
    pairs[1] = img_grid
    
    grid = make_grid(pairs, nrow=2)
    transform = transforms.ToPILImage()
    return transform(grid)    
    



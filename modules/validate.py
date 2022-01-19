import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import numpy as np


from tqdm.notebook import tqdm

from .glo_generator import SampleGenerator


class Validator():
    
    def __init__(self, model, val_loader):
        self.model = model
        self.val_loader = val_loader
        self.device = next(self.model.parameters()).device
        
    def validate(self, z, min_loss, loss_func, optimizer, max_iter=60):
        '''
        z: nn.Embedding 
        Never set model to eval mode, it requires gradients!!!
        '''
        running_loss = []
        for idx, img, _ in tqdm(self.val_loader, leave=True):
            # import ipdb; ipdb.set_trace()
            idx, img = idx.long().to(self.device), img.float().to(self.device)
            bs = len(idx)
            loss = torch.full(size=(bs, ), fill_value=min_loss+1.0)
            cnt = 0
            while torch.any(min_loss < loss) and cnt < max_iter:
                # import ipdb; ipdb.set_trace()
                optimizer.zero_grad()
                preds = self.model(inputs=z(idx))
                loss = loss_func(preds, img)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    z.weight[idx] = SampleGenerator.reproject_to_unit_ball(z(idx))
                cnt += 1
                    
            running_loss.append(loss.mean().item())
            
        return z, np.mean(running_loss)

    def visualize_val_results(self, z, img, save_file=None):
        '''
        z: latent vectors to generate on
        img: ground truth image tensors
        '''
        preds = self.model(inputs=z).detach().cpu()
        img = img.detach().cpu()
        
        img_grid = make_grid(img, nrow=1)
        pred_grid = make_grid(preds, nrow=1)
        pairs = torch.empty(2, *img_grid.shape, dtype=torch.float32)
        pairs[0] = pred_grid
        pairs[1] = img_grid
        
        grid = make_grid(pairs, nrow=2)
        if save_file is not None:
            save_image(grid, save_file)
        transform = transforms.ToPILImage()
        return transform(grid)
             
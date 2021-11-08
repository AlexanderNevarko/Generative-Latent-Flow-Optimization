import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision import transforms
# import pytorch_lightning as pl
import numpy as np

from tqdm import tqdm

from collections import Counter


class GLOTrainer():
    def __init__(self, 
                 model, use_gpu,
                 logger=None):
        self.model = model
        self.logger = logger
        self.use_gpu = use_gpu
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    def train(self, n_epochs,
              train_loader,
              loss_func,
              generator_optimizer,
              z_optimizer,
              exp_name,
              model_path='',
              generator_scheduler=None,
              z_scheduler=None):
        if self.logger is not None:
            self.logger.set_name(exp_name)
        self.model.train()
        
        cnt = Counter()
        for epoch in range(n_epochs):
            running_loss = []
            z_grad = torch.zeros_like(self.model.z.weight).to(self.device)
            for i, (idx, img, target) in enumerate(tqdm(train_loader, leave=False)):
                idx, img = idx.long().to(self.device), img.float().to(self.device)
                
                generator_optimizer.zero_grad()
                z_optimizer.zero_grad()
                preds = self.model(idx=idx)
                loss = loss_func(preds, img)
                loss.backward(retain_graph=True)
                generator_optimizer.step()
                z_optimizer.step()
                
                # Don't forget to reproject z
                with torch.no_grad():
                    self.model.z.weight[idx] = \
                        self.model.sample_generator.reproject_to_unit_ball(self.model.z.weight[idx])
                # Log metrics
                running_loss.append(loss.item())
                if self.logger is not None:
                    self.logger.log_metric(f'Train loss', loss.item(), epoch=epoch, step=cnt['train'])
                z_grad += self.model.z.weight.grad.to_dense()
                cnt['train'] += 1
            # Apply schedulers
            if generator_scheduler is not None:
                if isinstance(generator_scheduler, ReduceLROnPlateau):
                    generator_scheduler.step(torch.mean(torch.tensor(running_loss)))
                else:
                    generator_scheduler.step()
            if z_scheduler is not None:
                if isinstance(z_scheduler, ReduceLROnPlateau):
                    z_scheduler.step(torch.mean(torch.tensor(running_loss)))
                else:
                    z_scheduler.step()
            # Log metrics
            if self.logger is not None:
                self.logger.log_metric(f'Average epoch train loss', np.mean(running_loss), epoch=epoch, step=epoch)
                try:
                    self.logger.log_image(visualize_paired_results(self.model, train_loader, 16), 
                                          name=f'Epoch {epoch}', step=epoch)
                except Exception as e:
                    self.logger.log_image(visualize_image_grid(self.model), name=f'Epoch {epoch}', step=epoch)
                self.logger.log_metric(f'Average z-gradient', torch.mean(torch.abs(z_grad)), epoch=epoch, step=epoch)
            print(f'Average epoch {epoch} loss: {np.mean(running_loss)}')
            torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))
            
                
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
    



# class GLOOptimizer(torch.optim.Adam):
#     def __init__(self, *args, z_param, zlr):
#         super(GLOOptimizer, self).__init__(*args)
#         self.z_param = z_param
#         self.zlr = zlr
    
#     def step(self, closure=None, z_grad=None):
#         loss = super().step()
        
#         self.z_param -= self.zlr * z_grad
#         self.z_param /= torch.max(torch.tensor([torch.sqrt(torch.sum(self.z_param**2)), 1.0]))
#         return loss
        


# class GLO_pl(pl.LightningModule):
    
#     def __init__(self, model, loss, zlr, experiment, exp_name):
#         super(GLO_pl, self).__init__()
#         self.model = model
#         self.loss = loss
#         self.zlr = zlr
#         self.experiment = experiment
#         self.experiment.set_name(exp_name)
        
#     def training_step(self, batch, bacth_idx):
#         z, img = batch
#         pred = self.model(z)
#         loss = self.loss(pred, img)
#         self.experiment.log_metric()
#         return loss
    
#     def validation_step(self, batch, bacth_idx):
#         z, img = batch.batch
#         pred = self.model(z)
#         loss = self.loss(pred, img)
        
    
#     def configure_optimizers(self):
#         return GLOOptimizer(self.model.parameters(), 
#                             z_param=self.model.z, 
#                             zlr=self.zlr)
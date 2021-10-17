import os
import torch
import torch.nn as nn
# import pytorch_lightning as pl
import numpy as np

from tqdm import tqdm

from collections import Counter


class GLOTrainer():
    def __init__(self, 
                 model, use_gpu,
                 logger):
        self.model = model
        self.logger = logger
        self.use_gpu = use_gpu
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    def train(self, n_epochs,
              train_loader,
              loss_func,
              optimizer,
              exp_name,
              model_path):
        cnt = Counter()
        self.logger.set_name(exp_name)
        for epoch in range(n_epochs):
            running_loss = []
            for i, (idx, img, target) in enumerate(tqdm(train_loader, leave=False)):
                # import ipdb; ipdb.set_trace()
                idx, img = idx.long().to(self.device), img.float().to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(idx=idx)
                loss = loss_func(preds, img)
                loss.backward()
                optimizer.step()
                # Don't forget to reproject z
                with torch.no_grad():
                    self.model.z[idx] = \
                        self.model.sample_generator.reproject_to_unit_ball(self.model.z[idx])
                # Log metrics
                running_loss.append(loss.item())
                self.logger.log_metric(f'Train loss', loss.item(), epoch=epoch, step=cnt['train'])
                cnt['train'] += 1
            self.logger.log_metric(f'Average epoch train loss {np.mean(running_loss)}')
            print(f'Average epoch {epoch} loss: {np.mean(running_loss)}')
            torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))
            
                
        



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
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision import transforms
# import pytorch_lightning as pl
import numpy as np

from .utils import standard_normal_logprob

from tqdm.notebook import tqdm

from collections import Counter


class FlowTrainer():
    def __init__(self, model, use_gpu=False, logger=None):
        self.model = model
        self.use_gpu = use_gpu
        self.logger = logger
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    def train(self, n_epochs,
              train_loader, 
            #   criterion, 
              optimizer, 
              scheduler=None,
              exp_name='',
              model_path=''):
        if self.logger is not None:
            self.logger.set_name(exp_name)
        self.model.train()
        cnt = Counter()
        best_epoch_loss = np.inf
        for epoch in range(n_epochs):
            running_loss = []
            for latents, attributes in tqdm(train_loader, leave=True):
                latents = latents.long().to(self.device)
                bs, lat_dim = latents.shape
                
                approx21, delta_log_p2 = self.model(latents)
                approx2 = standard_normal_logprob(approx21).view(bs, -1).sum(1, keepdim=True)
                delta_log_p2 = delta_log_p2.view(bs, lat_dim, 1).sum(1)
                log_p2 = (approx2 - delta_log_p2)

                loss = -log_p2.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                running_loss.append(loss.item())
                if self.logger is not None:
                    self.logger.log_metric(f'Train loss', loss.item(), epoch=epoch, step=cnt['train'])
            if scheduler is not None:
                scheduler.step()
            epoch_loss = np.mean(running_loss)
            print(f'Average epoch {epoch} train loss: {epoch_loss}')
            if self.logger is not None:
                self.logger.log_metric(f'Average epoch train loss', epoch_loss, epoch=epoch, step=epoch)
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))
                
            
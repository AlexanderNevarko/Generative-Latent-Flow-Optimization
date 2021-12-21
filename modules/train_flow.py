import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision import transforms
# import pytorch_lightning as pl
import numpy as np

from .utils import standard_normal_logprob
from .visualization import visualize_image_grid


from tqdm.notebook import tqdm

from collections import Counter


class FlowTrainer():
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        self.device = next(iter(self.model.parameters())).device
    
    def train(self, n_epochs,
              train_loader, 
            #   criterion, 
              optimizer, 
              scheduler=None,
              generator_model=None,
              exp_name='',
              model_path=''):
        if self.logger is not None:
            self.logger.set_name(exp_name)
        self.model.train()
        cnt = Counter()
        best_epoch_loss = np.inf
        for epoch in range(n_epochs):
            running_loss = []
            pbar = tqdm(train_loader, leave=True)
            for latents in pbar:
                latents = latents.float().to(self.device)
                bs, lat_dim = latents.shape
                optimizer.zero_grad()
                loss = -self.model.log_prob(inputs=latents).mean()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                if self.logger is not None:
                    self.logger.log_metric(f'Train loss', loss.item(), epoch=epoch, step=cnt['train'])
                cnt['train'] += 1
            if scheduler is not None:
                scheduler.step()
            epoch_loss = np.mean(running_loss)
            print(f'Average epoch {epoch} train loss: {epoch_loss}')
            if self.logger is not None:
                self.logger.log_metric(f'Average epoch train loss', epoch_loss, epoch=epoch, step=epoch)
                if generator_model is not None:
                    self.logger.log_image(visualize_image_grid(generator_model, inputs=self.model.sample(16)), 
                                          name=f'Epoch {epoch}', step=epoch)
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))
                
            
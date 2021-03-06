import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision import transforms
# import pytorch_lightning as pl
import numpy as np

from .utils import standard_normal_logprob
from .visualization import visualize_image_grid, img_side_by_side
from .loss import ValLoss


from tqdm.notebook import tqdm

from collections import Counter


class FlowTrainer():
    def __init__(self, model, logger=None, generator_model=None, basic_model=None):
        self.model = model
        self.logger = logger
        self.generator = generator_model
        self.basic = basic_model
        self.device = next(iter(self.model.parameters())).device
    
    def train(self, n_epochs,
              train_loader, 
              val_latents,
              optimizer, 
              scheduler=None,
              exp_name='',
              model_path=''):
        if self.logger is not None:
            self.logger.set_name(exp_name)
        cnt = Counter()
        best_epoch_loss = np.inf
        val_latents = val_latents.detach().cpu().numpy()
        for epoch in range(n_epochs):
            self.model.train()
            running_loss = []
            pbar = tqdm(train_loader, leave=True)
            for latents in pbar:
                latents = latents.float().to(self.device)
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
                if self.generator is not None:
                    if self.basic is None:
                        self.logger.log_image(visualize_image_grid(self.generator, 
                                                                   inputs=self.model.sample(16)), 
                                              name=f'Epoch {epoch}', step=epoch)
                    else:
                        flow_img = self.generator(inputs=self.model.sample(16).to(self.device))
                        basic_img = self.generator(inputs=torch.tensor(self.basic.sample(16)[0], 
                                                                       dtype=torch.float).to(self.device))
                        self.logger.log_image(img_side_by_side(flow_img, basic_img), 
                                              name=f'Epoch {epoch} flow vs gaussian inference',
                                              step=epoch)
                if epoch % 5 == 0:
                    print(f'Calculating FID on epoch {epoch}')
                    size = len(val_latents)
                    fake_lats = self.model.sample(size).detach().cpu().numpy()
                    fid = ValLoss.calc_fid(val_latents, fake_lats)
                    self.logger.log_metric(f'FID on latents', fid, epoch=epoch, step=epoch)
                        
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))
                
            
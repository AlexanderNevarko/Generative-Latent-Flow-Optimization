import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from .visualization import visualize_image_grid, visualize_paired_results
from .loss import ValLoss

from tqdm.notebook import tqdm

from collections import Counter


class GLOTrainer():
    def __init__(self, 
                 model,
                 logger=None):
        self.model = model
        self.logger = logger
        self.device = next(iter(self.model.parameters())).device
    
    def train(self, n_epochs,
              train_loader,
              loss_func,
              generator_optimizer,
              z_optimizer,
              exp_name,
              fid_loader=None,
              fid_loss=None,
              model_path='',
              generator_scheduler=None,
              z_scheduler=None):
        if self.logger is not None:
            self.logger.set_name(exp_name)
        self.model.train()
        
        cnt = Counter()
        for epoch in range(n_epochs):
            running_loss = []
            self.model.train()
            for i, (idx, img, _) in enumerate(tqdm(train_loader, leave=False)):
                idx, img = idx.long().to(self.device), img.float().to(self.device)
                
                generator_optimizer.zero_grad()
                z_optimizer.zero_grad()
                preds = self.model(idx=idx)
                loss = loss_func(preds, img)
                loss.backward(retain_graph=True)
                generator_optimizer.step()
                z_optimizer.step()
                
                # Log metrics
                running_loss.append(loss.item())
                if self.logger is not None:
                    self.logger.log_metric(f'Train loss', loss.item(), epoch=epoch, step=cnt['train'])
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
            self.model.eval()
            if self.logger is not None:
                self.logger.log_metric(f'Average epoch train loss', np.mean(running_loss), epoch=epoch, step=epoch)
                try:
                    self.logger.log_image(visualize_paired_results(self.model, train_loader, 16), 
                                          name=f'Epoch {epoch}', step=epoch)
                except Exception as e:
                    self.logger.log_image(visualize_image_grid(self.model), name=f'Epoch {epoch}', step=epoch)
                
                if fid_loss is not None:
                    if epoch % 5 == 0:
                        print(f'Calculating FID and IS on epoch {epoch}')
                        fid, inception_score = fid_loss(self.model, fid_loader)
                            
                        self.logger.log_metric(f'FID on train', fid, epoch=epoch, step=epoch)
                        self.logger.log_metric(f'IS on train', inception_score, epoch=epoch, step=epoch)
                    
                
                
            print(f'Average epoch {epoch} loss: {np.mean(running_loss)}')
            if (i+1) % 2 == 0:
                self.model.tree.rebuild().to(self.device)
            torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))

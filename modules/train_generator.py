import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
from .visualization import visualize_image_grid, visualize_paired_results
from .loss import ValLoss

from tqdm.notebook import tqdm

from collections import Counter


class GLOTrainer():
    def __init__(self, 
                 model, use_gpu,
                 logger=None):
        self.model = model
        self.logger = logger
        self.use_gpu = use_gpu
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.val_loss = ValLoss().to(self.device)
    
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
            self.model.train()
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
            self.model.eval()
            if self.logger is not None:
                self.logger.log_metric(f'Average epoch train loss', np.mean(running_loss), epoch=epoch, step=epoch)
                try:
                    self.logger.log_image(visualize_paired_results(self.model, train_loader, 16), 
                                          name=f'Epoch {epoch}', step=epoch)
                except Exception as e:
                    self.logger.log_image(visualize_image_grid(self.model), name=f'Epoch {epoch}', step=epoch)
                self.logger.log_metric(f'Average z-gradient', torch.mean(torch.abs(z_grad)), epoch=epoch, step=epoch)
                
                if epoch % 3 == 0:
                    print('Calculate FID, IS')
                    real_ft, fake_ft, fake_pr = [], [], []
                    for idx, img, _ in train_loader:
                        idx, img = idx.to(self.device), img.to(self.device)
                        gen_img = self.model(idx=idx)
                        real_features, fake_features, fake_probs = self.val_loss.calc_data([img], [gen_img])
                        real_ft.append(real_features)
                        fake_ft.append(fake_features)
                        fake_pr.append(fake_probs)
                        
                        
                    real_ft = np.concatenate(real_ft)
                    fake_ft = np.concatenate(fake_ft)
                    fake_pr = np.concatenate(fake_pr)
                    fid = self.val_loss.calc_fid(real_ft, fake_ft)
                    inception_score = self.val_loss.calc_is(fake_pr)
                    self.logger.log_metric(f'FID on train', fid, epoch=epoch, step=epoch)
                    self.logger.log_metric(f'IS on train', inception_score, epoch=epoch, step=epoch)
                
                
                
            print(f'Average epoch {epoch} loss: {np.mean(running_loss)}')
            torch.save(self.model.state_dict(), os.path.join(model_path, f'{exp_name}_model.pth'))

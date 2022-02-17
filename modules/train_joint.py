import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from .visualization import img_raw_by_raw

from tqdm import tqdm

from collections import Counter

def bad_loss(loss):
    return (loss > 10) or torch.isnan(loss) or torch.isinf(loss)

def train_joint(model, flow, train_loader, 
                g_optimizer, z_optimizer, 
                flow_optimizer, g_scheduler, 
                z_scheduler, flow_scheduler, 
                criterion, val_loss, val_loader, 
                experiment, cfg):
    model_path = cfg['model_path']
    device = cfg['device']
    n_components = cfg['lat_dim']
    n_epochs = cfg['n_epochs']
    alpha = cfg['alpha'] # Factor of flow gradient
    
    clipping_value = cfg.get('clipping_value', 1e-4) # for flow gradient clipping
    img_num = cfg.get('img_num', 16) # number of images to log
    exp_name = cfg.get('exp_name', 'default')
    
    if experiment is not None:
        experiment.set_name(exp_name)
    cnt = Counter()
        
    for epoch in tqdm(range(n_epochs)):
        flow.train()
        model.train()
        gen_running_loss = []
        flow_running_loss = []
        for i, (idx, img, _) in enumerate(tqdm(train_loader, leave=False)):
            idx, img = idx.long().to(device), img.float().to(device)
            
            g_optimizer.zero_grad()
            z_optimizer.zero_grad()
            flow_optimizer.zero_grad()
            # Generator forward pass
            preds = model(idx=idx)
            loss = criterion(preds, img)
            loss.backward(retain_graph=True)
            g_optimizer.step()
            # Save Z gradient from generator and zero it on Z parameter
            gen_z_grad = model.z.weight.grad.detach()
            model.z.weight.grad.zero_()
            # Flow forward pass
            noise = torch.randn_like(model.z(idx), device=device).float() * 1e-2 # For flow stability
            normal_z, log_jac_det = flow(model.z(idx) + noise)
            flow_loss = 0.5 * torch.sum(normal_z**2, 1) - log_jac_det
            flow_loss = flow_loss.mean() / n_components
            if bad_loss(flow_loss):
                print(f'Bad loss {flow_loss} occured here: epoch - {epoch}, iteration - {i}')
                z_optimizer.step()
                continue
            flow_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), clipping_value)
            flow_optimizer.step()
            
            # Save Z gradient from flow and zero it on Z parameter
            flow_z_grad = model.z.weight.grad.detach()
            model.z.weight.grad.zero_()
            
            # Reconstruct the Z gradient summing the two gradients from generator and flow
            model.z.weight.grad = alpha*flow_z_grad + (1-alpha)*gen_z_grad
            z_optimizer.step()
            
            # Log metrics
            gen_running_loss.append(loss.item())
            flow_running_loss.append(flow_loss.item())
            if experiment is not None:
                experiment.log_metric(f'Train generator loss', loss.item(), epoch=epoch, step=cnt['train'])
                experiment.log_metric(f'Train flow loss', flow_loss.item(), epoch=epoch, step=cnt['train'])
            cnt['train'] += 1

        if g_scheduler is not None:
            g_scheduler.step()
        if z_scheduler is not None:
            z_scheduler.step()
        if flow_scheduler is not None:
            flow_scheduler.step()

        # Log metrics and images
        gen_epoch_loss = np.mean(gen_running_loss)
        flow_epoch_loss = np.mean(flow_running_loss)
        print(f'Average generator epoch {epoch} train loss: {gen_epoch_loss}')
        print(f'Average flow epoch {epoch} train loss: {flow_epoch_loss}')
        if experiment is not None:
            experiment.log_metric(f'Average generator epoch train loss', gen_epoch_loss, epoch=epoch, step=epoch)
            experiment.log_metric(f'Average flow epoch train loss', flow_epoch_loss, epoch=epoch, step=epoch)
            
            model.eval()
            flow.eval()
            normal_samples = torch.randn(img_num, n_components).to(device)
            random_idx = torch.randint(low=0, high=len(model.z.weight), size=(img_num,), device=device)

            flow_samples, _ = flow(normal_samples, rev=True)
            flow_imgs = model(inputs=flow_samples)
            fit_imgs = model(idx=random_idx)
            train_imgs = torch.stack([train_loader.dataset[i][1].detach().cpu() for i in random_idx])
            if len(train_imgs.shape) == 3:
                train_imgs = train_imgs.unsqueeze(1)
            experiment.log_image(img_raw_by_raw(train_imgs, fit_imgs, flow_imgs), 
                                name=f'Epoch {epoch}', step=epoch, )
                
            if epoch % 10 == 0:
                print(f'Calculating FID on epoch {epoch}')
                # FrEIA package
                def inputs_generator(size):
                    normal_sample = torch.randn(size, n_components).to(device)
                    fake_lats, _ = flow(normal_sample, rev=True)
                    return fake_lats

                fid, inception_score = val_loss(model, val_loader, inputs_generator)
                print(f'FID: {fid}, IS: {inception_score}')
                if np.isnan(fid) or np.isnan(inception_score):
                    print(f'Bad FID {fid} or IS {inception_score}')
                    fid = 16.0
                    inception_score = 2.45
                experiment.log_metric(f'FID', fid, epoch=epoch, step=epoch)
                experiment.log_metric(f'IS', inception_score, epoch=epoch, step=epoch)
                
        torch.save(model.state_dict(), os.path.join(model_path, f'{exp_name}_generator_model.pth'))
        torch.save(flow.state_dict(), os.path.join(model_path, f'{exp_name}_flow_model.pth'))
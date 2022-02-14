import yaml
import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam, SparseAdam, SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from modules.glo_generator import SampleGenerator, GLOGenerator, GLOModel
from modules.loss import LapLoss, Lap35Loss, ValLoss
from modules.dataset import IdxDataset
from modules.train_joint import train_joint



import warnings
warnings.filterwarnings("ignore")

def get_dataset(cfg_data):
    if cfg_data['type'] == 'mnist':
        dataset = IdxDataset(MNIST(root=cfg_data['root'], train=True, 
                                   download=True, transform=transforms.ToTensor()))
    elif cfg_data['type'] == 'cifar10':
        dataset = IdxDataset(CIFAR10(root=cfg_data['root'], train=True, 
                                     download=True, transform=transforms.ToTensor()))
    loader = DataLoader(dataset, batch_size=cfg_data['batch_size'], 
                        shuffle=True, num_workers=cfg_data['num_workers'], pin_memory=True)
    sampler_loader = DataLoader(dataset, batch_size=cfg_data['batch_size'], 
                                shuffle=False, num_workers=cfg_data['num_workers'], pin_memory=True)
    return loader, sampler_loader
    
    
def get_optimizer(model, cfg_opt):
    if cfg_opt['type'] == 'adam':
        return Adam(model.parameters(), lr=cfg_opt['lr'])
    if cfg_opt['type'] == 'sparse_adam':
        return SparseAdam(model.parameters(), lr=cfg_opt['lr'])
    if cfg_opt['type'] == 'sgd':
        return SGD(model.parameters(), lr=cfg_opt['lr'])
    else:
        print(f'Unknown optimizer in config: {cfg_opt["type"]}')

def get_scheduler(opt, cfg_opt):
    if 'lr_policy' not in cfg_opt:
        return None
    if cfg_opt['lr_policy']['type'] == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg_opt['lr_policy']['step_size'],
            gamma=cfg_opt['lr_policy']['gamma'])
    elif cfg_opt['lr_policy']['type'] == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            opt,
            step_size=cfg_opt['lr_policy']['steps'],
            gamma=cfg_opt['lr_policy']['gamma'])
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(cfg_opt['lr_policy']['type']))
    return scheduler

def get_loss(cfg_loss, device):
    if cfg_loss['type'] == 'LapLoss':
        return LapLoss(**cfg_loss['params'], device=device)
    if cfg_loss['type'] == 'Lap35Loss':
        return Lap35Loss(**cfg_loss['params'], device=device)
    else:
        print(f'Unknown loss type in config: {cfg_loss["type"]}')


def get_flow(n_components, cfg_flow):
    def subnet_fc(dims_in, dims_out):
        middle_dim = cfg_flow['middle_dim']
        fc1 = nn.Linear(dims_in, middle_dim)
        torch.nn.init.xavier_normal_(fc1.weight)
        fc2 = nn.Linear(middle_dim, dims_out)
        torch.nn.init.zeros_(fc2.weight)
        return nn.Sequential(fc1, nn.ReLU(), fc2)
    flow = Ff.SequenceINN(n_components)
    N = cfg_flow['n_blocks']
    for k in range(N):
        if cfg_flow['base_block'] == 'AllInOneBlock':
            flow.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        else:
            print(f'Unknown base block in config: {cfg_flow["base_block"]}')
        if k < N-1:
            if cfg_flow['transitional_block'] == 'PermuteRandom':
                flow.append(Fm.PermuteRandom)
            elif cfg_flow['transitional_block'] == 'OrthogonalTransform':
                flow.append(Fm.OrthogonalTransform)
            else:
                print(f'Unknown transitional block in config: {cfg_flow["transitional_block"]}')
    return flow


def main():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('-c', '--cfg', help='Config file path', type=str, default='', required=True)
    args = parser.parse_args()
    cfg_file = args.cfg
    with open(cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
    device = torch.device(cfg['device'])
    n_components = cfg['lat_dim']
    bw_method = cfg['bw_method']
    kwargs = {'cfg': cfg}
    kwargs['train_loader'], sampler_loader = get_dataset(cfg['data'])
    sampler = SampleGenerator(sampler_loader, z_dim=n_components, bw_method=bw_method)
    generator = GLOGenerator(dataloader=kwargs['train_loader'], latent_channels=n_components, **cfg['generator']).to(device)
    kwargs['model'] = GLOModel(generator, sampler, sparse=True).to(device)
    kwargs['flow'] = get_flow(n_components, cfg['flow']).to(device)
    
    kwargs['g_optimizer'] = get_optimizer(kwargs['model'].generator, cfg['g_optimizer'])
    kwargs['z_optimizer'] = get_optimizer(kwargs['model'].z, cfg['z_optimizer'])
    kwargs['flow_optimizer'] = get_optimizer(kwargs['flow'], cfg['flow_optimizer'])
    kwargs['g_scheduler'] = get_scheduler(kwargs['g_optimizer'], cfg['g_optimizer'])
    kwargs['z_scheduler'] = get_scheduler(kwargs['z_optimizer'], cfg['z_optimizer'])
    kwargs['flow_scheduler'] = get_scheduler(kwargs['flow_optimizer'], cfg['flow_optimizer'])
    kwargs['criterion'] = get_loss(cfg['loss'], device)
    kwargs['val_loss'] = ValLoss(device)
    
    train_joint(**kwargs)
    
    

if __name__ == '__main__':
    main()
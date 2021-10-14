import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from res_block import PreActResBlock


def generate_samples(dataloader, z_dim=128, bw_method=0.1):
    _, img = zip(*[(idx, img_) for idx, img_, _ in dataloader])
    img = torch.cat(img)
    img = img.view(img.shape[0], -1).numpy()
    pca = PCA(n_components=z_dim)
    z_dataset = pca.fit_transform(img).T
    kernel = gaussian_kde(z_dataset, bw_method=bw_method)
    samples = kernel.resample(size=len(dataloader.dataset)).T
    samples = torch.tensor(samples, requires_grad=True)
    return samples


class GLOGenerator(nn.Module):
    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_blocks: int,
                 dataloader):
        '''
        dataloader: 
            must be indesed dataloader and return (idx, img, target)
        '''
        super(GLOGenerator, self).__init__()
        _, sample, _ = next(iter(dataloader))
        self.output_size = [sample.shape[-2], sample.shape[-1]]
        self.out_channels = sample[-3]
        if max_channels != min_channels * 2**num_blocks:
            raise ValueError(f'Wrong channels num: {max_channels}, {min_channels}')
        
        self.embed_features = noise_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.min_channels = min_channels
        self.max_channels = max_channels
        
        self.act = nn.ReLU(inplace=False)
        
        self.linear = spectral_norm(nn.Linear(noise_channels, 4*4*max_channels))
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(PreActResBlock(max_channels//2**i, max_channels//2**(i+1), 
                                                  noise_channels, upsample=True))
        self.bn = nn.BatchNorm2d(min_channels)
        self.end_conv = spectral_norm(nn.Conv2d(min_channels, self.out_channels, kernel_size=3, padding=1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, noise):
        bs = noise.shape[0]
        out = self.linear(noise)
        out = out.reshape((bs, self.max_channels, 4, 4))
        
        for i in range(self.num_blocks):
            # Scince we have pre-act blocks, we don't need activations inbetween
            out = self.res_blocks[i].forward(out, noise)
        out = self.bn.forward(out)
        out = self.act.forward(out)
        out = self.end_conv(out)
        out = self.sigmoid.forward(out)
        out = F.interpolate(out, size=(noise.shape[0], self.out_channels, *self.output_size))

        assert out.shape == (noise.shape[0], self.out_channels, *self.output_size)
        return out


class GLOModel(nn.Module):
    def __init__(self, generator, dataloader, z_dim, bw_method):
        self.generator = generator
        self.z = torch.Parameter(generate_samples(dataloader,
                                                  z_dim=z_dim,
                                                  bw_method=bw_method))
    
    def forward(self, idx=None, inputs=None):
        if inputs:
            return self.generator(inputs)
        return self.generator(self.z[idx])
        
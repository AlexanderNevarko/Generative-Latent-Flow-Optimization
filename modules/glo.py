import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from .res_block import PreActResBlock, AdaptiveBatchNorm


class SampleGenerator():
    def __init__(self, dataloader, z_dim, bw_method):
        _, img = zip(*[(idx, img_) for idx, img_, _ in dataloader])
        img = torch.cat(img)
        img = img.view(img.shape[0], -1).numpy()
        self.pca = PCA(n_components=z_dim)
        self.z_dataset = self.pca.fit_transform(img)
        self.kernel = gaussian_kde(self.z_dataset.T, bw_method=bw_method)
        
    def generate_samples(self, n_samples=None):
        n_samples = n_samples if n_samples else self.z_dataset.shape[0]
        samples = self.kernel.resample(size=n_samples).T
        samples = torch.tensor(samples, requires_grad=True)
        samples = self.reproject_to_unit_ball(samples)
        return samples
    
    def get_z_dataset(self):
        return SampleGenerator.reproject_to_unit_ball(torch.tensor(self.z_dataset, requires_grad=True))
    
    @staticmethod
    def reproject_to_unit_ball(z):
        # Inplace reprojection
        l2norm = torch.sqrt(torch.sum(z**2, axis=1))
        ones = torch.ones_like(l2norm)
        z = z / (torch.amax(torch.vstack([l2norm, ones]), dim=0)).view(z.shape[0], 1)
        return z.float()


class GLOGenerator(nn.Module):
    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_blocks: int,
                 dataloader,
                 normalization: str = '',
                 lrelu_slope: float = 0.2):
        '''
        dataloader: 
            must be indesed dataloader and return (idx, img, target)
        '''
        super(GLOGenerator, self).__init__()
        _, sample, _ = next(iter(dataloader))
        self.output_size = [sample.shape[-2], sample.shape[-1]]
        self.out_channels = sample.shape[-3]
        if max_channels != min_channels * 2**num_blocks:
            raise ValueError(f'Wrong channels num: {max_channels}, {min_channels}')
        
        self.embed_features = noise_channels
        self.num_blocks = num_blocks
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.normalization = normalization
        
        self.act = nn.ReLU(inplace=False)
        
        self.linear = spectral_norm(nn.Linear(noise_channels, 4*4*max_channels))
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(PreActResBlock(max_channels//2**i, max_channels//2**(i+1), 
                                                  noise_channels, upsample=True, 
                                                  lrelu_slope=lrelu_slope, norm=normalization))
        if normalization == 'in':
            self.last_norm = nn.InstanceNorm2d(min_channels)
        elif normalization == 'ada':
            self.last_norm = AdaptiveBatchNorm(min_channels, noise_channels)
        elif normalization == 'gn':
            self.last_norm = nn.GroupNorm(num_groups=8, num_channels=min_channels)
        else:
            self.last_norm = nn.BatchNorm2d(min_channels)
        self.end_conv = spectral_norm(nn.Conv2d(min_channels, self.out_channels, kernel_size=3, padding=1))
        # self.fine_tune_block = nn.Sequential(nn.ReLU(), 
        #                                      nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        self.sigmoid = nn.Tanh()
        
    def forward(self, noise):
        bs = noise.shape[0]
        out = self.linear(noise)
        out = out.reshape((bs, self.max_channels, 4, 4))
        
        for i in range(self.num_blocks):
            # Scince we have pre-act blocks, we don't need activations inbetween
            out = self.res_blocks[i](out, noise)
        if self.normalization == 'ada':
            out = self.last_norm(out, noise)
        else:
            out = self.last_norm(out)
        out = self.act(out)
        out = self.end_conv(out)
        out = F.interpolate(out, size=self.output_size, mode='bilinear')
        # out - self.fine_tune_block(out)
        
        out = self.sigmoid(out)

        assert out.shape == (noise.shape[0], self.out_channels, *self.output_size)
        return out

    # def optimize_to_img(self, img, loss_func, min_loss, optimizer, init_z=None):
    #     '''
    #     img: target images tensor with shape (B x C x H x W), must be on model device!
    #     loss_func: loss function for images
    #     min_loss: minimal value of loss to stop iterations
    #     optimizer: optimizer for Z latent vectors
    #     init_z: initial z values
    #     '''
    #     bs, channels, height, width = img.shape
    #     if init_z is None:
    #         init_z = torch.randn(size=(bs, self.embed_features), device=img.device)
    #     import ipdb; ipdb.set_trace()
    #     z = SampleGenerator.reproject_to_unit_ball(init_z)
    #     z.requires_grad_(True)
    #     loss = torch.full(size=(bs, ), fill_value=min_loss+1.0)
    #     while torch.any(min_loss < loss):
    #         optimizer.zero_grad()
    #         preds = self(z)
    #         loss = loss_func(preds, img)
    #         loss.backward()
    #         optimizer.step()
    #         with torch.no_grad():
    #             z = SampleGenerator.reproject_to_unit_ball(z)
    #             z.requires_grad_(True)
    #     return z.detach(), loss
        

class GLOModel(nn.Module):
    def __init__(self, generator, dataloader, sample_generator, sparse):
        super(GLOModel, self).__init__()
        self.generator = generator
        self.sample_generator = sample_generator
        self.z = nn.Embedding.from_pretrained(self.sample_generator.get_z_dataset(),
                                              sparse=sparse, freeze=False)
        self.z.requires_grad = True
    
    def forward(self, idx=None, inputs=None):
        if inputs is not None:
            return self.generator(inputs)
        return self.generator(self.z(torch.tensor(idx)))
    
    # def optimize_to_img(self, img, loss_func, min_loss, optimizer, init_z=None):
    #     return self.generator.optimize_to_img(img, loss_func, min_loss, optimizer, init_z)
        
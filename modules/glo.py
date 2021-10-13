import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from res_block import PreActResBlock


class GLO(nn.Module):
    
    def __init__(self, 
                min_channels: int, 
                max_channels: int,
                noise_channels: int,
                num_classes: int,
                num_blocks: int,
                use_class_condition: bool):
        super(GLO, self).__init__()
        self.output_size = 4 * 2**num_blocks
        if max_channels != min_channels * 2**num_blocks:
            raise ValueError(f'Wrong channels num: {max_channels}, {min_channels}')
        
        self.num_classes = num_classes
        self.embed_features = noise_channels
        self.num_blocks = num_blocks
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.use_class_condition = use_class_condition
        
        self.act = nn.ReLU(inplace=False)
        
        if self.use_class_condition:
            self.embed = nn.Embedding(num_classes, noise_channels)
            self.linear = spectral_norm(nn.Linear(noise_channels*2, 4*4*max_channels))
        else:
            self.linear = spectral_norm(nn.Linear(noise_channels, 4*4*max_channels))
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if use_class_condition:
                self.res_blocks.append(PreActResBlock(max_channels//2**i, max_channels//2**(i+1), noise_channels*2,
                                                      batchnorm=use_class_condition, upsample=True))
            else:
                self.res_blocks.append(PreActResBlock(max_channels//2**i, max_channels//2**(i+1), noise_channels,
                                                      batchnorm=use_class_condition, upsample=True))
        self.bn = nn.BatchNorm2d(min_channels)
        self.end_conv = spectral_norm(nn.Conv2d(min_channels, 3, kernel_size=3, padding=1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, noise, labels):
        bs = noise.shape[0]
        embeds = noise
        if self.use_class_condition:
            embeds = torch.cat([embeds, self.embed(labels)], dim=1)
        out = self.linear(embeds)
        out = out.reshape((bs, self.max_channels, 4, 4))
        
        for i in range(self.num_blocks):
            # Scince we have pre-act blocks, we don't need activations inbetween
            out = self.res_blocks[i].forward(out, embeds)
        out = self.bn.forward(out)
        out = self.act.forward(out)
        out = self.end_conv(out)
        outputs = self.sigmoid.forward(out)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs

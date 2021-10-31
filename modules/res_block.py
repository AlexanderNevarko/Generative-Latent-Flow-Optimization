import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm


class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer 
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "noise" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        self.num_features = num_features
        self.embed_features = embed_features
        # self.bn = nn.BatchNorm2d(num_features, affine=False) # super().forward(inputs)
        self.f_gamma = nn.Linear(embed_features, num_features)
        self.f_bias = nn.Linear(embed_features, num_features)

    def forward(self, inputs, noise):
        """
        noise: [B x embed_features]
        inputs: [B x C x H x W]
        """
        gamma = self.f_gamma(noise)
        bias = self.f_bias(noise)

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs)

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 noise_channels: int = None,
                 ada_in: bool = False,
                 batchnorm: bool = False,
                 instancenorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.noise_channels = noise_channels
        self.batchnorm = batchnorm
        self.ada_in = ada_in
        self.instancenorm = instancenorm
        self.upsample = upsample
        self.downsample = downsample
        self.act = nn.ReLU(inplace=False)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
        if upsample:
            self.ups = nn.Upsample(scale_factor=2, mode='bilinear')
        
        if ada_in:
            self.bn1 = AdaptiveBatchNorm(in_channels, noise_channels)
            self.bn2 = AdaptiveBatchNorm(out_channels, noise_channels)
        elif instancenorm:
            self.in1 = nn.InstanceNorm2d(in_channels)
            self.in2 = nn.InstanceNorm2d(out_channels)
        elif batchnorm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
        if in_channels != out_channels:
            self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.skip = None
          
          

    def forward(self, 
                inputs, # regular features 
                noise=None): # noise used in adaptive batch norm
        if self.upsample:
            inputs = self.ups(inputs)
        
        identity = inputs
        out = inputs
        if self.ada_in:
            out = self.bn1(out, noise)
        elif self.batchnorm:
            out = self.bn1(out)
        elif self.instancenorm:
            out = self.in1(out)
        
        out = self.act.forward(out)
        out = self.conv1(out)
        
        if self.ada_in:
            out = self.bn2(out, noise)
        elif self.batchnorm:
            out = self.bn2(out)
        elif self.instancenorm:
            out = self.in2(out)
        out = self.act.forward(out)
        out = self.conv2(out)

        if self.skip:
            identity = self.skip(identity)
        outputs = identity + out
        
        if self.downsample:
            outputs = F.avg_pool2d(outputs, kernel_size=2)

        return outputs


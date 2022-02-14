import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
from scipy import linalg

from torch.utils.data import DataLoader
from tqdm import tqdm


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    if size == 3:
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]]) / 16
    elif size == 5:
        kernel = torch.tensor([[1., 4., 7., 4., 1],
                               [4., 16., 26., 16., 4.],
                               [7., 26., 41., 26., 7.],
                               [4., 16., 26., 16., 4.],
                               [1., 4., 7., 4., 1.]]) / 273
    # kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    size = kernel.shape[-1]
    if size == 3:
        img = torch.nn.functional.pad(img, (1, 1, 1, 1), mode='reflect')
    elif size == 5:
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(nn.Module):
    def __init__(self, max_levels=3, channels=3, size=5, device=torch.device('cpu')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(size=size, channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class Lap35Loss(nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu'), alpha=0.5):
        super(Lap35Loss, self).__init__()
        self.lap3 = LapLoss(max_levels, channels, 3, device)
        self.lap5 = LapLoss(max_levels, channels, 5, device)
        self.alpha = alpha
    
    def forward(self, input, target):
        lap3 = self.lap3(input, target)
        lap5 = self.lap5(input, target)
        return self.alpha*lap3 + (1-self.alpha)*lap5



class ValLoss(nn.Module):
    """
    Calculates FID and IS for generator model
    """
    def __init__(self, device):
        super(ValLoss, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)
        self.inception_v3.eval()
        self.device = device
        for p in self.inception_v3.parameters():
            p.requires_grad = False
        self.inception_v3.to(device)

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess data
        x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = (x - 0.5) * 2
        if x.shape[1] == 1:
            x = torch.stack([x, x, x], dim=1).squeeze(2)


        # N x 3 x 299 x 299
        x = self.inception_v3.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_v3.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_v3.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_v3.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_v3.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_v3.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_v3.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_v3.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_v3.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_v3.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_v3.Mixed_7c(x)
        # Adaptive average pooling
        x = self.inception_v3.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception_v3.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        return x


    @torch.no_grad()
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        # N x 2048
        x = self.inception_v3.fc(x)
        # N x 1000 (num_classes)
        x = F.softmax(x, dim=1)

        return x

    def calc_data(self, generator, dataloader, inputs_generator):
        real_features = []
        fake_features = []
        fake_probs = []
        gen_device = next(iter(generator.parameters())).device
        for idx, real_img, _ in tqdm(dataloader, leave=False):
            bs = real_img.shape[0]
            if inputs_generator is None:
                idx = idx.long().to(gen_device)
                fake_img = generator(idx=idx).to(self.device)
            else:
                inputs = torch.nan_to_num(inputs_generator(bs), posinf=1, neginf=-1).to(gen_device)
                if torch.any(torch.isnan(inputs)):
                    print(f'NaN samples occured: {inputs}')
                fake_img = generator(inputs=inputs).to(self.device)
            
            real_img = real_img.to(self.device)
            
            real_features_batch = self._features(real_img)
            real_features.append(real_features_batch.detach().cpu().numpy())   
            
            fake_features_batch = self._features(fake_img)
            fake_probs_batch = self._classifier(fake_features_batch)
            fake_features.append(fake_features_batch.detach().cpu().numpy())
            fake_probs.append(fake_probs_batch.detach().cpu().numpy())
            
        real_features = np.concatenate(real_features)
        fake_features = np.concatenate(fake_features)
        fake_probs = np.concatenate(fake_probs)

        return (real_features, 
                np.nan_to_num(fake_features, posinf=1, neginf=-1), 
                np.nan_to_num(fake_probs, posinf=0.5, neginf=-0.5))

    @staticmethod
    def calc_fid(real_features, fake_features):
        mu_r = np.mean(real_features, axis=0)
        mu_f = np.mean(fake_features, axis=0)
        cov_r = np.cov(real_features, rowvar=False)
        cov_f = np.cov(fake_features, rowvar=False)
        cov_mean = linalg.sqrtm(cov_r @ cov_f)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        sum_mu = np.sum((mu_r - mu_f)**2)
        return sum_mu + np.trace(cov_r + cov_f - 2*cov_mean)

    @staticmethod
    def calc_is(fake_probs, eps=1e-10):
        marginal_distr = np.mean(fake_probs, axis=0)[None]
        kl_divergence = fake_probs * (np.log(fake_probs+eps) - np.log(marginal_distr+eps))
        kl_sum = np.sum(kl_divergence, axis=1)
        score = np.exp(kl_sum.mean())
        return score
        

    def forward(self, generator, dataloader: DataLoader, inputs_generator=None) -> torch.Tensor:
        real_features, fake_features, fake_probs = self.calc_data(generator, dataloader, inputs_generator)
        
        fid = self.calc_fid(real_features, fake_features)

        inception_score = self.calc_is(fake_probs)

        return fid, inception_score
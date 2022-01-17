import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
from scipy import linalg

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


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
    def __init__(self, model):
        super(ValLoss, self).__init__()
        self.model = model
        self.device = next(iter(model.parameters())).device

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.features(x)

    @torch.no_grad()
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model.to_logits(x), dim=1)

    def calc_data(self, dataloader):
        real_features = []
        fake_features = []
        fake_probs = []
        for idx, real_img, _ in tqdm(dataloader, leave=False):
            idx, real_img = idx.long().to(self.device), real_img.to(self.device)
            fake_img = self.model(idx=idx)
            
            real_features_batch = self._features(real_img)
            real_features.append(real_features_batch.detach().cpu().numpy())   
            
            fake_features_batch = self._features(fake_img)
            fake_probs_batch = self._classifier(fake_features_batch)
            fake_features.append(fake_features_batch.detach().cpu().numpy())
            fake_probs.append(fake_probs_batch.detach().cpu().numpy())
            
        real_features = np.concatenate(real_features)
        fake_features = np.concatenate(fake_features)
        fake_probs = np.concatenate(fake_probs)

        return real_features, fake_features, fake_probs

    @staticmethod
    def calc_fid(real_features, fake_features):
        mu_r = np.mean(real_features, axis=1)
        mu_f = np.mean(fake_features, axis=1)
        cov_r = np.cov(real_features, rowvar=True)
        cov_f = np.cov(fake_features, rowvar=True)
        cov_mean = linalg.sqrtm(cov_r @ cov_f)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        sum_mu = np.sum((mu_r - mu_f)**2)
        return sum_mu + np.trace(cov_r + cov_f - 2*cov_mean)

    @staticmethod
    def calc_is(fake_probs):
        marginal_distr = np.mean(fake_probs, axis=0)[None]
        kl_divergence = fake_probs * (np.log(fake_probs) - np.log(marginal_distr))
        kl_sum = np.sum(kl_divergence, axis=1)
        score = np.exp(kl_sum.mean())
        return score
        

    def forward(self, dataloader: DataLoader) -> torch.Tensor:
        real_features, fake_features, fake_probs = self.calc_data(dataloader)

        fid = self.calc_fid(real_features, fake_features)

        inception_score = self.calc_is(fake_probs)

        return fid, inception_score
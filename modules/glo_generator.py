import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import gaussian_kde

from .res_block import PreActResBlock, AdaptiveBatchNorm


class SampleGenerator():
    def __init__(self, dataloader, z_dim, bw_method):
        '''
        dataloader: test or train dataloder. 
                    IMPORTANT: requires unshuffled data for better initialization
        z_dim: int, latent vectors dimentionality
        bw_method: bw_method for gaussian kde from scipy
        '''
        _, img, target = zip(*[(idx, img_, tg) for idx, img_, tg in dataloader])
        img = torch.cat(img)
        self.target = torch.cat(target)
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
    
    def get_classes(self):
        return self.target
        
    def get_z_dataset(self):
        return torch.tensor(self.z_dataset, requires_grad=True)
    
    @staticmethod
    def reproject_to_unit_ball(z):
        # Inplace reprojection
        l2norm = torch.sqrt(torch.sum(z**2, axis=1))
        ones = torch.ones_like(l2norm)
        z = z / (torch.amax(torch.vstack([l2norm, ones]), dim=0)).view(z.shape[0], 1)
        return z.float()

class LatentTree(nn.Module):
    def __init__(self, latents, node_degree, sparse, verbose=False):
        super(LatentTree, self).__init__()
        self.latents = latents
        self.node_degree = node_degree
        self.sparse = sparse
        self.verbose = verbose
        self.create_tree_()
        
    def forward(self, idx):
        return self.node_sum_(idx)

    def rebuild(self):
        device = next(iter(self.parameters())).device
        idx = torch.arange(0, len(self.latents), device=device)
        self.latents = self(idx).to(torch.device('cpu'))
        self.create_tree_()
        return self
    
    def node_sum_(self, idx):
        lat = 0
        for i, embed in enumerate(self.node_values.values()):
            lat += embed(idx)
            if i < len(self.nodes):
                idx = torch.tensor(self.nodes[i][idx])
        return lat
    
    def create_tree_(self):
        self.depth = 0
        self.nodes = nn.ParameterList()
        self.node_values = nn.ModuleDict()
        data = self.latents.clone().detach()
        n_centroids = len(data) // self.node_degree
        
        kmeans = MiniBatchKMeans(n_clusters=n_centroids, init='k-means++').fit(data)
        labels = kmeans.labels_
        centroids = torch.zeros_like(torch.tensor(kmeans.cluster_centers_))
        for label in np.unique(labels):
            label_idx = np.argwhere(labels==label).T
            centroids[label] = torch.mean(data[label_idx], dim=0)
            data[label_idx] -= centroids[label]
        self.node_values.update({str(self.depth): nn.Embedding.from_pretrained(data.clone().detach(), 
                                                                               freeze=False, 
                                                                               sparse=self.sparse)})
        self.nodes.append(nn.Parameter(torch.tensor(labels).long(), requires_grad=False))
        if self.verbose:
            print(f'Created {len(data)} initial nodes at height {self.depth}')
        data = centroids
        n_centroids = len(data) // self.node_degree
        self.depth += 1
        
        while n_centroids > 1:
            kmeans = MiniBatchKMeans(n_clusters=n_centroids, init='k-means++').fit(data)
            labels = kmeans.labels_
            centroids = torch.zeros_like(torch.tensor(kmeans.cluster_centers_))
            for label in np.unique(labels):
                label_idx = np.argwhere(labels==label).T
                centroids[label] = torch.mean(data[label_idx], dim=0)
                data[label_idx] -= centroids[label]
            self.node_values.update({str(self.depth): nn.Embedding.from_pretrained(data.clone().detach(), 
                                                                                   freeze=False, 
                                                                                   sparse=self.sparse)})
            self.nodes.append(nn.Parameter(torch.tensor(labels).long(), requires_grad=False))
            if self.verbose:
                print(f'Created {len(data)} nodes at height {self.depth}')
            data = centroids
            n_centroids = len(data) // self.node_degree
            self.depth += 1
            
        self.node_values.update({str(self.depth): nn.Embedding.from_pretrained(data.clone().detach(), 
                                                                               freeze=False, 
                                                                               sparse=self.sparse)})
        if self.verbose:
                print(f'Created {len(data)} nodes at height {self.depth}')
        


class GLOGenerator(nn.Module):
    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 latent_channels: int,
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
        
        self.embed_features = latent_channels
        self.num_blocks = num_blocks
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.normalization = normalization
        
        self.act = nn.ReLU(inplace=False)
        
        self.const = nn.Parameter(torch.randn([max_channels, 4, 4]))
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(PreActResBlock(max_channels//2**i, max_channels//2**(i+1), 
                                                  latent_channels, upsample=True, 
                                                  lrelu_slope=lrelu_slope, norm=normalization))
        if normalization == 'in':
            self.last_norm = nn.InstanceNorm2d(min_channels)
        elif normalization == 'ada':
            self.last_norm = AdaptiveBatchNorm(min_channels, latent_channels)
        elif normalization == 'gn':
            self.last_norm = nn.GroupNorm(num_groups=8, num_channels=min_channels)
        else:
            self.last_norm = nn.BatchNorm2d(min_channels)
        self.end_conv = spectral_norm(nn.Conv2d(min_channels, self.out_channels, kernel_size=3, padding=1))
        # self.fine_tune_block = nn.Sequential(nn.ReLU(), 
        #                                      nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, latents):
        bs = latents.shape[0]
        out = self.const.unsqueeze(0).repeat([bs, 1, 1, 1])
        
        for i in range(self.num_blocks):
            # Scince we have pre-act blocks, we don't need activations inbetween
            out = self.res_blocks[i](out, latents)
        if self.normalization == 'ada':
            out = self.last_norm(out, latents)
        else:
            out = self.last_norm(out)
        out = self.act(out)
        out = self.end_conv(out)
        out = F.interpolate(out, size=self.output_size, mode='bilinear')
        # out - self.fine_tune_block(out)
        
        out = self.sigmoid(out)

        assert out.shape == (latents.shape[0], self.out_channels, *self.output_size)
        return out

        

class GLOModel(nn.Module):
    def __init__(self, generator, sample_generator, sparse, node_degree, lat_regularization=None):
        super(GLOModel, self).__init__()
        self.generator = generator
        self.sample_generator = sample_generator
        self.tree = LatentTree(self.sample_generator.get_z_dataset(), node_degree, sparse, verbose=True)
    
    def forward(self, idx=None, inputs=None):
        if inputs is not None:
            return self.generator(inputs)
        return self.generator(self.tree(torch.tensor(idx)))
    
    # def optimize_to_img(self, img, loss_func, min_loss, optimizer, init_z=None):
    #     return self.generator.optimize_to_img(img, loss_func, min_loss, optimizer, init_z)
        
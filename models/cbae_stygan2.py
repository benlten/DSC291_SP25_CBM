import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.basic import Basic
import pickle

def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


class CB_AE(nn.Module):
    def __init__(self, noise_dim, concept_dim, num_ws=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, concept_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
        )

        self.num_ws = num_ws

        print('number of layers in CB-AE:', len(self.encoder) + len(self.decoder))

        self.apply(_weights_init)

    def enc(self, x):
        # the latent vector will be like (batch_size, 14, 512)
        # where the 512 vector is repeated 14 times since there are 14 layers in the stylegan synthesis network
        # so we can take any one of the 14 (but use mean to maintain differentiability)
        x = torch.mean(x, dim=1)
        return self.encoder(x)
    
    def dec(self, x):
        x = self.decoder(x)
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        return x


    def forward(self, x):
        return self.dec(self.enc(x))


class cbAE_StyGAN2(Basic):
    def _build_model(self):
        pretrained_model_path = self.config['model']['pretrained']
        print(f'loading stylegan2 from {pretrained_model_path}')
        with open(pretrained_model_path, 'rb') as f:
            self.gen = pickle.load(f)['G_ema']
        
        assert self.num_ws is not None
        print(f'Total concepts (including unknown): {sum(self.concepts_output)}')
        self.cbae = CB_AE(self.noise_dim, sum(self.concepts_output), self.num_ws)


class CC(nn.Module):
    def __init__(self, noise_dim, concept_dim, num_ws=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(noise_dim),
            nn.Linear(noise_dim, concept_dim),
        )

        self.num_ws = num_ws

        print('number of layers in CB-E:', len(self.encoder))

        self.apply(_weights_init)

    def enc(self, x):
        # the latent vector will be like (batch_size, 14, 512)
        # where the 512 vector is repeated 14 times since there are 14 layers in the stylegan synthesis network
        # so we can take any one of the 14 (but use mean to maintain differentiability)
        x = torch.mean(x, dim=1)
        return self.encoder(x)
    
    def forward(self, x):
        return self.enc(x)


class CC_StyGAN2(Basic):
    def _build_model(self):
        pretrained_model_path = self.config['model']['pretrained']
        print(f'loading stylegan2 from {pretrained_model_path}')
        with open(pretrained_model_path, 'rb') as f:
            self.gen = pickle.load(f)['G_ema']
        
        assert self.num_ws is not None
        self.cbae = CC(self.noise_dim, sum(self.concepts_output), self.num_ws)
import os
import sys
sys.path.append('.')

import pickle
import torch
import torch.nn as nn
from PIL import Image

pretrained_path = 'models/checkpoints/stylegan2-celebahq-256x256.pkl'

with open(pretrained_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

print(G.z_dim) ## expected output: 512

z = torch.randn([1, G.z_dim]).cuda()    # latent codes
w = G.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
img = G.synthesis(w, noise_mode='const')
print(w.shape) ## expected output (1, 14, 512)

os.makedirs('images', exist_ok=True)
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'images/stygan2_celebahq_test0.png')

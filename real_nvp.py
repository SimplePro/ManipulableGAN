import torch
from torch import nn
from torch import distributions

import numpy as np

from time import time

from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

import os

import matplotlib.pyplot as plt

class RealNVP(nn.Module):
    def __init__(self, dim=16):
        super(RealNVP, self).__init__()

        nets = lambda: nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, dim),
            nn.Tanh()
        )

        nett = lambda: nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, dim)
        )

        self.masks = torch.from_numpy(np.array([[0 if i < dim // 2 else 1 for i in range(dim)], [1 if i < dim // 2 else 0 for i in range(dim)]] * 3).astype(np.float32))
        # print(self.masks)
        # self.masks = torch.from_numpy(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 3).astype(np.float32))
        self.prior = distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        
        self.t = torch.nn.ModuleList([nett().to(device) for _ in range(len(self.masks))])
        self.s = torch.nn.ModuleList([nets().to(device) for _ in range(len(self.masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            s = self.s[i](x_)*(1-self.masks[i])
            t = self.t[i](x_)*(1-self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)

        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](z_) * (1-self.masks[i])
            # print(s.shape, self.masks[i])
            t = self.t[i](z_) * (1-self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)

        
        return z, log_det_J
    
    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        # return self.prior(z) + logp
    
    def sample(self, batch_size):
        z = self.prior.sample((batch_size, 1)).to(device)
        # z = torch.randn((batch_size, 1)).to(device)
        # logp = self.prior.log_prob(z)
        x = self.g(z)
        return x




def train_real_nvp(data, iters):
    flow = RealNVP(dim=data[0].shape[1])

    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)

    for t in range(iters):

        for i in range(len(data)//600):
            code = torch.concat(data[i*100:(i+1)*100], dim=0).to(device)
            loss = -flow.log_prob(code).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 500 == 0:
            print("iter %s" % t, "loss = %.3f" % loss)
            
    return flow


if __name__ == '__main__':

    device = "cpu"

    dataset = "mnist"
    expr_name = "2d_original_perceptual_wgan-gp"

    if expr_name == "2d_original_perceptual_wgan-gp":
        from original_experiments import TripletNet, ManipulableAutoEncoder, Generator

    else:
        from gan import TripletNet, ManipulableAutoEncoder, Generator

    root_dir = os.path.join(f"{dataset}_results", expr_name)

    tripletnet = TripletNet().to(device)
    tripletnet.load_state_dict(torch.load(os.path.join(root_dir, "mnist_tripletnet_state_dict.pth")))
    tripletnet.eval()

    if expr_name != "2d_original_perceptual_wgan-gp":
        manipulable_ae = ManipulableAutoEncoder().to(device)
        manipulable_ae.load_state_dict(torch.load(os.path.join(root_dir, "mnist_manipulable_ae_state_dict.pth")))
        manipulable_ae.eval()

    gen = Generator().to(device)
    gen.load_state_dict(torch.load(os.path.join(root_dir, "mnist_gen_state_dict.pth")))
    gen.eval()
    
    trainset = MNIST(
        root="./MNIST/images/",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    validset = MNIST(
        root="./MNIST/images/",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    codes = []

    for x, y in trainset:
        pred = tripletnet(x.to(device).unsqueeze(0))
        codes.append(pred.detach())
        
    start_time = time()

    flow = train_real_nvp(codes, iters=2000)
    torch.save(flow.state_dict(), os.path.join(root_dir, "real_nvp_state_dict.pth"))
    samples = flow.sample(100000)

    if expr_name == "2d_original_perceptual_wgan-gp":
        codes_2d = [code.reshape(-1) for code in codes]

    else:

        codes_2d = []

        for sample in samples:
            codes_2d.append(manipulable_ae.encode(sample.unsqueeze(0)).reshape(-1).detach())

    elapsed_time = int(time() - start_time)

    print(f"elapsed_time: {elapsed_time//3600}h {elapsed_time%3600//60}m {elapsed_time%60}s")

    plt.figure(figsize=(6, 5))
    plt.scatter([codes_2d[i][0].item() for i in range(len(codes_2d))], [codes_2d[i][1].item() for i in range(len(codes_2d))], s=0.1, alpha=0.05)
    plt.xlim((-8, 6))
    plt.ylim((-8, 9))
    plt.savefig(os.path.join(root_dir, "real_nvp_2d_visualization.png"))
    # plt.show()

    pred = gen(samples[:100])

    grid = make_grid(pred, nrow=10, normalize=True)
    img = TF.resize(TF.to_pil_image(grid), (512, 512))
    img.save(os.path.join(root_dir, "samples.png"))
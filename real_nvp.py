import torch
from torch import nn
from torch import distributions

import numpy as np

device = "cpu"

class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()

        nets = lambda: nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.Tanh()
        )

        nett = lambda: nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16)
        )

        self.masks = torch.from_numpy(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 3).astype(np.float32))
        self.prior = distributions.MultivariateNormal(torch.zeros(16), torch.eye(16))
        
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
    flow = RealNVP()

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
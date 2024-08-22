import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from random import choices, choice

import os

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class TripletDataset(Dataset):

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
        self.class_info = dataset.classes
        self.class_index = [[] for _ in range(len(dataset.classes))]

        self.y = [d[1] for d in self.dataset]

        for i in range(len(dataset)):
            self.class_index[dataset[i][1]].append(i)

    def get_positive(self, anchor):

        # positive_index = choices(list(range(len(self.dataset))), weights=weights, k=1)

        positive_index = choice(self.class_index[anchor])

        return positive_index

        # return positive_index[0]

    def get_negative(self, anchor):
        # weights = [class_ != anchor for class_ in self.classes]
        
        # negative_index = choices(list(range(len(self.dataset))), weights=weights, k=1)
        
        negative_index = choice(self.class_index[choice([i for i in range(len(self.class_index)) if i != anchor])])
        return negative_index

        # return negative_index[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        anchor = self.y[index]
        positive_index = self.get_positive(anchor)
        negative_index = self.get_negative(anchor)

        return self.dataset[index], self.dataset[positive_index], self.dataset[negative_index]


class TripletNet(nn.Module):

    def __init__(self):
        super().__init__() 

        self.representation_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.Flatten(), # (4*4*64)

            nn.Linear(4*4*64, 16),
            # nn.Tanh(0.2),
        )

    def forward(self, x):
        return self.representation_encoder(x)


class ManipulableAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.manipulable_encoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),

            # nn.Linear(16, 16),
            # nn.LeakyReLU(0.2),

            # nn.Linear(16, 16),
            # nn.LeakyReLU(0.2),

            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),

            nn.Linear(8, 2),
            nn.LeakyReLU(0.2)
        )

        self.manipulable_decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(0.2),
            
            nn.Linear(8, 16),
            nn.LeakyReLU(0.2),

            # nn.Linear(16, 16),
            # nn.LeakyReLU(0.2),

            # nn.Linear(16, 16),
            # nn.LeakyReLU(0.2),

            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 16),
            nn.LeakyReLU(0.2)
        )

    def encode(self, x):
        return self.manipulable_encoder(x)
    
    def decode(self, x):
        return self.manipulable_decoder(x)

class Reshape(nn.Module):

    def __init__(self, shape):
        super().__init__()

        self.shape = shape
    
    def forward(self, x):
        return x.view(-1, *self.shape)
    

class Resize(nn.Module):

    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)
    
class WSConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bias = self.conv.bias
        self.conv.bias = None
        
        self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.size(0), 1, 1)

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        # input shape: (16)
        self.main = nn.Sequential(
            nn.Linear(16, 7*7*32),
            nn.LeakyReLU(0.2),

            nn.Linear(7*7*32, 7*7*32),
            nn.LeakyReLU(0.2),

            Reshape(shape=(32, 7, 7)),

            # nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            WSConv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            Resize(scale_factor=2.0),
            # nn.GELU(),
            nn.LeakyReLU(0.2),

            # nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1, bias=False),
            WSConv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            Resize(scale_factor=2.0),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            WSConv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
            WSConv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        # input shape: (1, 28, 28)
        self.main = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            WSConv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (8, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            WSConv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (16, 7, 7)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(), # (16*7*7)

            nn.Linear(16*7*7, 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(16, 1),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
def gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.shape[0], 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class Trainer:

    def __init__(self, expr_name, dataset_name, trainset, validset, device="cuda", learning_rate=0.002, batch_size=64):
        self.expr_name = expr_name
        self.dataset_name = dataset_name
        self.device = device
        self.tripletnet = TripletNet().to(self.device)
        self.manipulable_ae = ManipulableAutoEncoder().to(self.device)
        self.gen = Generator().to(self.device)
        self.disc = Discriminator().to(self.device)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.triplet_loss = nn.TripletMarginLoss().to(self.device)
        self.perceptual_loss = VGGPerceptualLoss(resize=True).to(self.device)

        self.alpha = 0.9
        self.alpha_decay = 0.99

        self.lambda_gp = 10

        self.optim_triplet = Adam(self.tripletnet.parameters(), lr=self.learning_rate)
        self.optim_manipulable = Adam(self.manipulable_ae.parameters(), lr=self.learning_rate)
        self.optim_gen = Adam(self.gen.parameters(), lr=self.learning_rate)
        self.optim_disc = Adam(self.disc.parameters(), lr=self.learning_rate)

        self.trainset = trainset
        self.validset = validset
        
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size)
        self.validloader = DataLoader(validset, batch_size=self.batch_size*2)


    def train_epoch(self):
        self.tripletnet.train()
        self.manipulable_ae.train()
        
        avg_triplet_loss = 0
        avg_reconstruction_loss = 0
        avg_perceptual_loss = 0
        avg_mse_loss = 0
        avg_gen_adversarial_loss = 0
        avg_disc_adversarial_loss = 0

        for i, ((anchor, _), (positive, _), (negative, _)) in tqdm(enumerate(self.trainloader)):
            # -------------------------- 1. TripletNet ---------------------------------
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_pred = self.tripletnet(anchor)
            positive_pred = self.tripletnet(positive)
            negative_pred = self.tripletnet(negative)

            triplet_loss = self.triplet_loss(anchor_pred, positive_pred, negative_pred)

            self.optim_triplet.zero_grad()
            (0.1 * triplet_loss).backward()
            self.optim_triplet.step()

            avg_triplet_loss += triplet_loss.item()

            # ------------------------- 2. Manipulable AutoEncoder -------------------------

            ### --------------------------- Inject Noise ------------------------
            anchor_pred = self.tripletnet(anchor).detach()
            # code_pred = self.manipulable_ae.encode(anchor_pred.detach() + torch.randn_like(anchor_pred) * 0.5)
            code_pred = self.manipulable_ae.encode(anchor_pred)
            reconstruction = self.manipulable_ae.decode(code_pred)
            reconstruction_loss = F.mse_loss(reconstruction, anchor_pred)

            self.optim_manipulable.zero_grad()
            (10 * reconstruction_loss).backward()
            self.optim_manipulable.step()

            avg_reconstruction_loss += reconstruction_loss.item()

            # --------------------------- 3. Generator ( +TripletNet ) -------------------------------
            if i % 5 == 0:

                self.gen.train()
                self.disc.eval()

                code1 = self.tripletnet(anchor).detach() # (1, 16)

                ### ---------------------- Inject Noise --------------------------
                # noisy_code1 = torch.randn_like(code1) * 0.5 + code1

                # fake = self.gen(noisy_code1)

                fake = self.gen(code1)

                disc_fake_pred = self.disc(fake)

                adversarial_loss = -disc_fake_pred.mean()

                perceptual_loss = self.perceptual_loss(fake, anchor)
                # mse_loss = F.mse_loss(fake, anchor)

                loss = 10 * perceptual_loss + adversarial_loss
                # loss = adversarial_loss + 10*((1-self.alpha) * perceptual_loss + (self.alpha) * mse_loss)
                # loss = adversarial_loss + 10 * (perceptual_loss + mse_loss)
                # loss = adversarial_loss + 10 * (perceptual_loss + 5 * mse_loss)
                # loss = 5*adversarial_loss + 10*perceptual_loss
                # loss *= 10
                # loss = adversarial_loss + 10 * perceptual_loss


                self.optim_gen.zero_grad()
                self.optim_triplet.zero_grad()
                loss.backward()
                self.optim_gen.step()
                self.optim_triplet.step()

                avg_gen_adversarial_loss += adversarial_loss.item()
                avg_perceptual_loss += perceptual_loss.item()
                # avg_mse_loss += mse_loss.item()

            # ------------------------- 4. Discrminator ---------------------------
            self.gen.eval()
            self.disc.train()

            code1 = self.tripletnet(positive).detach()

            fake = self.gen(code1).detach()

            disc_fake_pred = self.disc(fake)
            disc_real_pred = self.disc(positive)

            try:
                gp = gradient_penalty(self.disc, positive, fake, self.device)
            except: gp = 0

            adversarial_loss = disc_fake_pred.mean() - disc_real_pred.mean() + self.lambda_gp * gp
            # regularization = torch.mean(disc_real_pred**2)
            # loss = adversarial_loss + 0.001 * regularization
            loss = adversarial_loss

            self.optim_disc.zero_grad()
            loss.backward()
            self.optim_disc.step()

            avg_disc_adversarial_loss += adversarial_loss.item()
        
        avg_triplet_loss /= len(self.trainloader)
        avg_reconstruction_loss /= len(self.trainloader)
        avg_perceptual_loss /= len(self.trainloader)
        avg_mse_loss /= len(self.trainloader)
        avg_gen_adversarial_loss /= len(self.trainloader)
        avg_disc_adversarial_loss /= len(self.trainloader)

        self.alpha *= self.alpha_decay

        return avg_triplet_loss, avg_reconstruction_loss, avg_perceptual_loss, avg_mse_loss, avg_gen_adversarial_loss, avg_disc_adversarial_loss

    @torch.no_grad()
    def valid_epoch(self, epoch):
        self.tripletnet.eval()
        self.manipulable_ae.eval()
        self.gen.eval()
        self.disc.eval()

        codes = []

        for _ in range(len(self.validset.class_info)):
            codes.append({
                "x": [],
                "y": []
            })

        avg_triplet_loss = 0
        avg_reconstruction_loss = 0
        avg_perceptual_loss = 0
        avg_mse_loss = 0
        avg_gen_adversarial_loss = 0
        avg_disc_adversarial_loss = 0

        for i, ((anchor, anchor_label), (positive, _), (negative, _)) in tqdm(enumerate(self.validloader)):

            # ---------------------------- 1. Triplet Net -------------------------------
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_pred = self.tripletnet(anchor)
            positive_pred = self.tripletnet(positive)
            negative_pred = self.tripletnet(negative)

            loss = self.triplet_loss(anchor_pred, positive_pred, negative_pred)
            avg_triplet_loss += loss.item()

            # ----------------------------- 2. Manipulable AutoEncoder -----------------------------
            
            code_pred = self.manipulable_ae.encode(anchor_pred)
            reconstruction = self.manipulable_ae.decode(code_pred)

            reconstruction_loss = F.mse_loss(reconstruction, anchor_pred)

            avg_reconstruction_loss += reconstruction_loss.item()

            for i in range(len(anchor)):
                label = anchor_label[i]
                pred = code_pred[i]

                ## ----------------------- Inject Noise -------------------------
                # codes[label]["x"].append(pred[0].item() + torch.randn(1).item() * 0.5)
                # codes[label]["y"].append(pred[1].item() + torch.randn(1).item() * 0.5)
                codes[label]["x"].append(pred[0].item())
                codes[label]["y"].append(pred[1].item())

            # --------------------------------- 3. Generator ----------------------------------
            if i % 5 == 0:
                code1 = self.tripletnet(anchor).detach()
                ## ------------------------- Inject Noise ----------------------------
                # noisy_code1 = torch.randn_like(code1) * 0.5 + code1
                # fake = self.gen(noisy_code1).detach()

                fake = self.gen(code1).detach()
            
                perceptual_loss = self.perceptual_loss(fake, anchor)
                adversarial_loss = -self.disc(fake).mean()
                # mse_loss = F.mse_loss(fake, anchor)

                avg_gen_adversarial_loss += adversarial_loss.item()
                avg_perceptual_loss += perceptual_loss.item()
                # avg_mse_loss += mse_loss.item()

            # ----------------------------------- 4. Discriminator ----------------------------
            code1 = self.tripletnet(positive).detach()
            fake = self.gen(code1).detach()

            disc_fake_pred = self.disc(fake)
            disc_real_pred = self.disc(positive)

            try:
                gp = gradient_penalty(self.disc, positive, fake, self.device)
            except: gp = 0

            adversarial_loss = disc_fake_pred.mean() - disc_real_pred.mean() + self.lambda_gp * gp

            avg_disc_adversarial_loss += adversarial_loss.item()
            

        for i in range(len(codes)):
            plt.scatter(codes[i]["x"], codes[i]["y"], s=1, label=self.validset.class_info[i])

        plt.legend()
        plt.savefig(os.path.join(f"{self.dataset_name}_results", self.expr_name, "2d_visualization", str(epoch+1)+".jpg"))
        plt.clf()

        grid = make_grid(fake.cpu(), normalize=True).permute(1, 2, 0)
        # print(type(grid))

        plt.imshow(grid)
        plt.savefig(os.path.join(f"{self.dataset_name}_results", self.expr_name, "generation", str(epoch+1)+".jpg"))
        plt.clf()
    
        avg_triplet_loss /= len(self.validloader)
        avg_reconstruction_loss /= len(self.validloader)
        avg_perceptual_loss /= len(self.validloader)
        avg_mse_loss /= len(self.validloader)
        avg_gen_adversarial_loss /= len(self.validloader)
        avg_disc_adversarial_loss /= len(self.validloader)

        return avg_triplet_loss, avg_reconstruction_loss, avg_perceptual_loss, avg_mse_loss, avg_gen_adversarial_loss, avg_disc_adversarial_loss
    
    def run(self, EPOCHS):

        train_loss = {
            "triplet": [],
            "rec": [],
            "perceptual": [],
            "mse": [],
            "gen_adversarial": [],
            "disc_adversarial": []
        }
        
        
        valid_loss = {
            "triplet": [],
            "rec": [],
            "perceptual": [],
            "mse": [],
            "gen_adversarial": [],
            "disc_adversarial": []
        }

        for epoch in range(EPOCHS):
            print(f"EPOCH: {epoch+1}/{EPOCHS}, alpha: {self.alpha}")
            train_tripletloss, train_rec_loss, train_per_loss, train_mse_loss, train_gen_adv_loss, train_disc_adv_loss = self.train_epoch()
            valid_tripletloss, valid_rec_loss, valid_per_loss, valid_mse_loss, valid_gen_adv_loss, valid_disc_adv_loss = self.valid_epoch(epoch)

            print(f"TRAIN_LOSS\n(triplet: %.4f, rec: %.4f, perceptual: %.4f, mse: %.4f, gen-adv: %.4f, disc-adv: %.4f)\nVALID_LOSS\n(triplet: %.4f, rec: %.4f, perceptual: %.4f, mse: %.4f, gen-adv: %.4f, disc-adv: %.4f)\n" 
                  %(train_tripletloss, train_rec_loss, train_per_loss, train_mse_loss, train_gen_adv_loss, train_disc_adv_loss, valid_tripletloss, valid_rec_loss, valid_per_loss, valid_mse_loss, valid_gen_adv_loss, valid_disc_adv_loss))

            train_loss["triplet"].append(train_tripletloss)
            train_loss["rec"].append(train_rec_loss)
            train_loss["perceptual"].append(train_per_loss)
            train_loss["mse"].append(train_mse_loss)
            train_loss["gen_adversarial"].append(train_gen_adv_loss)
            train_loss["disc_adversarial"].append(train_disc_adv_loss)

            valid_loss["triplet"].append(valid_tripletloss)
            valid_loss["rec"].append(valid_rec_loss)
            valid_loss["perceptual"].append(valid_per_loss)
            valid_loss["mse"].append(valid_mse_loss)
            valid_loss["gen_adversarial"].append(valid_gen_adv_loss)
            valid_loss["disc_adversarial"].append(valid_disc_adv_loss)

            for key in list(train_loss.keys()):
                plt.plot(range(len(train_loss[key])), train_loss[key], label=key)
            
            plt.legend()
            plt.title("train_loss")
            plt.savefig(os.path.join(f"{self.dataset_name}_results", self.expr_name, "train_loss.png"))
            plt.clf()
            
            for key in list(valid_loss.keys()):
                plt.plot(range(len(valid_loss[key])), valid_loss[key], label=key)
                
            plt.legend()
            plt.title("valid_loss")
            plt.savefig(os.path.join(f"{self.dataset_name}_results", self.expr_name, "valid_loss.png"))
            plt.clf()


if __name__ == '__main__':

    expr_name = "gen_manipulable_ae_v4_perceptual_wgan-gp"
    dataset = "mnist"

    os.makedirs(os.path.join(f"{dataset}_results", expr_name, "2d_visualization"), exist_ok=True)
    os.makedirs(os.path.join(f"{dataset}_results", expr_name, "generation"), exist_ok=True)

    if dataset == "mnist":
        
        trainset = TripletDataset(
            dataset=MNIST(
                root="./MNIST/images/",
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
        )

        validset = TripletDataset(
            dataset=MNIST(
                root="./MNIST/images/",
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
        )

    elif dataset == "fashion_mnist":
        trainset = TripletDataset(
            dataset=FashionMNIST(
                root="./FashionMNIST/images/",
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
        )

        validset = TripletDataset(
            dataset=FashionMNIST(
                root="./FashionMNIST/images/",
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
        )


    trainer = Trainer(
        expr_name=expr_name,
        dataset_name=dataset,
        trainset=trainset,
        validset=validset,
        device="cuda",
        learning_rate=0.002,
        batch_size=64,
    )

    trainer.run(EPOCHS=200)

    torch.save(trainer.tripletnet.state_dict(), f"{dataset}_results/{expr_name}/{dataset}_tripletnet_state_dict.pth")
    torch.save(trainer.manipulable_ae.state_dict(), f"{dataset}_results/{expr_name}/{dataset}_manipulable_ae_state_dict.pth")
    torch.save(trainer.gen.state_dict(), f"{dataset}_results/{expr_name}/{dataset}_gen_state_dict.pth")
    torch.save(trainer.disc.state_dict(), f"{dataset}_results/{expr_name}/{dataset}_disc_state_dict.pth")
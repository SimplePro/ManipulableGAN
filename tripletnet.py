import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import MNIST
from torchvision import transforms

import matplotlib.pyplot as plt

from random import choices, choice

import os

from tqdm import tqdm


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
            nn.GELU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Flatten(), # (7*7*32)

            nn.Linear(7*7*32, 16),
            nn.GELU(),
        )

        self.manipulable_encoder = nn.Sequential(
            nn.Linear(16, 2)
        )

        self.manipulable_decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU()
        )
        

    def forward(self, x):
        return self.representation_encoder(x)

class Trainer:

    def __init__(self, trainset, validset, device="cuda", learning_rate=0.002, batch_size=64, visualization_plot_path=None):
        self.device = device
        self.model = TripletNet().to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.criterion = nn.TripletMarginLoss().to(self.device)

        self.optim = Adam(self.model.parameters(), lr=self.learning_rate)

        self.trainset = trainset
        self.validset = validset
        
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size)
        self.validloader = DataLoader(validset, batch_size=self.batch_size*2)

        self.visualization_plot_path = visualization_plot_path


    def train_epoch(self):
        self.model.train()
        
        avg_loss = 0
        avg_reconstruction_loss = 0

        for (anchor, _), (positive, _), (negative, _) in tqdm(self.trainloader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_pred = self.model(anchor)
            positive_pred = self.model(positive)
            negative_pred = self.model(negative)

            loss = self.criterion(anchor_pred, positive_pred, negative_pred)

            code_pred = self.model.manipulable_encoder(anchor_pred)
            reconstruction = self.model.manipulable_decoder(code_pred)
            reconstruction_loss = F.mse_loss(reconstruction, anchor_pred)

            self.optim.zero_grad()
            loss.backward(retain_graph=True)
            reconstruction_loss.backward()
            self.optim.step()

            avg_loss += loss.item()
            avg_reconstruction_loss += reconstruction_loss.item()

        return avg_loss / len(self.trainloader), avg_reconstruction_loss / len(self.trainloader)

    @torch.no_grad()
    def valid_epoch(self, epoch):
        self.model.eval()

        codes = []

        for _ in range(len(self.validset.class_info)):
            codes.append({
                "x": [],
                "y": []
            })

        avg_loss = 0
        avg_reconstruction_loss = 0

        for (anchor, anchor_label), (positive, _), (negative, _) in tqdm(self.validloader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            anchor_pred = self.model(anchor)
            positive_pred = self.model(positive)
            negative_pred = self.model(negative)

            loss = self.criterion(anchor_pred, positive_pred, negative_pred)
            
            code_pred = self.model.manipulable_encoder(anchor_pred)
            reconstruction = self.model.manipulable_decoder(code_pred)

            reconstruction_loss = F.mse_loss(reconstruction, anchor_pred)

            avg_loss += loss.item()
            avg_reconstruction_loss += reconstruction_loss.item()

            if self.visualization_plot_path is not None:
                for i in range(len(anchor)):
                    label = anchor_label[i]
                    pred = code_pred[i]

                    codes[label]["x"].append(pred[0].item())
                    codes[label]["y"].append(pred[1].item())

        if self.visualization_plot_path is not None:
            for i in range(len(codes)):
                plt.scatter(codes[i]["x"], codes[i]["y"], s=1, label=self.validset.class_info[i])

            plt.legend()
            plt.savefig(os.path.join(self.visualization_plot_path, str(epoch+1)+".jpg"))
            plt.clf()

        return avg_loss / len(self.validloader), avg_reconstruction_loss / len(self.trainloader)

    
    def run(self, EPOCHS):

        for epoch in range(EPOCHS):
            train_tripletloss, train_rec_loss = self.train_epoch()
            valid_tripletloss, valid_rec_loss = self.valid_epoch(epoch)

            print(f"EPOCH: {epoch+1}/{EPOCHS}, train_loss: (%.4f, %.4f), valid_loss: (%.4f, %.4f)"%(train_tripletloss, train_rec_loss, valid_tripletloss, valid_rec_loss))
            

if __name__ == '__main__':

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

    trainer = Trainer(
        trainset=trainset,
        validset=validset,
        device="cuda",
        learning_rate=0.002,
        batch_size=64,
        visualization_plot_path="./2d_visualization/"
    )

    trainer.run(EPOCHS=10)

    torch.save(trainer.model.state_dict(), "mnist_tripletnet_state_dict.pth")
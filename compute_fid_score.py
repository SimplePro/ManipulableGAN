import os
import torch
from gan import TripletNet, Generator
from real_nvp import RealNVP

from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import torchvision.transforms.functional as TF

if __name__ == '__main__':

    device = "cuda"
    expr_name = "gen_manipulable_ae_v2_denoising_perceptual_wgan-gp"
    dataset = "mnist"
    generate_type = "reconstruction" # ["reconstruction", "sampling"]

    img_path = os.path.join("fid_dataset", dataset + "_" + generate_type + "_" + expr_name)

    os.makedirs(img_path, exist_ok=True)
    
    tripletnet = TripletNet().to(device)
    tripletnet.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "_tripletnet_state_dict.pth")))

    gen = Generator().to(device)
    gen.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "_gen_state_dict.pth")))

    # real_nvp = RealNVP().to("cpu")
    # real_nvp.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "_real_nvp_state_dict.pth")))

    if dataset == "mnist":
        
        # trainset = MNIST(
        #     root="./MNIST/images/",
        #     train=True,
        #     download=True,
        #     transform=transforms.ToTensor()
        # )

        validset = MNIST(
            root="./MNIST/images/",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    elif dataset == "fashion_mnist":
        # trainset = FashionMNIST(
        #     root="./FashionMNIST/images/",
        #     train=True,
        #     download=True,
        #     transform=transforms.ToTensor()
        # )

        validset = FashionMNIST(
            root="./FashionMNIST/images/",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    if generate_type == "reconstruction":
        
        for i, (x, y) in enumerate(validset):
            code_pred = tripletnet(x.to(device).unsqueeze(0))

            img = gen(code_pred).squeeze(0).to("cpu").detach()
            img = TF.to_pil_image(img)

            img.save(os.path.join(img_path, str(i) + ".jpg"))

    elif generate_type == "sampling":

        pass
    
    
    os.system(f"python -m pytorch_fid {img_path} {os.path.join("fid_dataset", dataset)}")
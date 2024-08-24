import os
import torch
from gan import TripletNet, Generator, ManipulableAutoEncoder
# from original_experiments import TripletNet, Generator
from real_nvp import RealNVP

from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import torchvision.transforms.functional as TF

if __name__ == '__main__':

    device = "cuda"
    # expr_name = "2d_original_perceptual_wgan-gp"
    expr_name = "gen_manipulable_ae_v4_perceptual_wgan-gp"
    dataset = "mnist"
    generate_type = "reconstruction" # ["reconstruction", "sampling"]

    img_path = os.path.join("fid_dataset", dataset + "_" + generate_type + "_" + expr_name)

    os.makedirs(img_path, exist_ok=True)
    
    tripletnet = TripletNet().to(device)
    tripletnet.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "_tripletnet_state_dict.pth")))
    tripletnet.eval()

    gen = Generator().to(device)
    gen.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "_gen_state_dict.pth")))
    gen.eval()

    if generate_type == "sampling":
        real_nvp = RealNVP().to("cpu")
        real_nvp.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "_real_nvp_state_dict.pth")))
        real_nvp.eval()
        
        manipulable_ae = ManipulableAutoEncoder().to(device)
        manipulable_ae.load_state_dict(torch.load(os.path.join(dataset + "_results", expr_name, dataset + "manipulable_ae_state_dict.pth")))
        manipulable_ae.eval()


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
            img = img - torch.min(img)
            img = img / torch.max(img)
            img = TF.to_pil_image(img)

            img.save(os.path.join(img_path, str(i) + ".jpg"))

    elif generate_type == "sampling":

        samples = real_nvp.sample(10000)

        for i, sample in samples:
            sample = sample.to(device).unsqueeze(0)
            code = manipulable_ae.decode(sample)
            
            img = gen(code).squeeze(0).to("cpu").detach()
            img = img - torch.min(img)
            img = img / torch.max(img)
            img = TF.to_pil_image(img)
            
            img.save(os.path.join(img_path, str(i) + ".jpg"))
    
    
    os.system(f"python -m pytorch_fid {img_path} {os.path.join('fid_dataset', dataset)}")
    
    # gen_manipulable_ae_v4_perceptual_wgan-gp (reconstruction) - FID:  63.38593860529954
    # 2d_original_perceptual_wgan-gp (reconstruction) - FID:  128.22291183497998
    

import torch
import torchvision
from torch.utils.data import DataLoader
import os
from PIL import Image

from matplotlib import  pyplot as plt

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
def get_data(args):
    transfrom = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transfrom)
    dataloader = DataLoader(dataset=dataset,batch_size= args.batch_size, shuffle=True)
    
    return dataloader


def setup_logging(run_name):
    os.makedirs("models",exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models",run_name), exist_ok=True)
    
    
    
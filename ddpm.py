import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from modules import UNet
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


class Diffusion:
    def __init__(self, noise_step: int = 1000, beta_start: float = 1e-4, beta_end:float = 0.02, img_size: int = 64) -> None:
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = self.prepare_beta()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)
    
    def prepare_beta(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_step,device="cuda")    
    
    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        noise = torch.rand_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise
    
    def sample_timestep(self, n):
        return torch.randint(low = 1, high=self.noise_step, size=(n,),device=self.device)
    
    def sample(self, model, n):
        logging.info("Sampling {} new images...".format(n))
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size),device=self.device)
            for i in tqdm(reversed(range(1, self.noise_step)), position=0):
                t = (torch.ones(n, device=self.device)*i).long()
                predict_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha)/(torch.sqrt(1 - alpha_hat)) * predict_noise) + torch.sqrt(beta)* noise
        
        x = x.clamp(-1,1) + 1
        x = x * 2
        x = (x*255).type(torch.int8)
        return x
    
        
        
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args=args)
    model = UNet().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss().to(device=device)
    diffusion = Diffusion()
    logger = SummaryWriter(log_dir=os.path.join("runs",args.run_name))
    l = len(dataloader)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}...")        
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            iamges = images.to(device)
            t = diffusion.sample_timestep(images.shape[0])
            x_t, noise = diffusion.noise_image(images,t)
            predict_noise = model(x_t, t)
            loss = mse(noise, predict_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE=loss.item())
            
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l +i)
        
        sampled_images = diffusion.sample(model=model, n=images.shape[0])
        save_images(sampled_images, path= os.path.join("results", args.run_name,f"{epoch}.jpg"))
        
        
def launch():
    import argparse
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.run_name = "DDPM"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = "data/path/here"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == "__main__":
    launch()
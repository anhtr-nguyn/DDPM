import logging
import torch
import torch.nn as nn


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
    
    
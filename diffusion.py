import torch
import tqdm
from config import Config

class Diffusion:
    def __init__(self, noise_steps=None, beta_start=None, beta_end=None, img_size=None, device=None):
        # get hyperparameters
        self.noise_steps = noise_steps or Config.noise_steps
        self.beta_start = beta_start or Config.beta_start
        self.beta_end = beta_end or Config.beta_end
        self.img_size = img_size or Config.image_size
        self.device = device or Config.device

        # beta, alpha
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        # random noise
        eps = torch.randn_like(x)
        
        # set variables related to alpha
        # [:, None, None, None] means matching shape with x(N, c, h, w)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] 
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        # x_t
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return noisy_image, eps

    # sampling t as batch size
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)
        
    @torch.no_grad()
    def denoise(self, model, n, sketches, cfg_scale=None):
        if cfg_scale is None:
            cfg_scale = Config.cfg_scale
        
        # eval
        model.eval()
        
        # x_T
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        
        for i in tqdm(reversed(range(self.noise_steps)), total=self.noise_steps, desc="Sampling"):
            t = (torch.ones(n) * i).long().to(self.device)
            
            # cfg
            if cfg_scale > 0:
                # uncondition: use zero
                uncond_sketches = torch.zeros_like(sketches)
                predicted_noise_uncond = model(x, t, uncond_sketches)
                
                # condition
                predicted_noise_cond = model(x, t, sketches)
                
                # final score
                predicted_noise = predicted_noise_uncond + cfg_scale * (predicted_noise_cond - predicted_noise_uncond)
            else:
                # not use cfg
                predicted_noise = model(x, t, sketches)
                
            # denosing
            alpha_t = self.alpha[t][:, None, None, None]
            alpha_hat_t = self.alpha_hat[t][:, None, None, None]
            beta_t = self.beta[t][:, None, None, None]
            
            # x_{t-1}
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise)
            
            # add a little noise without t=0
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
                
        # train
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2 # (-1 ~ 1) -> (0 ~ 1)
        return x
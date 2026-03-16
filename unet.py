import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, H * W).transpose(1, 2)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return x + attn_out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.double_conv(x)

class unetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class unetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        # Upsampling & get diff height, width pixels
        x1 = self.up(x1)
        diff_height = x2.shape[2] - x1.shape[2] 
        diff_width = x2.shape[3] - x1.shape[3]
        
        # padding to x1 for matching the shape with x2
        x1 = F.pad(x1, [diff_width // 2, diff_width - (diff_width // 2), diff_height // 2, diff_height - (diff_height // 2)])
        
        # shorcut
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# positional encoding 
class PositionEncoding(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        encoding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)
        return encoding

# 3층 UNet
class UNet(nn.Module):
    def __init__(self, c_in=4, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
    
        self.inc = DoubleConv(c_in, 64)
        
        self.down1 = unetEncoder(64, 128)
        self.down2 = unetEncoder(128, 256)
        self.down3 = unetEncoder(256, 256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512) # maintain channels
        self.attention_block = SelfAttention(512) # attention block 
        self.bot3 = DoubleConv(512, 256)

        self.up1 = unetDecoder(512, 128) # 512
        self.up2 = unetDecoder(256, 64)
        self.up3 = unetDecoder(128, 64)
        
        # pointwise conv(1x1 conv)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.pos_embedding = nn.Sequential(
            PositionEncoding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        
        self.emb_bot = nn.Linear(time_dim, 256)
        self.emb_up1 = nn.Linear(time_dim, 128)
        self.emb_up2 = nn.Linear(time_dim, 64)
        self.emb_up3 = nn.Linear(time_dim, 64)

    def forward(self, x, t, y_cond, p_uncond=0.1):
        if self.training and p_uncond > 0:
            mask = torch.rand(x.shape[0], device=x.device) > p_uncond
            mask = mask.view(-1, 1, 1, 1).float()
            y_cond = y_cond * mask
            
        x_input = torch.cat([x, y_cond], dim=1)
        
        t_emb = self.pos_embedding(t)
        
        x1 = self.inc(x_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        
        x4 = self.attention_block(x4)
        
        x4 = self.bot3(x4) + self.emb_bot(t_emb)[:, :, None, None]

        x = self.up1(x4, x3) + self.emb_up1(t_emb)[:, :, None, None] # match 128 channels
        x = self.up2(x, x2) + self.emb_up2(t_emb)[:, :, None, None]  # match 64 channels
        x = self.up3(x, x1) + self.emb_up3(t_emb)[:, :, None, None]  # match 64 channels
        
        output = self.outc(x)
        return output

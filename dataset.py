from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageFilter
import numpy as np
import random

class AnimeColorizationDataset(Dataset):
    def __init__(self, split="train", image_size=256):
        print(f"[{split}] 데이터셋을 불러오는 중입니다...")
        self.dataset = load_dataset("cudata/Anime-face-dataset-diffusion_model", split=split) 
        
        # resize
        self.base_resize = transforms.Resize((image_size, image_size))
        
        # color
        self.color_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # -1 ~ 1
        ])
        
        # sketch
        self.sketch_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # -1 ~ 1
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image'].convert("RGB")
        img = self.base_resize(img)
        
        # data argumentation(horizontal flip)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        # color tensor
        color_tensor = self.color_to_tensor(img)
        
        # sketch tensor
        edges = img.filter(ImageFilter.FIND_EDGES)
        edges_inverted = Image.fromarray(255 - np.array(edges)).convert("L")
        sketch_tensor = self.sketch_to_tensor(edges_inverted)
        
        return color_tensor, sketch_tensor

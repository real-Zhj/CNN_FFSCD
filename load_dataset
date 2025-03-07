import torch
import os
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir, num_begin, num_end, transform=None, aug=None, num_aug=0):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.num_begin = num_begin
        self.num_end = num_end
        self.transform = transform
        self.aug = aug
        self.num_aug = num_aug

        #load data in[num_begin, num_end]
        self.feature_samples = sorted([f for f in os.listdir(feature_dir) if num_begin <= int(f[7:]) <= num_end],key=lambda x: int(x[7:]))
        self.label_samples = sorted([f for f in os.listdir(label_dir) if num_begin <= int(f[7:]) <= num_end],key=lambda x: int(x[7:]))

        self.num_sum = len(self.feature_samples)
        self.total_samples = self.num_sum * (1 + self.num_aug)

        print(f'All data: {self.total_samples} \n')

    def __len__(self):
        return self.total_samples

    def _load_sample(self, idx):
        feature_sample_path = os.path.join(self.feature_dir, self.feature_samples[idx])
        label_sample_path = os.path.join(self.label_dir, self.label_samples[idx])

        feature_imgs = []
        for i in range(4):
            feature_img = Image.open(os.path.join(feature_sample_path, f"feature_{i}.bmp")).convert("L")
            feature_img = transforms.ToTensor()(feature_img)
            feature_img = feature_img.squeeze(0)
            feature_imgs.append(feature_img)

        label_imgs = []
        for i in range(3):
            label_csv = pd.read_csv(os.path.join(label_sample_path, f"label_{i}.csv"), header=None)
            label_img = transforms.ToTensor()(label_csv.values).to(dtype=torch.float32)
            label_img = label_img.squeeze(0)
            label_imgs.append(label_img)

        feature_imgs = torch.stack(feature_imgs)  # [4, H, W]
        label_imgs = torch.stack(label_imgs)  # [3, H, W]

        return feature_imgs, label_imgs

    def __getitem__(self, idx):
        original_idx = idx // (1 + self.num_aug)
        feature_imgs, label_imgs = self._load_sample(original_idx)

        if self.aug and idx % (1 + self.num_aug) != 0:
            imgs = torch.cat((feature_imgs, label_imgs), dim=0)  # [7, H, W]
            imgs = self.aug(imgs)
            feature_imgs = imgs[:4]  # [4, H, W]
            label_imgs = imgs[4:]  # [3, H, W]

        if self.transform:
            feature_imgs = self.transform(feature_imgs)

        return feature_imgs, label_imgs

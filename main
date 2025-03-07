import torch
import time
from Load_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from Train import train_ch6
from Net import EncoderDecoder
from d2l import torch as d2l
from torchvision import transforms


aug_num = 4
batch_size = 64
num_epochs = 400
lr = 0.0001

data_dir = "../data/"
feature_dir = data_dir+"features/"
label_dir = data_dir+"labels/"
transform = transforms.Normalize([0.2366, 0.2860, 0.2366, 0.1890], [0.2873, 0.3474, 0.2873, 0.2479])
aug=transforms.RandomResizedCrop(224, ratio=(1, 1))
dataset = CustomDataset(feature_dir, label_dir, 0, 40999, transform, aug, aug_num)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = EncoderDecoder()
train_ch6(net, train_loader, test_loader, num_epochs, lr, d2l.try_gpu())
torch.save(net.state_dict(), '../net_trained/net_trained.pt')

import torch
from Load_dataset import CustomDataset
from torch.utils.data import DataLoader


def get_mean_std_value(loader):
    data_sum,data_squared_sum,num_batches = 0,0,0
    for data,_ in loader:
        data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        num_batches += 1
    mean = data_sum/num_batches
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

data_dir = "../data/"
feature_dir = data_dir+"features/"
label_dir = data_dir+"labels/"
dataset = CustomDataset(feature_dir, label_dir, 0, 40999)

train_size = int(1 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dataset_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
mean,std = get_mean_std_value(dataset_loader)
print('mean = {},std = {}'.format(mean,std))


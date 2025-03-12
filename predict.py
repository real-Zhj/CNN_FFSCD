import torch
from torch.utils.data import DataLoader
from Load_dataset import CustomDataset
from d2l import torch as d2l
from Loss import loss_one, loss
from torchvision import transforms
import os


def Predict(net, writer, feature_dir, label_dir, transform, sample_start_num, sample_end_num):
    batch_size = 128

    dataset_show_X = CustomDataset(feature_dir, label_dir, sample_start_num, sample_end_num)
    pre_iter_show_X = DataLoader(dataset_show_X, batch_size=128, shuffle=False)
    step = 0
    for X, Y in pre_iter_show_X:
        X = X.reshape((4 * len(X), 1, 224, 224))
        writer.add_images("features", X, step)
        step += 1

    loss_sum = 0
    batch_sum = 0
    dataset = CustomDataset(feature_dir, label_dir, sample_start_num, sample_end_num, transform=transform)
    pre_iter = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X, Y in pre_iter:
            X = X.to(d2l.try_gpu())
            Y = Y.to(d2l.try_gpu())

            Y_hat = net(X)
            if sample_start_num == sample_end_num:
                loss_sum += loss_one(Y_hat, Y)
            else:
                loss_sum += loss(Y_hat, Y)

            Y_hat = Y_hat.reshape((3 * len(Y_hat), 1, 224, 224))
            Y = Y.reshape((3 * len(Y), 1, 224, 224))
            for i in range(len(Y_hat)):
                Y_hat_max = round(torch.max(Y_hat[i]).item(), 4)
                Y_hat_min = round(torch.min(Y_hat[i]).item(), 4)
                Y_max = round(torch.max(Y[i]).item(), 4)
                Y_min = round(torch.min(Y[i]).item(), 4)
                print(f'No.{int(i/3)}_{int(i%3)}  predict：', "{:.4f}".format(Y_hat_min), "{:.4f}".format(Y_hat_max),\
                                 '\tground truth：', "{:.4f}".format(Y_min),   "{:.4f}".format(Y_max))
                Y_hat[i] = (Y_hat[i] - Y_hat_min) / (Y_hat_max - Y_hat_min)
                Y_hat_img = transforms.ToPILImage()(Y_hat[i])
                Y_hat_img.save(os.path.join("../data/predict_EXP/", str(int(i/3))+"_"+str(int(i%3))+ "_predict_"+str(Y_hat_max)+"_"+str(Y_hat_min)+".bmp"))
            writer.add_images("predict_labels", Y_hat, batch_sum)
            writer.add_images("real_labels", Y, batch_sum)
            batch_sum += 1
    writer.close()
    print('batch_sum：', batch_sum)
    print('mean loss：', loss_sum/batch_sum)

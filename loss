import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def loss(Y_hat, Y):
    a = 0.5
    b = 0.25
    c = 1 - a - b
    loss1 = 1-ms_ssim(Y_hat, Y, data_range=1, size_average=True)
    loss2 = (torch.abs(Y_hat-Y)).mean()
    loss3 = (torch.pow((Y_hat - Y), 2)).mean()
    return a*loss1 + b*loss2 + c*loss3



def loss_one(Y_hat, Y):
    a = 0.5
    b = 0.25
    c = 1 - a - b
    loss_sum = 0
    for i in range(len(Y_hat[0])):
        Y_hat_i = Y_hat[0][i]
        Y_i = Y[0][i]
        Y_hat_i = torch.unsqueeze(Y_hat_i, dim=0)
        Y_hat_i = torch.unsqueeze(Y_hat_i, dim=0)
        Y_i = torch.unsqueeze(Y_i, dim=0)
        Y_i = torch.unsqueeze(Y_i, dim=0)
        loss1 = 1-ms_ssim(Y_hat_i, Y_i, data_range=1, size_average=True)
        loss2 = (torch.abs(Y_hat_i-Y_i)).mean()
        loss3 = (torch.pow((Y_hat_i - Y_i), 2)).mean()
        loss_sum += a*loss1 + b*loss2 + c*loss3
    return loss_sum/len(Y_hat)

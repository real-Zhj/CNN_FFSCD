import torch
from torch import nn
from Loss import loss
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter


def evaluate_accuracy_gpu(net, test_iter, device=None):
    net.eval()
    l_sum = 0
    with torch.no_grad():
        for X, Y in test_iter:
            X = X.to(device)
            Y = Y.to(device)
            y_hat = net(X)
            l = loss(y_hat, Y)
            l_sum += l
    return l_sum / len(test_iter)

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('Start training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    timer, num_batches = d2l.Timer(), len(train_iter)
    actual_samples_sum = 0
    writer = SummaryWriter(log_dir='../log/log_runs')
    for epoch in range(num_epochs):
        l_sum = 0
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                l_sum += l
                actual_samples_sum += X.shape[0]
            timer.stop()

        scheduler.step()
        print(f'Epoch [{epoch + 1}/50], LR: {scheduler.get_last_lr()[0]:.6f}')

        train_loss = l_sum/num_batches
        test_loss = evaluate_accuracy_gpu(net, test_iter, device)
        writer.add_scalars('loss', {'_test':test_loss, '_train':train_loss}, epoch)

        print(f'epoch: {epoch+1}/{num_epochs}', f'\t',\
              f'train_loss: ', "{:.4f}".format(train_loss.item()), f'\t',\
              f'test_loss: ', "{:.4f}".format(test_loss.item()),f'\t')

    writer.close()

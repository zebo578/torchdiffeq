import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')"选择网络类型：Resnet或者NODE
parser.add_argument('--tol', type=float, default=1e-3)"求解器的误差容限
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])"是否使用伴随法
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])"降采样方法：用普通卷积还是ResBlock
parser.add_argument('--nepochs', type=int, default=160)"村里的epoch数（完整遍历数据集的次数）
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])"是否使用数据增强
parser.add_argument('--lr', type=float, default=0.1)"设置的学习率
parser.add_argument('--batch_size', type=int, default=128)"训练时每批的样本数
parser.add_argument('--test_batch_size', type=int, default=1000)"测试时每批的样本数（可以更大因为不需要计算梯度）

parser.add_argument('--save', type=str, default='./experiment1')"保存日志位置
parser.add_argument('--debug', action='store_true')"是否开启调试模式
parser.add_argument('--gpu', type=int, default=0)"使用哪块GPU（编号）
args = parser.parse_args()"获取上述参数

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint"依旧是统一命名ode求解器
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding""""""3x3 卷积，带填充（padding=1 保持尺寸不变）"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
"in_planes: 输入通道数
"out_planes: 输出通道数
"bias=False: 不使用偏置（因为后面有 BatchNorm/GroupNorm）
"ODE 中，动力学函数会被 ODE 求解器反复调用，次数不固定。BatchNorm 的 running statistics 会变得混乱，而 GroupNorm 只依赖当前输入

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution""" """1x1 卷积（用于改变通道数，不改变空间尺寸）"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)"""归一化层：GroupNorm，分组数取 min(32, dim)，dim输入通道数"""
"归一化层的目的是让数据分布稳定，加速训练

class ResBlock(nn.Module):
    expansion = 1"通道扩展倍数，但前面卷积层只是1*1所以通道数没变，不需要扩展通道匹配输入通道数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)"第一个归一化层
        self.relu = nn.ReLU(inplace=True)"激活函数
        self.downsample = downsample"可选的下采样层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)"第二个归一化层
        self.conv2 = conv3x3(planes, planes)
"这是一个新的设计norm-relu-conv;而非传统的conv-norm-relu，这样可以让梯度流动更流畅
    def forward(self, x):
        shortcut = x"原始输入或下采样后的输入

        out = self.relu(self.norm1(x))"主分支学习的特征变换

        if self.downsample is not None:
            shortcut = self.downsample(out)"调整shortcut形状，输出与out匹配，解决维度不匹配的问题

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)"经过卷积-归一化-激活函数-卷积后的特征

        return out + shortcut"残差相加：经过两层卷积处理的特征+原始特征或下采样后的特征（下采样指的是降低数据的空间分辨率，让特征图变小）


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
"dim_in:输入特征的通道数；dim_out：输出特征的通道数；ksize:卷积核大小；stride：步长；padding:填充；transpose:是否转置卷积
"这里dim_in+1也是为了匹配多出来的一个时间维度

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t"创建时间通道
        ttx = torch.cat([tt, x], 1)"把时间拼接到输入特征中去
        return self._layer(ttx)"用于（dim_in+1）通道的卷积处理
"时间感知的卷积层，专门为了neural ode设计，其核心思想是：把时间t作为额外的输入通道，让卷积操作知道当前处于ode积分的哪个时刻
"这样ode函数就能根据不同时间点产生不同的行为，模拟更复杂的动力学系统
class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):"前向传播过程
        self.nfe += 1"NFE计数，可用于调试和性能分析，NFE反应了求解的难度（值越大说明ode越僵硬）
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out"返回导数，也即是特征的变化率，求解器会用这个导数来更新状态
"定义了一个neural ode的核心动力学函数

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc"保存动力学函数
        self.integration_time = torch.tensor([0, 1]).float()"积分时间区间

    def forward(self, x):"前向传播
        self.integration_time = self.integration_time.type_as(x)"确保 integration_time 和输入 x 在同一设备上（CPU/GPU）
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]"t=1时候的状态，也就是演化后的状态

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
"odeblock是neural ode的核心封装层，把ode求解器odeint与动力学函数odefunc组合起来，形成一个可以像普通神经网络层一样可以使用的模块
"其作用是把特征从初始状态通过连续时间演化，变换到最终状态（数学上等价于欧拉公式）
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()"torch.prod是计算所有元素的乘积：每个样本的总特征数
        return x.view(-1, shape)"-1表示这个维度自动计算，只要总元素不变，pytorch会根据总元素推断出batch size
"Flatten 是一个 展平层，用于把多维特征图（如 4D 的卷积输出）压缩成 2D 矩阵，方便后续的全连接层处理。

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

"一个指数移动平均计算器，用于平滑地追踪训练过程中的指标(如损失，NFE，训练时间等):这样可以平滑噪声，让你看到更稳定的趋势，而不是剧烈波动的原始数据。
def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader
"get_mnist_loaders 是一个 数据加载器工厂函数，负责准备 MNIST 数据集并返回三个 DataLoader：train_loader,test_loader,train_eval_loader

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
"无限数据迭代器：把有限的 DataLoader 变成无限循环的迭代器，避免手动重启 epoch。

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn
"创建一个分段常数学习率调度器，在训练的不同阶段使用不同的学习率。

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
"把类别标签转换成 one-hot 向量（独热编码）。

def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)
"计算模型在整个数据集上的分类准确率。

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
"统计可训练参数数量

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
"安全创建目录（已存在则跳过）

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger
"日志记录器:同时输出到控制台和文件，记录当前代码内容（方便复现实验）

if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)"记录日志

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')"选择设备

    is_odenet = args.network == 'odenet'"判断网络类型

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]"卷积下采样层
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]"残差下采样层

    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]"特征提取层
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]"分类层

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)"组装模型

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))"打印模型信息

    criterion = nn.CrossEntropyLoss().to(device)"计算分类任务的交叉熵损失

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )"数据加载器

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
"无限迭代器
    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )"学习率调度器

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)"参数优化器

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
"初始化统计
    for itr in range(args.nepochs * batches_per_epoch):"循环总次数

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)
"动态调整学习率
        optimizer.zero_grad()"清空梯度
        x, y = data_gen.__next__()"获取下一个batch
        x = x.to(device)
        y = y.to(device)"x图片，y标签移到GPU
        logits = model(x)"前向传播
        loss = criterion(logits, y)"计算损失
"前向传播
        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0
"读取前向NFE
        loss.backward()"反向传播：计算梯度
        optimizer.step()"更新参数

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0
"记录反向NFE
        batch_time_meter.update(time.time() - end)"更新训练时间
        if is_odenet:
            f_nfe_meter.update(nfe_forward)"更新前向NFE平均值
            b_nfe_meter.update(nfe_backward)"更新反向NFE平均值
        end = time.time()"重置计时

        if itr % batches_per_epoch == 0:"每个epoch结束进行一次评估
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)"评估准确率
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc"看准确率保存最佳模型
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc"记录日志
                    )
                )

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')"选择ode求解器
parser.add_argument('--data_size', type=int, default=1000)"数据集的大小：生成多少个时间点
parser.add_argument('--batch_time', type=int, default=10)"时间批次大小：每次训练采样多少个时间点
parser.add_argument('--batch_size', type=int, default=20)"批大小：每次训练采用多少个样本的初始状态(这里也就是说完成一次训练要20个时间点的初始状态--这20个批次中每批又有10个采样时间点)
parser.add_argument('--niters', type=int, default=2000)"训练迭代次数：总共训练多少轮（梯度更新多少次）
parser.add_argument('--test_freq', type=int, default=20)"测试频率：每训练多少轮评估一次模型/打印一次日志
parser.add_argument('--viz', action='store_true')"可视化开关
parser.add_argument('--gpu', type=int, default=0)"gpu编号：使用哪块 GPU（如果有多张显卡）
parser.add_argument('--adjoint', action='store_true')"是否使用伴随灵敏度法
args = parser.parse_args()"在终端读取参数

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint"统一ode求解器参数命名
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')"检查是否有可用gpu

true_y0 = torch.tensor([[2., 0.]]).to(device)"初始状态：这里起始点是x=2，y=0,但后面还是多了一个中括号其实是加上了一个batch维度，因为后续训练要用到多个批次作为起始点
t = torch.linspace(0., 25., args.data_size).to(device)"时间序列点：生成从 0 到 25 秒的均匀时间点，生成1000个
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)"真实的动力系统矩阵：定义一个线性动力学系统的系数矩阵（dx/dt = -0.1*x + 2.0*y
dy/dt = -2.0*x - 0.1*y）


class Lambda(nn.Module):"定义一个 PyTorch 模块类

    def forward(self, t, y):"定义动力学函数 dy/dt = f(t, y)
        return torch.mm(y**3, true_A)"计算状态的导数，torch.mm是矩阵乘法，生成的动力系统：dx/dt = -0.1*x³ + 2.0*y³
dy/dt = -2.0*x³ - 0.1*y³


with torch.no_grad():"关闭自动求导，节省内存
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')"根据dopri5法生成训练数据（真实轨迹）


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  " (M, D)
    batch_t = t[:args.batch_time]  " (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  " (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
"这里的s是指从全部1000个时间点数据集中随机选取20个时间点作为初始点，再从这些初始点往后按照顺序采样10个点作为一个批次，总共20个批次，这也是为什么最大初始点的取值不能超过data_size-batch_time，不然就有可能取样到后面会取到1001，1002...等
"这里的batch_y0是根据随机索引 s，从完整轨迹 true_y 中提取对应时刻的状态作为初始状态
"batch_t就是直接取1000个时间点中的前10个点，然后batch_y是取得真实时间点对应的系统状态，再把这些相对时间和真实状态对应起来其实就是把原来的1000个时间点数据截成一段一段的每段的初始时间换成0



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)"创建目录（文件夹）


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)"轨迹图：随时间变化的状态值
    ax_phase = fig.add_subplot(132, frameon=False)"相位图：2D 平面上的轨迹（螺旋线），也即是轨线压缩时间轴后投影到一个坐标面
    ax_vecfield = fig.add_subplot(133, frameon=False)"向量场：动力学系统的方向场
    plt.show(block=False)
"初始化可视化界面——创建一个窗口，包含 3 个子图，用于实时显示训练过程

def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()"清空上一次绘图的内容
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')"真实轨迹的 x 和 y 分量，因为系统其实是有两个分量的,要画出每个分量是如何随时间变化的
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')"预测轨迹的 x 和 y 分量
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()
" 原始数据
"t.shape           " (1000,)        时间点
"true_y.shape      " (1000, 1, 2)   轨迹数据
 
" 索引操作
"true_y[:, 0, 0]
"      │  │  └─ 取第 0 个维度（x 分量）
"      │  └──── 取第 0 个样本（只有 1 个轨迹）
"      └─────── 所有时间步
 
" 结果
"true_y[:, 0, 0].shape  " (1000,)

        
        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')"显然这两个相位图对比前面的轨迹图都少了时间这个维度
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]"生成网格点： 创建一个 21×21 的网格，覆盖 x∈[-2,2], y∈[-2,2]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)"归一化向量：streamplot 只需要方向，不需要真实速度大小，带了大小还不好看变化趋势
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")"绘制流线图：这个图直观展示神经网络学到的"力场"
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()"调用父类 nn.Module 的 __init__ 方法，完成神经网络模块的初始化

        self.net = nn.Sequential(
            nn.Linear(2, 50),"输入层：输入2维，中间隐藏层50维
            nn.Tanh(),"激活函数：选取Tanh是因为其光滑性
            nn.Linear(50, 2),"输出层：输出二维
        )
"网络结构
        for m in self.net.modules():"遍历所有模块进行初始化
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)"用正态分布初始化权重矩阵
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):"前向传播
        return self.net(y**3)"实际上是减轻了学习难度，直接给出了一个物理先验，相当于只用学习对应系数了


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
"平滑指标参数：平滑后的曲线更容易看出是否在收敛

if __name__ == '__main__':

    ii = 0"可视化计数器

    func = ODEFunc().to(device)"创建神经网络
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)"创建参数优化器：这里选择RMSpro(自适应学习率，适合非平稳目标)
    end = time.time()"记录当前时间（用于计算每次迭代耗时）

    time_meter = RunningAverageMeter(0.97)"平滑每次迭代的时间
    
    loss_meter = RunningAverageMeter(0.97)" 平滑训练 loss

    for itr in range(1, args.niters + 1):"循环每个训练过程：每次迭代都是一个完整的训练步骤
        optimizer.zero_grad()"清空梯度：如果不清空，会把上一次的梯度加到这一次上
        batch_y0, batch_t, batch_y = get_batch)"获取训练批次
        pred_y = odeint(func, batch_y0, batch_t).to(device)"前向传播ode求解
        loss = torch.mean(torch.abs(pred_y - batch_y))"计算损失
        loss.backward()"反向传播，优化loss
        optimizer.step()"更新参数

        time_meter.update(time.time() - end)"计算本次训练耗时
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:"默认每 20 次迭代测试一次
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()"重置计时器，准备下次迭代

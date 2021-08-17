
import torch
from torch.nn import Parameter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def draw_function(ax3, function, size, gap=0.5):
    x_ = np.arange(*size, gap)
    y_ = np.arange(*size, gap)
    x, y = np.meshgrid(x_, y_)
    z = function(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    ax3.plot_wireframe(x, y, z, rstride=1, cstride=1, linewidth=0.5)
    return ax3

def compute_SGD(st, loss_fn, step=10, lr=0.5):
    x, y = torch.from_numpy(np.array(st))
    x, y = Parameter(x.float()), Parameter(y.float())
    loss = loss_fn(x, y)
    points_list = [np.array([x.data.numpy(), y.data.numpy(), loss.data.numpy()])]
    for i in range(step):
        grad = torch.autograd.grad(loss, (x, y))
        x = x - lr * grad[0]
        y = y - lr * grad[1]
        loss = loss_fn(x, y)
        points_list.append(np.array([x.data.numpy(), y.data.numpy(), loss.data.numpy()]))
    return points_list


def compute_cos_sim(a, b):
    aa = np.sqrt((a ** 2).sum())
    bb = np.sqrt((b ** 2).sum())
    ab = (a * b).sum()
    return ab / (aa * bb)


def compute_o2SGD(st, loss_fn, step=10, lr=0.5, beta=0.01):
    x, y = torch.from_numpy(np.array(st))
    x, y = Parameter(x.float()), Parameter(y.float())
    e_loss = -1.5
    loss = loss_fn(x, y)

    points_list = [np.array([x.data.numpy(), y.data.numpy(), loss.data.numpy()])]

    for i in range(step):
        grad = torch.autograd.grad(loss, (x, y))
        x_ = Parameter((x - beta * grad[0]).data)
        y_ = Parameter((y - beta * grad[1]).data)
        loss_ = loss_fn(x_, y_)
        grad_ = torch.autograd.grad(loss_, (x_, y_))

        rgrad = (-torch.sin(x) * torch.cos(x), -torch.sin(y) * torch.cos(y))
        print(grad, (torch.Tensor(grad_) - torch.Tensor(grad)) / beta,  rgrad, compute_cos_sim(np.array([*grad]), np.array([*grad_])))

        mu = 2 * (loss - e_loss)
        print(loss - e_loss, mu)
        x = x - lr * (grad[0] + mu * (grad_[0] - grad[0]) / beta)
        y = y - lr * (grad[1] + mu * (grad_[1] - grad[1]) / beta)
        # x = x - lr * grad_[0]
        # y = y - lr * grad_[1]
        # print(grad, grad_,  compute_cos_sim(np.array([*grad]), np.array([*grad_])))

        loss = loss_fn(x, y)
        points_list.append(np.array([x.data.numpy(), y.data.numpy(), loss.data.numpy()]))
    return points_list


def draw_lines(ax3, lines, color='red'):
    x, y, z = np.array(lines).T
    ax3.scatter3D(x, y, z, color=color)
    ax3.plot3D(x, y, z, color)


if __name__ == '__main__':

    function = lambda x, y: torch.sin(x) + torch.sin(y)

    # function = lambda x, y: (1 - x / 2 + x ** 5 + y ** 3) * torch.exp(-x ** 2 - y ** 2)
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    draw_function(ax3, function, (-2.5, 2.5))
    st =  [0, 1.36] #[-0.936715  , -0.26373297]#
    SGD_lines = compute_SGD(st, function, 10)
    print(SGD_lines)
    o2_lines = compute_o2SGD(st, function, 10)
    print(o2_lines)
    draw_lines(ax3, SGD_lines)
    draw_lines(ax3, o2_lines, 'green')
    plt.show()




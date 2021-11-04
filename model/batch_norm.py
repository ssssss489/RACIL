
import torch
import torch.nn as nn
from utils import *

class torchBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, device=None, dtype=None):
        super(torchBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum)

    def forward(self, x, t):
        return self.bn(x)



class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, device=None, dtype=None):
        super(BatchNorm2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = torch.nn.Parameter(torch.zeros(num_features, **factory_kwargs), requires_grad=False)
        self.running_var = torch.nn.Parameter(torch.ones(num_features, **factory_kwargs), requires_grad=False)

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def formula(self, cx, mu, var, diff=None):
        if diff is None:
            diff = cx - mu.view(self.num_features, 1, 1, 1)
        norm = diff / (var.sqrt().view(self.num_features, 1, 1, 1) + self.eps)
        if self.affine:
            norm = norm * self.gamma.view(self.num_features, 1, 1, 1) + self.beta.view(self.num_features, 1, 1, 1)
        return norm.transpose(0, 1)


    def forward(self, x, t=None):
        if self.training:
            cx = x.transpose(0, 1)
            mu_hw = cx.mean([2, 3])  # c * b
            mu = mu_hw.mean(-1)
            diff = cx - mu.view(self.num_features, 1, 1, 1)
            var = diff.square().mean([1, 2, 3])
            re = self.formula(cx, mu, var, diff)
            self.running_mean.data = self.running_mean * (1 - self.momentum) + mu * self.momentum
            self.running_var.data = self.running_var * (1 - self.momentum) + var * self.momentum
            return re
        else:
            cx = x.transpose(0, 1)
            mu = self.running_mean
            var = self.running_var
            re = self.formula(cx, mu, var)
            return re

class taskBatchNorm2d(BatchNorm2d):
    def __init__(self, num_features, num_tasks, eps=1e-5, momentum=0.1, affine=False, device=None, dtype=None):
        super(taskBatchNorm2d, self).__init__(num_features, eps, momentum, affine, )
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.num_tasks = num_tasks
        self.task_prior_mean = torch.nn.Parameter(torch.zeros([num_tasks, num_features], **factory_kwargs), requires_grad=False)
        self.task_prior_precision = 1.0
        self.obverse_tasks = set()

    def update_prior_mean(self, posterior_mean, task_p):
        self.task_prior_mean[task_p].data.copy_(self.task_prior_mean[task_p] * (1 - self.momentum) + posterior_mean * self.momentum)

    def compute_posterior_mean(self, task_p, x):
        n = x.shape[0]
        cx = x.transpose(0, 1)
        data = cx.mean([2, 3]) # c b
        rand_idx = np.random.choice(np.arange(data.shape[1]), size=int(data.shape[1] / 2))
        data = data[:,rand_idx]
        data_mean = data.mean(-1)
        # data_var = data.var(-1)
        known_precision = 1

        prior_precision = self.task_prior_precision
        posterior_precision = prior_precision + n * known_precision
        weight = n * known_precision / posterior_precision
        prior_mean = self.task_prior_mean[task_p]
        weight = 1.0
        posterior_mean = weight * data_mean + (1 - weight) * prior_mean
        return posterior_mean
        # random_mean = torch.randn(self.num_features).cuda() / np.sqrt(posterior_precision) + posterior_mean
        # return random_mean


    def forward(self, x, t):
        if self.training:
            tset = set(to_numpy(t))
            self.obverse_tasks.update(tset)
            means = []
            current_task = max(tset)
            for task_p in tset:
                task_x = x[t == task_p]
                mean = self.compute_posterior_mean(task_p, task_x)
                self.update_prior_mean(mean, task_p)
                means.append(mean)
            mu = torch.stack(means, 0).mean(0)
            cx = x.transpose(0, 1)
            diff = cx - mu.view(self.num_features, 1, 1, 1)
            var = diff.square().mean([1, 2, 3])
            re = self.formula(cx, mu, var, diff)
            if np.random.random() > 0.0:
                re += torch.randn(re.shape).cuda() * 0.05
            if self.affine:
                re = self.affine_fn(re)
            self.running_var.data = self.running_var * (1 - self.momentum) + var * self.momentum
            self.running_mean.data = self.running_mean * (1 - self.momentum) + mu * self.momentum
            return re
        else:
            cx = x.transpose(0, 1)
            mu = self.task_prior_mean[:len(self.obverse_tasks)].mean(0)
            var = self.running_var
            re = self.formula(cx, mu, var)
            if self.affine:
                re = self.affine_fn(re)
            return re


class vatBatchNorm2d(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, device=None, dtype=None):
        super(vatBatchNorm2d, self).__init__(num_features, eps, momentum, affine, )
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.noise = None

    def formula(self, cx, mu, var, diff=None):
        if diff is None:
            diff = cx - mu.view(self.num_features, 1, 1, 1)
        norm = diff / (var.sqrt().view(self.num_features, 1, 1, 1) + self.eps)
        if self.training:
            if self.noise is None:
                self.noise = torch.zeros_like(norm, requires_grad=True).cuda()
                norm += self.noise
            else:
                norm += self.noise.grad.data * 1e-1
                self.noise = None
        if self.affine:
            norm = norm * self.gamma.view(self.num_features, 1, 1, 1) + self.beta.view(self.num_features, 1, 1, 1)
        return norm.transpose(0, 1)


class noiseBatchNorm2d(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, device=None, dtype=None):
        super(noiseBatchNorm2d, self).__init__(num_features, eps, momentum, affine, )

    def formula(self, cx, mu, var, diff=None):
        if diff is None:
            diff = cx - mu.view(self.num_features, 1, 1, 1)
        if self.training:
            std = var.sqrt().view(self.num_features, 1, 1, 1)
            norm = diff / (std + self.eps) + torch.randn([cx.shape[0], 1, cx.shape[2], cx.shape[3]]).cuda() * 0.2# uCIR prefer 0.05, other 0.1
        else:
            norm = diff / (var.sqrt().view(self.num_features, 1, 1, 1) + self.eps)
        if self.affine:
            norm = norm * self.gamma.view(self.num_features, 1, 1, 1) + self.beta.view(self.num_features, 1, 1, 1)
        return norm.transpose(0, 1)

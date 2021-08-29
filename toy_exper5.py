



import os
import logging
import torch
import argparse
from dataset import task_data_loader, mkdir
from model.base_model import MLP, ResNet18
from train.base_train import base_train
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from copy import deepcopy
from model.ER import ER
from model.buffer import Distill_buffer
from model.buffer import Buffer
from utils import *
from model.base_model import *


def gradient_project(param_dicts, ref_param_dicts):
    mul, mod = 0, 0
    for p in param_dicts:
        grad = param_dicts[p].grad.data
        ref_grad = ref_param_dicts[p]
        mul += torch.sum(grad * ref_grad)
        mod += torch.sum(ref_grad * ref_grad)
    if mul < 0:
        l = mul / mod
        for p in param_dicts:
            grad = param_dicts[p].grad
            ref_grad = ref_param_dicts[p]
            grad -= l * ref_grad





class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class model(torch.nn.Module):
    def __init__(self, data, args):
        super(model, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.n_classes = models_setting[data].classifier[-1]
        self.feature_net = MLP_Feature_Extractor(models_setting[data].feature_extractor)
        self.common_feature_net = MLP_Feature_Extractor(models_setting[data].feature_extractor)
        self.GRL = GRL(1.0)
        self.buffer = Buffer(args.n_task, args.n_memory, self.n_classes)
        self.reverse_classifier = MLP_Classifier(models_setting[data].classifier)
        self.classifier = MLP_Classifier(models_setting[data].classifier)
        if args.cuda:
            self.cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = args.lr
        self.feature_net_optimizer = torch.optim.SGD(self.feature_net.parameters(), lr=args.lr)
        self.common_feature_net_optimizer = torch.optim.SGD(self.common_feature_net.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.reverse_classifier_optimizer = torch.optim.SGD(self.reverse_classifier.parameters(), lr=args.lr)
        self.feature_net_fine_optimizer = torch.optim.SGD(self.feature_net.parameters(), lr=args.lr * 0.1)
        self.classifier_fine_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr * 0.1)
        self.observed_tasks = []
        self.observed_task_common_feature_net = {}

        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")

    def forward(self, x):
        feature = self.feature_net(x)
        common_feature = self.common_feature_net(x)
        l = torch.sum(common_feature * feature, dim=1) / torch.sum(common_feature * common_feature, dim=1)
        class_feature = feature - common_feature * l.view([-1,1])
        class_feature = class_feature - class_feature.mean(dim=-1).view([-1,1])
        std = class_feature.std(dim=-1)
        class_feature = class_feature / class_feature.std(dim=-1).view([-1,1])
        y = self.classifier(class_feature)
        return y

    def reverse_forward(self, x):
        common_feature = self.common_feature_net(x)
        reverse_feature = self.GRL(common_feature)
        # reverse_feature = reverse_feature - reverse_feature.mean(dim=-1).view([-1,1])
        # reverse_feature = reverse_feature / reverse_feature.std(dim=-1).view([-1,1])
        # reverese_logits = self.classifier(reverse_feature)
        reverese_logits = self.classifier(common_feature)
        return reverese_logits


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            self.common_feature_net = MLP_Feature_Extractor(models_setting['mnist'].feature_extractor).cuda()
            self.observed_task_common_feature_net[task_p] = self.common_feature_net
            self.common_feature_net_optimizer = torch.optim.SGD(self.common_feature_net.parameters(), lr=0.1)

        if task_p > 0:
            # pt = np.random.choice(self.observed_tasks[:-1])
            pt_inputs, pt_labels, pts = self.buffer.get_buffer_samples(self.observed_tasks[:-1], labels.shape[0])
            pt_class_offset = get_class_offset(pts)
            pt_logits = self.forward(pt_inputs)
            loss_ = self.loss_fn(*compute_output_offset(pt_logits, pt_labels, pt_class_offset))
            # loss += 1.0 * loss_
            loss_.backward()
            ref_classifier_param_dicts = {n: p.grad.data.clone()for n, p in self.classifier.named_parameters()}
            ref_feature_net_param_dicts = {n: p.grad.data.clone()for n, p in self.feature_net.named_parameters()}
            # self.zero_grad()
            # loss += loss_

        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, class_offset)
        loss = self.loss_fn(logits, labels)

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        # re_logits = self.reverse_forward(inputs)
        # re_loss = self.loss_fn(re_logits, labels)
        # # loss += re_loss * 0.1
        # re_argmax = torch.argmax(re_logits, dim=1)

        loss.backward()

        # if task_p > 0:
        #     gradient_project(dict(self.classifier.named_parameters()), ref_classifier_param_dicts)
        #     gradient_project(dict(self.feature_net.named_parameters()), ref_feature_net_param_dicts)

        if task_p == 0:
            self.classifier_optimizer.step()
            self.feature_net_optimizer.step()
            self.common_feature_net_optimizer.step()
        else:
            self.classifier_fine_optimizer.step()
            # self.feature_net_fine_optimizer.step()
            # self.classifier_optimizer.step()
            # self.feature_net_optimizer.step()
            self.common_feature_net_optimizer.step()
        self.reverse_classifier_optimizer.step()

        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()
        return float(loss.item()), float(acc.item())


    def predict(self, inputs, class_offset=None):
        self.eval()
        self.zero_grad()
        logits = self.forward(inputs)
        if class_offset:
            logits = logits[:, class_offset[0]: class_offset[1]]
            predicts = torch.argmax(logits, dim=1) + class_offset[0]
        else:
            predicts = torch.argmax(logits, dim=1)
        return predicts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--result_path', default='result/', help='input directory')
    parser.add_argument('--dataset', default='mnist_rot_10', help='learn task')
    parser.add_argument('--n_task', default=5, type=int, help='number of tasks')
    parser.add_argument('--samples_per_task', type=int, default=-1, help='training samples per task (all if negative)')
    parser.add_argument('--seed', default=5, type=int, help='random seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_epoch', default=1, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU')
    parser.add_argument('--n_memory', default=100, help='number of memories per task')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))

    task = args.dataset.split('_')[0]
    # model = MLP(task, args)
    model = model(task, args)

    data_loader = task_data_loader(task_datasets, args)

    trainer = base_train(data_loader, model, args, logger)
    trainer.train()

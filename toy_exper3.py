

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
from utils import *


class distill_train(base_train):
    def __init__(self, data_loader, model, args, logger):
        super(distill_train, self).__init__(data_loader, model, args, logger)

    def train(self):
        total = self.data_loader.task_n_sample[0]
        losses, train_acc = [], []
        task_p = 0
        epoch = 0
        bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
        self.class_offset = self.get_class_offset(task_p)
        self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(0, epoch=epoch)):
            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            param_dict = dict(self.model.model.named_parameters())
            loss, acc = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
            losses.append(loss)
            train_acc.append(acc)
            bar.update(batch_labels.size(0))
        bar.close()
        print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
        self.metric(task_p)
        self.logger.info(f'task {task_p} has trained over ,eval result is shown below.')

        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(1, epoch=epoch)):
            tar_model1 = self.model.model
            tar_model2 = deepcopy(self.model.model)
            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            tar_model2.train_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
            object_param_grad = [p.grad.data.clone() for p in tar_model2.parameters()]

            buffer_inputs, buffer_outputs = self.model.buffer.get_buffer_samples([0], 100, False)
            # tar_model2.train_step(buffer_inputs, buffer_outputs, self.get_class_offset, task_p)
            # tar_1_param_grad = tar_model1.compute_grad(buffer_inputs, buffer_outputs, self.get_class_offset, task_p)
            # print(compute_cos_similar(tar_1_param_grad, object_param_grad))
            # print(compute_euclid_similar(tar_1_param_grad, object_param_grad))

            def show_similar():
                tar_1_param_grad = tar_model1.compute_grad(self.model.buffer.normal_inputs,
                                                           buffer_outputs, self.get_class_offset, task_p)
                print(compute_cos_similar(tar_1_param_grad, object_param_grad))
                # print(compute_euclid_similar(tar_1_param_grad, object_param_grad))
            for k in range(30):
                show_similar()

                self.model.buffer.train_augment(task_p, tar_model1, tar_model2)

            break






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--result_path', default='result/', help='input directory')
    parser.add_argument('--dataset', default='mnist_perm_10', help='learn task')
    parser.add_argument('--n_task', default=2, type=int, help='number of tasks')
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
    model = ER(MLP, task, args, Distill_buffer)

    data_loader = task_data_loader(task_datasets, args)

    trainer = distill_train(data_loader, model, args, logger)
    trainer.train()

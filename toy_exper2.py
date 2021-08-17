
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

class vison_train(base_train):
    def __init__(self, data_loader, model, args, logger):
        super(vison_train, self).__init__(data_loader, model, args, logger)
        self.vision_param = defaultdict(lambda: defaultdict(list))
        self.vision_loss = defaultdict(lambda: defaultdict(list))
        self.task_parameters = []
        self.over_train = False

    def train(self):
        last_task_p = -1
        last_epoch = -1
        bar = None
        total = self.data_loader.task_n_sample[0]
        losses, train_acc = [], []
        newtask_flag = True
        for i, (task_p, epoch, batch_inputs, batch_labels) in enumerate(self.data_loader):
            if last_epoch != epoch or task_p != last_task_p:
                if bar:
                    bar.close()
                print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
                if epoch != 0:
                    self.metric(last_task_p)
                    if last_task_p != -1:
                        if epoch == self.n_epoch - 1 and epoch > 0:
                            self.task_parameters.append(deepcopy(dict(self.model.state_dict())))
                            self.over_train = True
                            print('    store_parameters')
                else:
                    self.metric(last_task_p)
                    self.logger.info(f'task {last_task_p} has trained over ,eval result is shown below.')
                    self.class_offset = self.get_class_offset(task_p)
                    self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
                    last_task_p = task_p
                    total = self.data_loader.task_n_sample[task_p]
                    if last_task_p != 0 and len(self.task_parameters) > 0 :
                        self.model.load_state_dict(self.task_parameters[-1])
                        print('    reload_parameters')
                        self.model.over_train = False
                bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
                losses, train_acc = [], []
                last_epoch = epoch

            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            param_dict = dict(self.model.model.named_parameters())
            self.vision_param[task_p][epoch].append(param_dict['model.3.bias'].cpu().data.numpy())
            loss, acc = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
            self.vision_loss[task_p][epoch].append(loss)
            losses.append(loss)
            train_acc.append(acc)
            bar.update(batch_labels.size(0))

        if bar:
            bar.close()
        print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
        self.metric(last_task_p)
        self.logger.info(f'task {last_task_p} has trained over ,eval result is shown below.')

    def draw_tSNE(self):
        fit_data = []
        loss_data = []
        color_map = {0: 'r', 1: 'b', 2: 'g'}
        gap = 10
        tsne = TSNE(n_components=2, init='pca', random_state=0)

        for task in self.vision_param.keys():
            for epoch in self.vision_param[task].keys():
                fit_data += self.vision_param[task][epoch][::gap]
                loss_data += self.vision_loss[task][epoch][::gap]
        trans_data = tsne.fit_transform(np.stack(fit_data))
        # trans_data = np.stack(fit_data)
        start = 0
        for task in self.vision_param.keys():
            for epoch in self.vision_param[task].keys():
                end = start + len(self.vision_param[task][epoch][::gap])
                data = trans_data[start:end]
                start = end
                plt.plot(data[:, 0], data[:, 1], color_map[task])

        plt.scatter(trans_data[::1, 0], trans_data[::1, 1], c=loss_data[::1], cmap='rainbow')
        plt.colorbar()
        plt.show()
        return



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
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU')
    parser.add_argument('--n_memory', default=100, help='number of memories per task')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))

    task = args.dataset.split('_')[0]
    # model = MLP(task, args)
    model = ER(MLP, task, args)

    data_loader = task_data_loader(task_datasets, args)
    trainer = vison_train(data_loader, model, args, logger)
    trainer.train()
    trainer.draw_tSNE()








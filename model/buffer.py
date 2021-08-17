
from collections import defaultdict
import torch
import numpy as np
from easydict import EasyDict
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable


class Buffer(object):
    def __init__(self, n_task, n_memory, n_classes):
        self.memory_inputs = defaultdict(torch.FloatTensor)
        self.memory_labels = defaultdict(torch.LongTensor)

        self.n_memory = n_memory
        self.n_task = n_task
        self.n_classes = n_classes


    def save_buffer_samples(self, inputs, labels, p_task):
        task_memory_inputs = self.memory_inputs[p_task]
        task_memory_labels = self.memory_labels[p_task]
        task_memory_labels = torch.cat((task_memory_labels, labels.cpu()))
        task_memory_inputs = torch.cat((task_memory_inputs, inputs.cpu()))
        class_idx = defaultdict(list)
        for idx, label in enumerate(task_memory_labels.data):
            class_idx[int(label.numpy())].append(idx)
        n_each_class_samples = int(self.n_memory / self.n_classes)
        indices = set()
        for c, idx in class_idx.items():
            indices.update(np.random.choice(idx, min(n_each_class_samples, len(idx)), replace=False))
        while len(indices) < min(self.n_memory, task_memory_labels.shape[0]):
            indices.add(np.random.randint(0, task_memory_labels.shape[0]))

        indices = np.array(list(indices))
        self.memory_inputs[p_task] = task_memory_inputs[indices]
        self.memory_labels[p_task] = task_memory_labels[indices]


    def get_buffer_samples(self, list_p_task, size):
        inputs, labels = [], []
        for p_task in list_p_task:
            task_memory_inputs = self.memory_inputs[p_task]
            task_memory_labels = self.memory_labels[p_task]
            idx = torch.from_numpy(np.random.choice(np.arange(self.n_memory), size, replace=False))
            inputs.append(task_memory_inputs[idx])
            labels.append(task_memory_labels[idx])
        return torch.cat(inputs).cuda(), torch.cat(labels).cuda()

class Distill_buffer(Buffer, nn.Module):
    def __init__(self, n_task, n_memory, n_classes):
        super(Distill_buffer, self).__init__(n_task, n_memory, n_classes)
        self.cuda = True
        self.normal_inputs = Variable(torch.zeros([self.n_memory, 28, 28]).cuda(), requires_grad=True)
        self.optimizer = torch.optim.SGD([self.normal_inputs], lr=1)
        # self.augment = defaultdict(lambda:self.augment_net())
        # for i in range(n_task):
        #     print(f'    init task {i} augment parameters')
        #     for name, param in self.augment[0].net.named_parameters():
        #         print(f"    Layer: {name} | Size: {param.size()} ")


    def augment_net(self):
        net = nn.Sequential(nn.Linear(32, 512),
                            nn.ReLU(),
                            nn.Linear(512, 28 * 28),
                            nn.ReLU())
        if self.cuda:
            net.cuda()
        optimizer = torch.optim.SGD(self.normal_inputs, lr=1)
        return EasyDict(net=net, optimizer=optimizer)

    def train_augment(self, p_task, tar_model1, tar_model2):
        tar_model1.zero_grad()
        tar_model2.zero_grad()
        # augment = self.augment[p_task]
        # augment.net.train()
        # augment.net.zero_grad()
        memory_inputs = self.memory_inputs[p_task].cuda()
        memory_labels = self.memory_labels[p_task].cuda()
        loss_fn = torch.nn.CrossEntropyLoss()
        # outputs = augment.net(self.normal_inputs).reshape([-1,28,28])
        outputs = self.normal_inputs
        outputs.grad = None
        loss_1 = loss_fn(tar_model1(outputs), memory_labels)
        loss_2 = loss_fn(tar_model2(outputs), memory_labels)
        loss = loss_2 - loss_1
        loss.backward()
        print(f'loss={loss.data}')
        self.optimizer.step()

    def get_buffer_samples(self, list_p_task, size, use_augment=True):
        if not use_augment:
            return super(Distill_buffer, self).get_buffer_samples(list_p_task, size)
        else:
            inputs, labels = [], []
            for p_task in list_p_task:
                task_memory_inputs = self.memory_inputs[p_task]
                task_memory_labels = self.memory_labels[p_task]
                idx = torch.from_numpy(np.random.choice(np.arange(self.n_memory), size, replace=False))
                inputs.append(task_memory_inputs[idx] + self.normal_inputs[:size])
                labels.append(task_memory_labels[idx])
            return torch.cat(inputs).cuda(), torch.cat(labels).cuda()






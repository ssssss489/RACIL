
from collections import defaultdict
import torch
import numpy as np
from easydict import EasyDict
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable
from utils import *


def random_retrieve(buffer, n_retrieve, excl_indices=None):
    filled_indices = np.arange(buffer.current_index)
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    n_retrieve = min(n_retrieve, valid_indices.shape[0])
    indices = torch.from_numpy(np.random.choice(valid_indices, n_retrieve, replace=False)).long()

    x = buffer.memory_inputs[indices]
    y = buffer.memory_labels[indices]
    return x.cuda(), y.cuda().long()

def random_update(buffer, x, y):
    bs = x.size(0)
    place_left = max(0, buffer.n_memories - buffer.current_index)
    if place_left:
        offset = min(place_left, bs)
        buffer.memory_inputs[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
        buffer.memory_labels[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])
        buffer.current_index += offset
        buffer.n_seen_so_far += offset

        if offset == bs:
            filled_idx = list(range(buffer.current_index - offset, buffer.current_index, ))
            return filled_idx

    x, y = x[place_left:], y[place_left:]
    indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()
    valid_indices = (indices < buffer.n_memories).long()

    idx_new_data = valid_indices.nonzero().squeeze(-1)
    idx_buffer = indices[idx_new_data]
    buffer.n_seen_so_far += x.size(0)
    if idx_buffer.numel() == 0:
        return []

    idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

    buffer.memory_inputs[list(idx_map.keys())] = x[list(idx_map.values())].cpu()
    buffer.memory_labels[list(idx_map.keys())] = y[list(idx_map.values())].cpu()
    return list(idx_map.keys())



class Buffer(object):
    def __init__(self, n_tasks, n_memories, n_classes, input_dims):
        self.memory_inputs = torch.FloatTensor()
        self.memory_labels = torch.LongTensor()
        self.memory_tasks = torch.LongTensor()
        self.abandoned_class_inputs = defaultdict(list)

        self.n_memories = n_memories
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.current_index = 0
        self.n_seen_so_far = 0
        self.memory_keep_prob = torch.FloatTensor(n_memories).fill_(-1e10)
        self.class_n_samples = {}

    def shuffer(self):
        random_idx = np.arange(self.memory_inputs.shape[0])
        np.random.shuffle(random_idx)
        self.random_idx = random_idx






    def save_buffer_samples(self, inputs, labels, p_task):
        task_memory_inputs = self.memory_inputs[p_task]
        task_memory_labels = self.memory_labels[p_task]
        task_memory_labels = torch.cat((task_memory_labels, labels.cpu()))
        task_memory_inputs = torch.cat((task_memory_inputs, inputs.cpu()))
        class_idx = defaultdict(list)
        for idx, label in enumerate(task_memory_labels.data):
            class_idx[int(label.numpy())].append(idx)
        n_each_class_samples = int(self.n_memories / self.n_classes)
        indices = set()
        for c, idx in class_idx.items():
            indices.update(np.random.choice(idx, min(n_each_class_samples, len(idx)), replace=False))
        while len(indices) < min(self.n_memories, task_memory_labels.shape[0]):
            indices.add(np.random.randint(0, task_memory_labels.shape[0]))

        indices = np.array(list(indices))
        self.memory_inputs[p_task] = task_memory_inputs[indices]
        self.memory_labels[p_task] = task_memory_labels[indices]


    def get_buffer_samples(self, list_p_task, size):
        inputs, labels, pts = [], [], []
        pt_size = [int(size / len(list_p_task))] * len(list_p_task)
        pt_size[-1] = size - sum(pt_size[:-1])
        for p_task, s in zip(list_p_task, pt_size):
            task_memory_inputs = self.memory_inputs[p_task]
            task_memory_labels = self.memory_labels[p_task]
            idx = torch.from_numpy(np.random.choice(np.arange(self.n_memories), s, replace=False))
            inputs.append(task_memory_inputs[idx])
            labels.append(task_memory_labels[idx])
            pts += [p_task] * s
        if len(set(pts)) == 1:
            pts = pts[0]
        return torch.cat(inputs).cuda(), torch.cat(labels).cuda(), pts

class Distill_buffer(Buffer, nn.Module):
    def __init__(self, n_tasks, n_memories, n_classes):
        super(Distill_buffer, self).__init__(n_tasks, n_memories, n_classes)
        self.cuda = True
        self.normal_inputs = Variable(torch.zeros([self.n_memories, 28, 28]).cuda(), requires_grad=True)
        self.optimizer = torch.optim.SGD([self.normal_inputs], lr=1)
        # self.augment = defaultdict(lambda:self.augment_net())
        # for i in range(n_tasks):
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
                idx = torch.from_numpy(np.random.choice(np.arange(self.n_memories), size, replace=False))
                inputs.append(task_memory_inputs[idx] + self.normal_inputs[:size])
                labels.append(task_memory_labels[idx])
            return torch.cat(inputs).cuda(), torch.cat(labels).cuda()






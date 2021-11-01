

from collections import defaultdict
import torch
import numpy as np
from easydict import EasyDict
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable
from utils import *


def random_retrieve(buffer, n_retrieve):
    new_index = buffer.current_index + n_retrieve
    indices = np.arange(buffer.current_index, new_index)
    indices = indices % buffer.memory_labels.shape[0]
    buffer.current_index = new_index
    if 0 in indices:
        buffer.shuffer()
    indices = buffer.random_idx[indices]
    x = buffer.memory_inputs[indices]
    y = buffer.memory_labels[indices]
    t = buffer.memory_tasks[indices]
    return x.cuda(), y.cuda().long(), t.cuda().long()


def episodic_random_update(buffer, x, y, t):
    bs = x.size(0)
    x = x.cpu()
    y = y.cpu()
    if buffer.memory_labels.shape[0] < buffer.n_memories:
        place_left = buffer.n_memories - buffer.memory_labels.shape[0]
        buffer.memory_inputs = torch.cat([buffer.memory_inputs, x[:place_left]], dim=0)
        buffer.memory_labels = torch.cat([buffer.memory_labels, y[:place_left]], dim=0)
        buffer.memory_tasks = torch.cat([buffer.memory_tasks, torch.zeros_like(y[:place_left]).fill_(t)], dim=0)
        buffer.n_seen_so_far += y.shape[0]
    else:
        indices = np.random.randint(0, buffer.n_seen_so_far, bs)
        data_indices = np.nonzero(indices < buffer.n_memories)[0]
        mem_indices = indices[data_indices]

        buffer.memory_inputs[mem_indices] = x[data_indices]
        buffer.memory_labels[mem_indices] = y[data_indices]
        buffer.memory_tasks[mem_indices] = torch.zeros_like(y[data_indices]).fill_(t)
        buffer.n_seen_so_far += len(data_indices)

def class_average_update(buffer, x, y, t):
    x = x.cpu()
    y = y.cpu()
    class_samples_upper = int(buffer.n_memories / buffer.n_classes)
    current_classes = set(to_numpy(y))
    add_inputs = []
    add_labels = []
    add_tasks = []
    for t_cls in current_classes:
        if buffer.class_n_samples[t_cls] < class_samples_upper:
            t_cls_idx = torch.nonzero(y == t_cls).squeeze()[:class_samples_upper - buffer.class_n_samples[t_cls]]
            t_cls_x = x[t_cls_idx]
            t_cls_y = y[t_cls_idx]
            buffer.class_n_samples[t_cls] += len(t_cls_y)
            add_inputs.append(t_cls_x)
            add_labels.append(t_cls_y)
            add_tasks.append(torch.zeros_like(t_cls_y).fill_(t))
    if len(add_labels) != 0:
        buffer.memory_inputs = torch.cat(add_inputs + [buffer.memory_inputs], dim=0)
        buffer.memory_labels = torch.cat(add_labels + [buffer.memory_labels], dim=0)
        buffer.memory_tasks = torch.cat(add_tasks + [buffer.memory_tasks], dim=0)


def total_class_average_update(buffer, x, y, t):
    x = x.cpu()
    y = y.cpu()
    class_samples_upper = int(buffer.n_memories / ((t + 1) * 10)) #(buffer.n_classes / buffer.n_tasks)

    if t not in to_numpy(buffer.memory_tasks):
        mem_inputs = []
        mem_labels = []
        mem_tasks = []
        for m_cls in buffer.class_n_samples.keys():
            m_cls_idx = torch.nonzero(buffer.memory_labels == m_cls).squeeze()[:class_samples_upper]
            m_cls_x = buffer.memory_inputs[m_cls_idx]
            m_cls_y = buffer.memory_labels[m_cls_idx]
            m_cls_t = buffer.memory_tasks[m_cls_idx]
            mem_inputs.append(m_cls_x)
            mem_labels.append(m_cls_y)
            mem_tasks.append(m_cls_t)
        if len(mem_inputs) > 0:
            buffer.memory_inputs = torch.cat(mem_inputs, dim=0)
            buffer.memory_labels = torch.cat(mem_labels, dim=0)
            buffer.memory_tasks = torch.cat(mem_tasks, dim=0)

    current_classes = set(to_numpy(y))
    add_inputs = []
    add_labels = []
    add_tasks = []
    for t_cls in current_classes:
        if buffer.class_n_samples[t_cls] < class_samples_upper:
            t_cls_idx = torch.nonzero(y == t_cls).squeeze()
            if len(t_cls_idx.shape) > 0:
                t_cls_idx = t_cls_idx[:class_samples_upper - buffer.class_n_samples[t_cls]]
                t_cls_x = x[t_cls_idx]
                t_cls_y = y[t_cls_idx]
                buffer.class_n_samples[t_cls] += len(t_cls_y)
                add_inputs.append(t_cls_x)
                add_labels.append(t_cls_y)
                add_tasks.append(torch.zeros_like(t_cls_y).fill_(t))
    if len(add_labels) != 0:
        buffer.memory_inputs = torch.cat(add_inputs + [buffer.memory_inputs], dim=0)
        buffer.memory_labels = torch.cat(add_labels + [buffer.memory_labels], dim=0)
        buffer.memory_tasks = torch.cat(add_tasks + [buffer.memory_tasks], dim=0)







class Buffer(object):
    def __init__(self, n_tasks, n_memories, n_classes):
        self.memory_inputs = torch.FloatTensor()
        self.memory_labels = torch.LongTensor()
        self.memory_tasks = torch.LongTensor()

        self.n_memories = n_memories
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.current_index = 0
        self.n_seen_so_far = 0
        self.class_n_samples = defaultdict(lambda :0)

        self.abandoned_class_inputs = defaultdict(list)
        self.memory_keep_prob = torch.FloatTensor(n_memories).fill_(-1e10)


    def shuffer(self):
        random_idx = np.arange(self.memory_labels.shape[0])
        np.random.shuffle(random_idx)
        self.random_idx = random_idx







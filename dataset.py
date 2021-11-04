
import os
import torch
import random
import argparse
import logging
from torch.utils.data import Dataset
from torchvision import datasets, io
import torchvision.transforms as transforms
from easydict import EasyDict
import subprocess
import numpy as np
from collections import OrderedDict
import pandas as pd
from utils import *

class task_data_loader:
    def __init__(self, task_datasets, args):
        self.task_datasets = task_datasets[:args.n_tasks]
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.task_n_sample = []
        self.task_permutations = list(range(args.n_tasks))
        self.n_epochs = 100# args.n_epochs
        self.n_tasks = args.n_tasks
        sample_permutations = []

        for t in range(args.n_tasks):
            N = task_datasets[t].train_images.size(0)
            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)
            self.task_n_sample.append(n)
            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.sample_permutations = []

        for t in range(args.n_tasks):
            task_t = self.task_permutations[t]
            p = []
            for _ in range(self.n_epochs):
                task_p = sample_permutations[task_t].numpy()
                np.random.shuffle(task_p)
                p.append(torch.LongTensor(task_p))
            self.sample_permutations.append(torch.cat(p))

        self.c_task = 0
        self.c_sample = 0

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def get_batch(self, t_idx, epoch, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        task_p = self.sample_permutations[t_idx]
        for start in range(epoch * self.task_n_sample[t_idx], (epoch + 1) * self.task_n_sample[t_idx], batch_size):
            s_idx = task_p[start: start + batch_size]
            inputs = self.task_datasets[t_idx].train_images[s_idx]
            labels = self.task_datasets[t_idx].train_labels[s_idx]
            yield inputs, labels


    def get_inputs_with_labels(self, t_idx, tar_labels):
        inputs = self.task_datasets[t_idx].train_images
        labels = self.task_datasets[t_idx].train_labels
        s_idx = []
        for t in tar_labels:
            s_idx.append(np.random.choice(torch.arange(labels.shape[0])[labels == t]))

        return inputs.index_select(0, torch.IntTensor(s_idx)).cuda()


    def __next__(self):
        if self.c_task == self.n_tasks:
            raise StopIteration
        else:
            task_p = self.sample_permutations[self.c_task]
            start = self.c_sample
            if start + self.batch_size < len(task_p):
                end = start + self.batch_size
                s_idx = task_p[start: end]
                t_idx = self.c_task
                self.c_sample = end
            else:
                end = len(task_p)
                s_idx = task_p[start:end]
                t_idx = self.c_task
                self.c_sample = 0
                self.c_task += 1
            inputs = self.task_datasets[t_idx].train_images[s_idx]
            labels = self.task_datasets[t_idx].train_labels[s_idx]

            return self.task_permutations[t_idx], int(start / self.task_n_sample[t_idx]), inputs, labels




def load_mnist(path):
    logger.info('load mnist dataset.')
    train = datasets.MNIST(path, train=True, transform=None, target_transform=None, download=True)
    test = datasets.MNIST(path, train=False, transform=None, target_transform=None, download=True)
    return train, test

def load_cifar100(path):
    logger.info('load cifar100 dataset')
    train = datasets.CIFAR100(path, train=True, transform=None, target_transform=None, download=True)
    test = datasets.CIFAR100(path, train=False, transform=None, target_transform=None, download=True)
    return train, test

def load_tinyimageNet(path):
    logger.info('load tinyimageNet dataset')
    if os.path.exists(os.path.join(path, 'tinyImageNet200.pt')):
        return torch.load(os.path.join(path, 'tinyImageNet200.pt'))

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200')):
        subprocess.call("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip", shell=True)
        subprocess.call(f"unzip -q -o tiny-imagenet-200.zip  -d {path}", shell=True)
    # train
    train_path = os.path.join(path, 'tiny-imagenet-200', 'train')
    classes = os.listdir(train_path)
    classes_images = {}
    for c in classes:
        bounds = OrderedDict()
        with open(os.path.join(train_path, c, f'{c}_boxes.txt')) as f:
            for line in f.readlines():
                s = line.strip().split()
                bounds[s[0]] = torch.IntTensor([int(i) for i in s[1:]])
        images = []
        class_images_path = os.path.join(train_path, c, 'images')
        pics = list(bounds.keys())
        for p in pics:
            image = io.read_image(os.path.join(class_images_path, p))
            if image.size(0) != 3:
                del bounds[p]
            else:
                images.append(image)
        images = torch.stack(images)
        classes_images[c] = EasyDict(images=images, bounds=list(bounds.values()))

    class_label_dict = {c: i for i, c in enumerate(classes)}
    train = EasyDict(images=torch.cat([ci.images.float() / 255.0 for ci in classes_images.values()]),
                     labels=torch.cat([torch.IntTensor([class_label_dict[c]]* ci.images.size(0)) for c, ci in classes_images.items()])
                     )
    test_path = os.path.join(path, 'tiny-imagenet-200', 'val')
    bounds = OrderedDict()
    labels = []
    images = []
    with open(os.path.join(test_path, 'val_annotations.txt')) as f:
        for line in f.readlines():
            s = line.strip().split()
            image = io.read_image(os.path.join(test_path, 'images', s[0]))
            if image.size(0) == 3:
                bounds[s[0]] = torch.IntTensor([int(i) for i in s[2:]])
                labels.append(class_label_dict[s[1]])
                images.append(image)
    test = EasyDict(images=torch.stack(images), labels=torch.IntTensor(labels))
    torch.save((train, test), os.path.join(path, 'tinyImageNet200.pt'))
    return train, test

def load_miniimageNet(path, dim=64):
    logger.info('load miniimageNet dataset')
    if os.path.exists(os.path.join(path, f'miniImageNet{dim}.pt')):
        return torch.load(os.path.join(path, f'miniImageNet{dim}.pt'))


    # train
    train_labels = os.path.join(path, 'train.csv')
    val_labels = os.path.join(path, 'val.csv')
    test_labels = os.path.join(path, 'test.csv')

    images_path = os.path.join(path, 'images')

    class_labels = {}

    def load_images_labels(data_idx_file):
        idx = pd.read_csv(data_idx_file)
        images, labels = [], []
        for row in idx.itertuples():
            _, image_file, cls = row
            if cls not in class_labels:
                class_labels[cls] = len(class_labels)
            image = io.read_image(os.path.join(images_path, image_file))
            image = transforms.CenterCrop(min(image.shape[1], image.shape[2]))(image)
            image = transforms.functional.resize(image, [dim, dim])
            images.append(image)
            labels.append(class_labels[cls])
        images = torch.stack(images)
        labels = torch.LongTensor(labels)
        return images, labels

    train_images, train_labels = load_images_labels(train_labels)
    val_images, val_labels = load_images_labels(val_labels)
    test_images, test_labels = load_images_labels(test_labels)

    images = torch.cat([train_images, val_images, test_images], dim=0)
    labels = torch.cat([train_labels, val_labels, test_labels], dim=0)

    torch.save((images, labels), os.path.join(path, f'miniImageNet{dim}.pt'))
    return images, labels



def create_mnist_rotation(file, min_rot, max_rot, n_tasks):
    logger.info('init mnist rotation dataset in {}.'.format(file))
    m_train, m_test = load_mnist(os.path.join(args.path, 'mnist'))

    def rot_images(dataset, rot):
        result = torch.zeros_like(dataset.train_data)
        to_tensor=transforms.PILToTensor()
        for i, (image, label) in enumerate(dataset):
            result[i] = to_tensor(image.rotate(rot)).view([28, 28])
        return result.float() / 255.0

    task_datasets = []
    for t in range(n_tasks):
        min_r = 1.0 * t / n_tasks * (max_rot - min_rot) + min_rot
        max_r = 1.0 * (t + 1) / n_tasks * (max_rot - min_rot) + min_rot
        rot = random.random() * (max_r - min_r) + min_r
        train_images = rot_images(m_train, rot)
        test_images = rot_images(m_test, rot)
        logger.info('rotation mnist task {}: rotation={:.4f} over'.format(t, rot))
        task_datasets.append(EasyDict(train_images=train_images, train_labels=m_train.train_labels,
                                      test_images=test_images, test_labels=m_test.test_labels, rotation=rot))

    torch.save(task_datasets, file)

def create_mnist_permutation(file, n_tasks):
    logger.info('init mnist permutation dataset in {}.'.format(file))
    m_train, m_test = load_mnist(os.path.join(args.path, 'mnist'))

    def perm_images(dataset, p):
        to_tensor = transforms.PILToTensor()
        result = torch.zeros_like(dataset.train_data)
        for i, (image, label) in enumerate(dataset):
            result[i] = to_tensor(image).view(-1).index_select(0, p).view([28, 28])
        return result.float() / 255.0

    task_datasets = []
    for t in range(n_tasks):
        p = torch.randperm(28 * 28).long().view(-1)
        train_images = perm_images(m_train, p)
        test_images = perm_images(m_test, p)
        logger.info('permutation mnist task {} over'.format(t))
        task_datasets.append(EasyDict(train_images=train_images, train_labels=m_train.train_labels,
                                      test_images=test_images, test_labels=m_test.test_labels, perm=p))

    torch.save(task_datasets, file)


def create_cifar100_split(file, n_tasks, task_class_nums=None):
    logger.info('init cifar100 split dataset in {}.'.format(file))
    m_train, m_test = load_cifar100(os.path.join(args.path, 'cifar100'))

    cls_map = np.arange(100)
    np.random.shuffle(cls_map)

    def select_sample(dataset, c1, c2):
        map_targets = cls_map[dataset.targets]
        idx = (torch.BoolTensor(map_targets >= c1)) & (torch.BoolTensor(map_targets < c2))
        images = torch.FloatTensor(dataset.data[idx] / 255.0).permute(0, 3, 1, 2)
        labels = torch.LongTensor(dataset.targets)[idx]
        return images, labels

    if task_class_nums:
        assert len(task_class_nums) == n_tasks
        cpts = [int(t * (100 / np.sum(task_class_nums))) for t in task_class_nums]
    else:
        cpts = [int(100 / n_tasks)] * n_tasks

    task_datasets = []
    u = 0
    for t in range(n_tasks):
        train_images, train_labels = select_sample(m_train, u, u+cpts[t])
        test_images, test_labels = select_sample(m_test, u, u+cpts[t])
        u += cpts[t]
        logger.info('split cifar100 task {} over'.format(t))
        classes = list(set(to_numpy(train_labels)))
        logger.info(f'classes: {classes}')
        task_datasets.append(EasyDict(train_images=train_images, train_labels=train_labels,
                                      test_images=test_images, test_labels=test_labels,
                                      classes=classes))

    torch.save(task_datasets, file)

def create_tinyimageNet_split(file, n_tasks):
    logger.info('init miniimageNet split dataset in {}.'.format(file))
    m_train, m_test = load_tinyimageNet(os.path.join(args.path, 'tinyimageNet'))

    def select_sample(dataset, c1, c2):
        idx = (dataset.labels >= c1) & (dataset.labels < c2)
        images = dataset.images[idx] / 255.0
        labels = dataset.labels[idx]
        return images, labels

    cpt = int(200 / n_tasks)
    task_datasets = []
    for t in range(n_tasks):
        train_images, train_labels = select_sample(m_train, cpt * t, cpt * (t + 1))
        test_images, test_labels = select_sample(m_test, cpt * t, cpt * (t + 1))
        logger.info('split tinyimageNet task {} over'.format(t))
        task_datasets.append(EasyDict(train_images=train_images, train_labels=train_labels,
                                      test_images=test_images, test_labels=test_labels,
                                      classes=[cpt * t, cpt * (t + 1)]))

    torch.save(task_datasets, file)


def creat_miniimageNet_split(file, n_tasks, dim=64, task_class_nums=None):
    logger.info('init imageNet100 split dataset in {}.'.format(file))
    images, labels = load_miniimageNet(os.path.join(args.path, 'miniimageNet'), dim=dim)

    cls_map = np.arange(100)
    np.random.shuffle(cls_map)

    def split_train_test(images, train_size):
        idx = np.arange(images.shape[0])
        np.random.shuffle(idx)
        return images[idx[:train_size]], images[idx[train_size:]]

    if task_class_nums:
        assert len(task_class_nums) == n_tasks
        cpts = [int(t * (100 / np.sum(task_class_nums))) for t in task_class_nums]
    else:
        cpts = [int(100 / n_tasks)] * n_tasks
    task_datasets = []
    u = 0
    for t in range(n_tasks):
        train_images_list, test_images_list = [], []
        train_labels_list, test_labels_list = [], []
        classes = []
        for cls in cls_map[u: u+cpts[t]]:
            cls_images = images[labels == cls]
            train_images, test_images = split_train_test(cls_images, 500)
            train_images_list.append(train_images)
            test_images_list.append(test_images)
            train_labels_list.append(torch.LongTensor(train_images.shape[0]).fill_(cls))
            test_labels_list.append(torch.LongTensor(test_images.shape[0]).fill_(cls))
            classes.append(cls)
        u += cpts[t]
        train_images = torch.cat(train_images_list, dim=0) / 255.0
        test_images = torch.cat(test_images_list, dim=0) / 255.0
        train_labels = torch.cat(train_labels_list, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)

        logger.info('split miniimageNet{} task {} over'.format(dim, t))
        task_datasets.append(EasyDict(train_images=train_images, train_labels=train_labels,
                                      test_images=test_images, test_labels=test_labels,
                                      classes=classes))

    torch.save(task_datasets, file)





def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--n_tasks', default=11, type=int, help='number of tasks')
    parser.add_argument('--min_rot', default=0.,
                        type=float, help='minimum rotation')
    parser.add_argument('--max_rot', default=270.,
                        type=float, help='maximum rotation')
    parser.add_argument('--seed', default=4, type=int, help='random seed')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    mkdir(args.path)

    # create_mnist_rotation(os.path.join(args.path, 'mnist_rot_{n_tasks}.pt'.format(n_tasks=args.n_tasks)),
    #                       args.min_rot, args.max_rot, args.n_tasks)

    # create_mnist_permutation(os.path.join(args.path, 'mnist_perm_{n_tasks}.pt'.format(n_tasks=args.n_tasks)), args.n_tasks)


    # create_cifar100_split(os.path.join(args.path, 'cifar100_{n_tasks}_{seed}.pt.pt'.format(n_tasks=args.n_tasks, seed=args.seed)), args.n_tasks)

    # create_tinyimageNet_split(os.path.join(args.path, 'tinyimageNet_{n_tasks}.pt'.format(n_tasks=args.n_tasks)), args.n_tasks)

    # creat_miniimageNet_split(os.path.join(args.path, 'miniimageNet{dim}_{n_tasks}_{seed}.pt'.format(dim=64, n_tasks=args.n_tasks, seed=args.seed)),
    #                          args.n_tasks, dim=64)

    task_class_nums = [10] + [1] * 10
    create_cifar100_split(os.path.join(args.path, 'cifar100_{n_tasks}_{seed}.pt'.format(n_tasks=args.n_tasks, seed=args.seed)),
                          args.n_tasks, task_class_nums)
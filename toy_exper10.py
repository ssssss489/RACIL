



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


class Prototype(torch.nn.Module):
    def __init__(self, n_center, n_class, hidden_size, ):
        super(Prototype, self).__init__()
        self.n_center = n_center
        self.n_class = n_class
        self.hidden_size = hidden_size

        self.class_cov = torch.zeros([self.n_class, self.n_center, self.hidden_size, self.hidden_size]).cuda()
        self.class_mu = torch.zeros([self.n_class, self.n_center, self.hidden_size,]).cuda()
        self.class_n_sample = torch.zeros([self.n_class, self.n_center]).cuda()
        self.class_multivar_normals = []


    def forward(self, x, y, type):
        y_ = one_hot(y, self.n_class).unsqueeze(-1).repeat(1,1,self.n_center)
        if type=='mu':
            x_ = x.view([x.shape[0], 1, 1, x.shape[1]]) * y_.unsqueeze(-1)
            self.class_mu += x_.sum(0)
            self.class_n_sample += y_.sum(0)
        elif type=='cov':
            diff = x.view([x.shape[0], 1, 1, x.shape[1]]) - self.class_mu[0]
            diff_ = diff * y_.unsqueeze(-1)
            self.class_cov += torch.matmul(diff_.unsqueeze(-1), diff_.unsqueeze(-2)).sum(0)


    def compute_mu(self):
        self.class_mu = self.class_mu / self.class_n_sample.unsqueeze(-1)

    def compute_cov(self):
        self.class_cov = self.class_cov / self.class_n_sample.unsqueeze(-1)

    def set_multivar_normals(self):
        for cls in range(self.n_class):
            t = []
            for center in range(self.n_center):
                t.append(torch.distributions.multivariate_normal.MultivariateNormal(self.class_mu[cls, center],
                                                                                    self.class_cov[cls, center]
                                                                                    + torch.eye(self.hidden_size).cuda() * 1e-3))
            self.class_multivar_normals.append(t)


    def augment(self, x, y):
        x = self.scale(x)
        log_gmm_probs = self.log_likelihood(x, y, reduction=None)
        log_gmm_probs_clamp = torch.clamp(log_gmm_probs, -self.augment_limit, self.augment_limit)
        log_gmm_probs_scale = log_gmm_probs_clamp / (- 2 * self.augment_limit) + 0.5
        noise = 0.1 * torch.randn(x.shape).cuda() * log_gmm_probs_scale.unsqueeze(-1)
        augment = noise + x
        augment = self.anti_scale(augment)
        return augment

    def sample(self, y):
        y_ = one_hot(y, self.n_class)
        class_n_sample = y_.sum(0)
        new_x, new_y = [], []
        center = 0
        for cls in range(self.n_class):
            if class_n_sample[cls] > 0:
                new_x.append(self.class_multivar_normals[cls][center].sample([int(class_n_sample[cls])]))
                new_y.append(torch.empty([int(class_n_sample[cls])]).fill_(cls))
        new_x = torch.cat(new_x, dim=0).cuda()
        new_y = torch.cat(new_y, dim=0).long().cuda()
        return new_x, new_y


class model(torch.nn.Module):
    def __init__(self, data, args):
        super(model, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.n_classes = models_setting[data].classifier[-1]

        self.encoder = MLP_Encoder(models_setting[data].encoder)
        self.decoder = MLP_Decoder(models_setting[data].decoder)
        self.classifier = MLP_Classifier(models_setting[data].classifier)
        self.Prototypes = []
        for _ in range(self.n_task):
            self.Prototypes.append(Prototype(1, self.n_classes, models_setting[data].encoder[-1]))

        self.buffer = Buffer(args.n_task, args.n_memory, self.n_classes)
        self.task_decoders = {}
        self.task_clusters = {}

        self.classifier_loss_fn = torch.nn.CrossEntropyLoss()
        self.decoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))

        self.lr = args.lr
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=args.lr)

        self.feature_net_fine_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr * 0.1)
        self.classifier_fine_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr * 0.1)

        self.observed_tasks = []
        self.cuda()

        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")

    def forward(self, x, with_details=False):
        feature = self.encoder(x)
        feature = normalize(feature)
        y = self.classifier(feature)
        if with_details:
            return y, feature
        return y

    def learn_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0

        class_offset = get_class_offset(task_p)
        _, labels = compute_output_offset(None, labels, class_offset)
        _, feature = self.forward(inputs, with_details=True)

        decoder_outputs = self.decoder(feature)
        loss += self.decoder_loss_fn(decoder_outputs, inputs) * 100

        decoder_outputs_numpy = decoder_outputs.view([-1,28,28]).data.cpu().numpy()
        inputs_numpy = inputs.data.cpu().numpy()

        if task_p > 0:
            pt = np.random.choice(self.observed_tasks[:-1])
            prototype = self.Prototypes[pt]
            decoder = self.task_decoders[task_p-1]
            pt_class_offset = get_class_offset(pt)
            _, pt_labels = compute_output_offset(None, labels, pt_class_offset)
            pt_feature, pt_labels = prototype.sample(pt_labels)
            pt_inputs = decoder(pt_feature).view([-1, 28, 28])
            pt_decoder_outputs = self.decoder(pt_feature)
            loss_ = self.decoder_loss_fn(pt_decoder_outputs, pt_inputs) * 100
            loss += loss_


        loss.backward(retain_graph=True)
        self.decoder_optimizer.step()

        with torch.no_grad():
            sample_feature = self.Prototypes[task_p].sample(labels)

        return float(loss.item())

    def compute_prototype(self,inputs, labels, get_class_offset, task_p, type):

        class_offset = get_class_offset(task_p)
        _, feature = self.forward(inputs, with_details=True)
        _, labels = compute_output_offset(None, labels, class_offset)
        self.Prototypes[task_p](feature, labels, type)


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)

            self.decoder = deepcopy(self.decoder)
            self.task_decoders[task_p] = self.decoder
            self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.lr)

        class_offset = get_class_offset(task_p)
        logits, feature = self.forward(inputs, with_details=True)
        logits, labels = compute_output_offset(logits, labels, class_offset)

        classifier_loss = self.classifier_loss_fn(logits, labels)
        loss += classifier_loss

        if (torch.isnan(loss).sum() > 0):
            print("here!")

        if task_p > 0:
            # pt = np.random.choice(self.observed_tasks[:-1])
            # pt_inputs, pt_labels, pts = self.buffer.get_buffer_samples([pt], labels.shape[0])
            # pt_class_offset = get_class_offset(pts)
            #
            # pt_logits, pt_feature = self.forward(pt_inputs, with_details=True)
            # loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
            # loss += 1.0 * loss_

            pt = np.random.choice(self.observed_tasks[:-1])
            prototype = self.Prototypes[pt]
            decoder = self.task_decoders[task_p-1]
            pt_class_offset = get_class_offset(pt)
            pt_feature, pt_labels = prototype.sample(labels)
            pt_inputs = decoder(pt_feature).view([-1,28,28])
            pt_logits, new_pt_feature = self.forward(pt_inputs, with_details=True)
            loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
            loss += 1.0 * loss_




        self.buffer.save_buffer_samples(inputs, labels, task_p)

        loss.backward(retain_graph=True)

        self.classifier_optimizer.step()
        self.encoder_optimizer.step()

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


class Decoder_train(base_train):
    def __init__(self, data_loader, model, args, logger):
        super(Decoder_train, self).__init__(data_loader, model, args, logger)
        self.n_epoch_train = args.n_epoch_train
        self.n_epoch_learn = args.n_epoch_learn

    def train(self):
        total = self.data_loader.task_n_sample[0]
        losses, train_acc = [], []
        self.metric(-1)
        for task_p in range(self.data_loader.n_task):
            self.class_offset = self.get_class_offset(task_p)
            self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
            self.logger.info(f'task {task_p} is beginning to train.')
            for epoch in range(0, self.n_epoch_train):
                bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
                for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(task_p, epoch=epoch)):
                    if self.cuda:
                        batch_inputs = batch_inputs.cuda()
                        batch_labels = batch_labels.to(torch.int64).cuda()
                    loss, acc = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
                    # loss += self.model.learn_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
                    losses.append(loss)
                    train_acc.append(acc)
                    bar.update(batch_labels.size(0))
                bar.close()
                print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
                self.metric(task_p)
            self.logger.info(f'task {task_p} has trained over ,eval result is shown below.')
            self.logger.info(f'task {task_p} is beginning to learn.')
            for epoch in range(self.n_epoch_train, self.n_epoch_train + self.n_epoch_learn):
                bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
                losses = []
                if epoch == self.n_epoch_train:
                    for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(task_p, epoch=epoch)):
                        if self.cuda:
                            batch_inputs = batch_inputs.cuda()
                            batch_labels = batch_labels.to(torch.int64).cuda()
                        self.model.compute_prototype(batch_inputs, batch_labels, self.get_class_offset, task_p, 'mu')
                        bar.update(batch_labels.size(0))
                    bar.close()
                    self.model.Prototypes[task_p].compute_mu()
                    print(f'mean of feature computed')
                elif epoch == self.n_epoch_train + 1:
                    for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(task_p, epoch=epoch)):
                        if self.cuda:
                            batch_inputs = batch_inputs.cuda()
                            batch_labels = batch_labels.to(torch.int64).cuda()
                        self.model.compute_prototype(batch_inputs, batch_labels, self.get_class_offset, task_p, 'cov')
                        bar.update(batch_labels.size(0))
                    bar.close()
                    self.model.Prototypes[task_p].compute_cov()
                    self.model.Prototypes[task_p].set_multivar_normals()
                    print(f'covariance of feature computed')
                else:
                    for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(task_p, epoch=epoch)):
                        if self.cuda:
                            batch_inputs = batch_inputs.cuda()
                            batch_labels = batch_labels.to(torch.int64).cuda()
                        loss = self.model.learn_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
                        losses.append(loss)
                        bar.update(batch_labels.size(0))
                    bar.close()
                    print(f'    loss = {np.mean(losses)}')
            self.logger.info(f'task {task_p} decoder has learned over')







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
    parser.add_argument('--n_epoch_train', default=1, type=int, help='number of epochs for train')
    parser.add_argument('--n_epoch_learn', default=4, type=int, help='number of epochs for train')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU')
    parser.add_argument('--n_memory', default=100, help='number of memories per task')
    args = parser.parse_args()

    args.n_epoch = args.n_epoch_train + args.n_epoch_learn

    torch.manual_seed(args.seed)

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))

    task = args.dataset.split('_')[0]
    # model = MLP(task, args)
    model = model(task, args)

    data_loader = task_data_loader(task_datasets, args)

    trainer = Decoder_train(data_loader, model, args, logger)
        #base_train(data_loader, model, args, logger)
    trainer.train()

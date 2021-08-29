



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


class Cluster(torch.nn.Module):
    def __init__(self, n_center, n_class, hidden_size,
                 sigma_upper_bound=1, sigma_lower_bound=10, init_mu=0.1,
                 log_prob_upper_bound=1.0, log_prob_lower_bound=0.0,
                 augment_limit=10,
                 lr_sigma=1, lr_mu=0.01, lr_pi=0.01,
                 beta=0.5, gama=5): #16 * 128 *
        super(Cluster, self).__init__()
        self.n_center = n_center
        self.n_class = n_class
        self.hidden_size = hidden_size

        self.class_sigma = None  # precision_matrix_diag std ** -1
        self.class_mu = None
        self.class_pi = None
        self.mask = None
        self.W = None

        self.sigma_upper_bound = sigma_upper_bound # std ** -1
        self.sigma_lower_bound = sigma_lower_bound
        self.init_mu = init_mu
        self.augment_limit = augment_limit
        self.log_prob_upper_bound = log_prob_upper_bound
        self.log_prob_lower_bound = log_prob_lower_bound

        self.lr_sigma = lr_sigma
        self.lr_mu = lr_mu
        self.lr_pi = lr_pi
        self.beta = beta
        self.gama = gama

        self.init_parameters()


    def init_parameters(self):
        self.class_mu = torch.nn.Parameter(
            torch.rand([self.n_class, self.n_center, self.hidden_size]) * (self.init_mu * 2) - self.init_mu,
        )
        self.class_sigma = torch.nn.Parameter(
            torch.ones(self.n_class, self.n_center, self.hidden_size) * self.sigma_upper_bound)
        class_pi = torch.ones(self.n_class, self.n_center).abs()
        self.class_pi = torch.nn.Parameter(class_pi / class_pi.sum(dim=1, keepdim=True))
        mask = (torch.rand([self.n_center, self.hidden_size]) > self.beta).float()
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        self.W = torch.nn.Parameter(torch.randn([self.hidden_size, self.hidden_size]) * 0.1 , requires_grad=False)
        self.cuda()

    def forward(self, x):
        mask = x.unsqueeze(1) * self.mask.unsqueeze(0)
        mul = torch.matmul(mask, self.W) # b , k , h
        distances = self.distance(mul)
        k_assign = distances
        k_assign = k_assign / k_assign.sum(-1, keepdim=True)

         # add pi  b, c, k   k.sum = 1
        return mul, k_assign, distances

    def distance(self, x): # x = b , k ,h
        diff = x.unsqueeze(1) - self.class_mu.unsqueeze(0) # b, c, k ,h
        score = (1 + (diff ** 2).sum(-1)) ** -1 # b, c, k
        score_norm = score / (score.sum(2, keepdim=True)+ 1e-8)
        return score_norm # b, c, k


    def log_prob(self, x): # x = b , k ,h
        # mu = torch.index_select(self.class_mu, dim=0, index=y) # k , h
        # sigma = torch.index_select(self.class_sigma, dim=0, index=y) #  b , k , h
        diff = x.unsqueeze(1) - self.class_mu.unsqueeze(0)
        norm = diff / self.class_sigma.unsqueeze(0) # b, c, k ,h
        # log_exp = - 0.5 * (norm.square().sum(-1)) # b * k
        # const = - 0.5 * self.hidden_size * np.log(2. * np.pi)
        log_exp = - 0.5 * norm.square() # b ,c, k , h
        log_det = - self.class_sigma.unsqueeze(0).log() # 1, c, k ,h
        const = - 0.5 * np.log(2. * np.pi)
        log_probs = log_exp + const + log_det # b , c, k , h
        log_probs = self.show_grad(log_probs)
        return log_probs # b , c, k , h

    def show_grad(self, v):
        self.n = torch.zeros_like(v, requires_grad=True).cuda()
        return v + self.n


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
        batch_shape = y.shape[0]
        uniform = torch.rand(batch_shape).unsqueeze(-1).cuda()
        softmax_pi = torch.index_select(torch.softmax(self.class_pi, dim=1), dim=0, index=y)
        upper_ladder_pi = torch.matmul(softmax_pi, torch.ones([self.n_center, self.n_center]).triu().cuda())
        lower_ladder_pi = torch.matmul(softmax_pi, torch.ones([self.n_center, self.n_center]).triu(1).cuda())
        center_idx = torch.logical_and(uniform > lower_ladder_pi, uniform < upper_ladder_pi).float() # b * k
        mu = (torch.index_select(self.class_mu, dim=0, index=y) * center_idx.unsqueeze(-1)).sum(1)
        sigma = (torch.index_select(self.class_sigma, dim=0, index=y) * center_idx.unsqueeze(-1)).sum(1)
        normal = torch.randn([batch_shape, self.hidden_size]).cuda()
        result = (normal / sigma) * 0.1 + mu
        result = self.anti_scale(result)
        return result


class model(torch.nn.Module):
    def __init__(self, data, args):
        super(model, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.n_classes = models_setting[data].classifier[-1]

        self.encoder = MLP_Encoder(models_setting[data].encoder)
        self.decoder = MLP_Decoder(models_setting[data].decoder)
        self.classifier = MLP_Classifier(models_setting[data].classifier)
        self.cluster = Cluster(4, self.n_classes, models_setting[data].encoder[-1])

        self.buffer = Buffer(args.n_task, args.n_memory, self.n_classes)
        self.task_decoders = {}
        self.task_clusters = {}

        self.classifier_loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
        self.decoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))

        self.lr = args.lr
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=args.lr)
        self.cluster_optimizer = torch.optim.SGD(self.cluster.parameters(), lr=args.lr)

        self.feature_net_fine_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr * 0.1)
        self.classifier_fine_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr * 0.1)

        self.observed_tasks = []
        self.cuda()

        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")

    def forward(self, x, with_details=False):
        feature = self.encoder(x)
        # feature = normalize(feature)
        cluster_feature, k_assign, distances = self.cluster(feature)
        shape = cluster_feature.shape # b , k , h
        logits_k = self.classifier(cluster_feature.view([shape[0] * shape[1], -1])) # b * k , c
        logits_k_softmax = logits_k.softmax(-1)
        softmax_k = logits_k_softmax.view([shape[0], shape[1], -1]).transpose(1, 2) # b , c , k

        if with_details:
            return softmax_k, cluster_feature, k_assign, distances

        softmax = (softmax_k * k_assign).max(-1)[0] # b , c, k -> b , c
        return softmax

    def learn_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0

        class_offset = get_class_offset(task_p)
        _, labels = compute_output_offset(None, labels, class_offset)
        _, feature = self.forward(inputs, with_details=True)
        augment_feature = self.cluster.augment(feature.data.detach(), labels.data.detach())

        decoder_outputs = self.decoder(augment_feature)
        loss += self.decoder_loss_fn(decoder_outputs, inputs) * 100

        decoder_outputs_numpy = decoder_outputs.view([-1,28,28]).data.cpu().numpy()
        inputs_numpy = inputs.data.cpu().numpy()

        loss.backward(retain_graph=True)
        self.decoder_optimizer.step()

        with torch.no_grad():
            sample_feature = self.cluster.sample(labels)

        return float(loss.item())


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            # self.cluster = Cluster(4, self.n_classes, models_setting['mnist'].encoder[-1])
            # self.task_clusters[task_p] = self.cluster
            # self.cluster_optimizer = torch.optim.SGD(self.cluster.get_parameters(), lr=args.lr * 0.1)

            # self.decoder = deepcopy(self.decoder)
            # self.task_decoders[task_p] = self.decoder
            # self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.lr)

        class_offset = get_class_offset(task_p)
        logits_k, cluster_feature, k_assign, distances = self.forward(inputs, with_details=True)
        logits_k, labels = compute_output_offset(logits_k, labels, class_offset)

        labels_onehot = one_hot(labels, self.n_classes).unsqueeze(-1)
        k_assign_ = k_assign.data.detach()
        kl_loss = (labels_onehot * k_assign_ * (k_assign_.log() - logits_k.log())).sum(-1).sum(-1)
        # entropy_loss = - (labels_onehot * k_assign * k_assign.log()).sum(-1).sum(-1)

        freq = k_assign.sum(0, keepdim=True)
        soft_assign = k_assign ** 2 / freq
        soft_assign = soft_assign / soft_assign.sum(-1, keepdim=True)
        # cluster_loss = (labels_onehot * soft_assign * (soft_assign.log() - k_assign.log())).sum(-1).sum(-1)
        cluster_loss = - (labels_onehot * k_assign * k_assign.log()).sum(-1).sum(-1)

        loss_ = kl_loss + cluster_loss * 0.01

        loss += loss_.mean()
        if (torch.isnan(loss).sum() > 0):
            print("here!")



        # if task_p > 0:
        #     pt = np.random.choice(self.observed_tasks[:-1])
        #     pt_inputs, pt_labels, pts = self.buffer.get_buffer_samples([pt], labels.shape[0])
        #     pt_class_offset = get_class_offset(pts)
        #
        #     pt_logits, pt_feature = self.forward(pt_inputs, with_details=True)
        #     loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
        #     loss += 1.0 * loss_

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        loss.backward(retain_graph=True)

        with torch.no_grad():
            self.cluster.class_mu.grad *= 1e4

        self.classifier_optimizer.step()
        self.encoder_optimizer.step()
        self.cluster_optimizer.step()

        with torch.no_grad():
            logits = self.forward(inputs)
            logits, labels = compute_output_offset(logits, labels, class_offset)
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
    parser.add_argument('--n_epoch_train', default=8, type=int, help='number of epochs for train')
    parser.add_argument('--n_epoch_learn', default=2, type=int, help='number of epochs for train')
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





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



def cluster_loss(q):
    # Unsupervised Deep Embedding for Clustering Analysis
    if torch.min(q) < 0:
        raise ValueError('cluster distances must be greater than 0')
    q = q / q.sum(dim=1, keepdim=True)
    f = torch.sum(q, dim=0, keepdim=True)
    t = q ** 2 / f
    p = (t / torch.sum(t, dim=1, keepdim=True)).data.detach()
    l = torch.sum(p * torch.log(p / q))
    return l * 1

class Cluster(torch.nn.Module):
    def __init__(self, n_center, n_class, hidden_size, init_mu=None, beta=1, gama=0.8): #16 * 128 *
        super(Cluster, self).__init__()
        self.n_center = n_center
        self.n_class = n_class
        self.hidden_size = hidden_size

        self.class_cov = None
        self.class_mu = None
        self.class_pi = None
        self.class_multivariate_normal = []

        self.init_mu = init_mu
        self.params_inited = False
        self.eps_mat = torch.eye(self.hidden_size).cuda() * 1e-4
        self.observed_class_resp_sum = torch.zeros([self.n_class, self.n_center]).cuda()

        self.beta = beta
        self.gama = gama
        self.init_class_mulitvariat_normal()

    def get_kmeans_mu(self, x, y, init_times=50, min_delta=1e-3):
        y_T = one_hot(y, self.n_class).T
        class_mu = []
        for c in range(y_T.shape[0]):
            c_idx = y_T[c] > 0
            class_x = x[c_idx]
            x_min, x_max = class_x.min(), class_x.max()
            class_x = (class_x - x_min) / (x_max - x_min)
            if class_x.shape[0] < self.n_center:
                u = torch.index_select(torch.eye(class_x.shape[0]), dim=0, index=torch.randint(class_x.shape[0], [self.n_center])) \
                    + torch.index_select(torch.eye(class_x.shape[0]), dim=0, index=torch.randint(class_x.shape[0], [self.n_center]))
                centers = torch.matmul((u / u.sum(dim=1, keepdim=True)).cuda(), class_x)
            else:
                centers = class_x[np.random.choice(np.arange(class_x.shape[0]), size=self.n_center, replace=False)]
            if centers.shape[0] < self.n_center:
                centers = torch.cat([centers, torch.randn([self.n_center - centers.shape[0], self.hidden_size]).cuda()], dim=0)
            delta = np.inf
            for i in range(init_times):
                if delta < min_delta:
                    break
                centers_old = centers.clone()
                dis = torch.norm((class_x.unsqueeze(1) - centers.unsqueeze(0)), p=2, dim=2)
                cls = dis.argmin(dim=1)
                for c in range(self.n_center):
                    centers[c] = class_x[cls == c].mean(dim=0)
                delta = torch.norm((centers_old - centers), dim=1).max()
                if (torch.isnan(centers).sum() > 0):
                    print("here!")
            mu = centers * (x_max - x_min) + x_min
            class_mu.append(mu)
        class_mu = torch.stack(class_mu, dim=0)

        return class_mu


    def init_class_mulitvariat_normal(self):
        if self.init_mu is not None:
            self.class_mu = torch.nn.Parameter(self.init_mu, requires_grad=False)
        else:
            self.class_mu = torch.nn.Parameter(torch.randn(self.n_class, self.n_center, self.hidden_size), requires_grad=False)

        self.class_cov = torch.nn.Parameter(torch.eye(self.hidden_size).view(1, 1, self.hidden_size, self.hidden_size).
                                            repeat(self.n_class, self.n_center, 1, 1) * 0.5, requires_grad=False)
        class_pi = torch.ones(self.n_class, self.n_center).abs()
        self.class_pi = torch.nn.Parameter(class_pi / class_pi.sum(dim=1, keepdim=True), requires_grad=False)
        self.cuda()
        class_mulitvariat_normal = []
        for cls in range(self.n_class):
            for k in range(self.n_center):
                mu = self.class_mu[cls, k]
                cov = self.class_cov[cls, k]
                mulitvariat_normal = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
                class_mulitvariat_normal.append(mulitvariat_normal)
        self.class_multivariate_normal = class_mulitvariat_normal


    def e_step(self, x, y):
        if self.init_mu is None:
            kmeans = self.get_kmeans_mu(x, y)
            self.__init__(self.n_center, self.n_class, self.hidden_size, init_mu=kmeans)
        y_ = one_hot(y, self.n_class)
        log_probs = []
        for mulitvariat_normal in self.class_multivariate_normal:
            log_probs.append(mulitvariat_normal.log_prob(x))
        log_probs = torch.stack(log_probs, dim=0).T.view([-1, self.n_class, self.n_center]).cuda()
        log_probs_class = torch.matmul(y_.unsqueeze(dim=1), log_probs).squeeze(dim=1)
        log_probs_class_scal = log_probs_class - log_probs_class.min(dim=1, keepdim=True)[0].data.detach()
        log_probs_class_scal = log_probs_class_scal / log_probs_class_scal.sum(dim=1, keepdim=True).abs().data.detach()
        #log_pi = torch.log(torch.index_select(self.class_pi, dim=0, index=y))
        ones = torch.ones_like(log_probs_class)
        log_pi = torch.log(ones / ones.sum(dim=1, keepdim=True))
        log_probs_sum = torch.logsumexp(log_pi + log_probs_class, dim=1, keepdim=True)
        resp = torch.exp(log_pi + log_probs_class - log_probs_sum)
        # log_probs_sum = torch.logsumexp(log_pi + log_probs_class_scal, dim=1, keepdim=True)
        # resp = torch.exp(log_pi + log_probs_class_scal - log_probs_sum)
        # log_probs_sum = torch.logsumexp(log_probs_class_scal, dim=1, keepdim=True)
        # resp = torch.exp(log_probs_class_scal - log_probs_sum)
        if (torch.isnan(resp).sum() > 0):
            print("here!")
        return log_probs_sum, resp, log_probs_class_scal - log_probs_sum

    def m_step(self, x, y, resp):
        # resp 100 * 16
        y_ = one_hot(y, self.n_class)
        # ones = torch.arange(4).unsqueeze(dim=0).repeat(100, 1).cuda() + 1
        # resp = ones / ones.sum(dim=1, keepdim=True)
        class_sample = torch.sum(y_, dim=0, keepdim=True).T # 10 * 1
        class_x = torch.matmul(y_.unsqueeze(2), x.unsqueeze(1)).transpose(1, 0) # 10 * 100 * 128
        class_resp = torch.matmul(y_.unsqueeze(2), resp.unsqueeze(1)).transpose(1, 0) # 10 * 100 * 16

        class_resp_sum = torch.sum(class_resp, dim=1) + 1e-8 # 10 * 16
        class_mu = torch.sum(class_resp.unsqueeze(3) * class_x.unsqueeze(2), dim=1) / class_resp_sum.unsqueeze(-1) # 10 * 16 * 128

        diff = class_x.unsqueeze(2) - class_mu.unsqueeze(1) # 10 * 100 * 16 * 128

        class_cov = (torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2))
                     * class_resp.view(list(class_resp.shape) + [1, 1])).sum(dim=1) / class_resp_sum.unsqueeze(-1).unsqueeze(-1)\
                     + self.eps_mat

        class_pi = (class_resp_sum + 1e-10) / (class_sample + 1e-10) #10 * 16
        class_pi = class_pi / class_pi.sum(dim=1, keepdim=True)
        return class_pi, class_mu, class_cov, class_resp_sum

    def update(self, x, y, resp):
        class_pi, class_mu, class_cov, class_resp_sum = self.m_step(x, y, resp)
        new_observed_class_resp_sum = self.observed_class_resp_sum + class_resp_sum
        w1, w2 = self.observed_class_resp_sum / new_observed_class_resp_sum, class_resp_sum / new_observed_class_resp_sum

        self.class_pi.copy_(w1 * self.class_pi + w2 * class_pi)
        self.class_pi.copy_(self.class_pi / self.class_pi.sum(dim=1, keepdim=True))
        self.class_mu.copy_(w1.unsqueeze(-1) * self.class_mu + w2.unsqueeze(-1) * class_mu)
        self.class_cov.copy_(w1.view([self.n_class, self.n_center, 1, 1]) * self.class_cov +
                             w2.view([self.n_class, self.n_center, 1, 1]) * class_cov)
        self.observed_class_resp_sum = new_observed_class_resp_sum

        class_mulitvariat_normal = []
        for cls in range(self.n_class):
            for k in range(self.n_center):
                mu = self.class_mu[cls, k]
                cov = self.class_cov[cls, k]
                mulitvariat_normal = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
                class_mulitvariat_normal.append(mulitvariat_normal)
        self.class_multivariate_normal = class_mulitvariat_normal




    #
    # def compute_loss(self, x, y):
    #     # x = x.cpu()
    #     y_ = one_hot(y, self.n_class)
    #     class_K_pi = torch.softmax(self.class_K_pi, dim=-1)
    #     log_probs = []
    #     for mulitvariat_normal in self.class_K_multivariate_normal:
    #         log_probs.append(mulitvariat_normal.log_prob(x))
    #     log_probs_t = torch.stack(log_probs, dim=1).view([-1, self.n_class, self.n_center]).cuda()
    #     if (torch.isnan(log_probs_t).sum() > 0):
    #         print("here!")
    #     # N * c_k
    #     log_probs_c = torch.matmul(y_.unsqueeze(dim=1), log_probs_t).squeeze(dim=1)
    #     # log_probs_scale = log_probs_c / log_probs_c.max(dim=1, keepdim=True)[0].abs().data.detach()
    #     pi = torch.index_select(class_K_pi, dim=0, index=y)
    #     probs_sum = torch.logsumexp(torch.log(pi) + log_probs_c, dim=1)
    #     # probs_sum = (pi * torch.exp(log_probs_scale)).sum(dim=1)
    #     # log_probs_sum = torch.log(probs_sum).sum()
    #     log_probs_sum = probs_sum.mean()
    #     return - log_probs_sum

    # def clear_grad(self):
    #     if self.class_K_pi.grad is not None:
    #         self.class_K_pi.grad.zero_()
    #     if self.class_K_mean_vector.grad is not None:
    #         self.class_K_mean_vector.grad.zero_()
    #     if self.class_K_tril_mats.grad is not None:
    #         self.class_K_tril_mats.grad.zero_()
    #
    # def parse_grad(self, lr, threshold=0.02):
    #     eye = torch.eye(self.hidden_size).unsqueeze(0).cuda()
    #     diag_val = self.class_K_tril_mats
    #     diag_grad = self.class_K_tril_mats.grad.data
    #     new_diag_val = diag_val - lr * diag_grad
    #     new_grad = torch.where((new_diag_val * eye + (1 - eye) < threshold), torch.zeros_like(diag_grad),
    #                 self.class_K_tril_mats.grad.data)
    #     self.class_K_tril_mats.grad.copy_(new_grad)



    # def update(self, ):

        # log_prob = torch.zeros(y.shape)
        # for i, l in enumerate(y):
        #     f = x[i]
        #     K_mulitvariat_normal = self.class_K_multivariate_normal[l]
        #     K_pi = class_K_pi[l]
        #     probs = torch.zeros([self.n_center])
        #     for k in range(self.n_center):
        #         probs[k].add_(torch.exp(K_mulitvariat_normal[k].log_prob(f)))
        #     log_prob[i].add_(torch.log(torch.matmul(K_pi, probs)))
        # return log_prob.sum().cuda()

    # def init_trans_mats(self):
    #     self.class_trans_mats = torch.autograd.Variable(
    #         torch.randn([self.n_class,  self.hidden_size, self.n_center]).cuda(), requires_grad=True)
    #
    #     # self.class_bias = torch.autograd.Variable(
    #     #     torch.randn([self.n_class,  self.n_center]).cuda(), requires_grad=True)
    #
    #
    # def compute_static(self, x, y):
    #     y_= one_hot(y, self.n_class)
    #     trans_mats = torch.index_select(self.class_trans_mats, dim=0, index=y)  # 100 * 4 * 128
    #     # bias = torch.index_select(self.class_bias, dim=0, index=y)
    #     trans_result = torch.matmul(x.view([-1, 1, self.hidden_size]), trans_mats).view([-1, self.n_center]) # + bias
    #     cls_trans_result = torch.matmul(y_.view([-1, self.n_class, 1]),
    #                                     trans_result.view([-1, 1, self.n_center])).transpose(0, 1)
    #     cov_mat = torch.matmul(cls_trans_result.transpose(1, 2), cls_trans_result)
    #     mean_vector = cls_trans_result.sum(dim=1)
    #     new_class_n_sample = self.class_n_sample + y_.sum(dim=0)
    #     self.class_cov_mats = (self.class_cov_mats * self.class_n_sample.view([-1,1,1]) + cov_mat) / new_class_n_sample.view([-1,1,1])
    #     self.class_means_vector = (self.class_means_vector * self.class_n_sample.view([-1, 1]) + mean_vector) / new_class_n_sample.view([-1, 1])
    #     self.class_n_sample = new_class_n_sample
    #
    # def compute_trans_mat_pinv(self):
    #     mats = to_numpy(self.class_trans_mats)
    #     pinvs = []
    #     for m in mats:
    #         p = np.linalg.pinv(m)
    #         pinvs.append(p)
    #     pinvs = np.array(pinvs)
    #     self.class_trans_mats_pinv = torch.from_numpy(pinvs).cuda()
    #     cov_trans = []
    #     for cov_mat in self.class_cov_mats:
    #         evals, evecs = torch.linalg.eigh(cov_mat)
    #         cov_trans_mat = torch.matmul(evecs, evals.sqrt().diag()).T
    #         cov_trans.append(cov_trans_mat)






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

        self.classifier_loss_fn = torch.nn.CrossEntropyLoss()
        self.cluster_loss_fn = cluster_loss
        self.decoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))
        self.encoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))
        #self.encoder_loss_fn = lambda x, y: torch.sum(y * torch.log(y / x))# torch.nn.KLDivLoss()(torch.log(x), y)

        self.lr = args.lr
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=args.lr)
        # self.cluster_optimizer = torch.optim.SGD(self.cluster.get_parameters(), lr=args.lr * 0.1)

        self.feature_net_fine_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr * 0.1)
        self.classifier_fine_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr * 0.1)

        self.observed_tasks = []
        self.cuda()

        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")

    def forward(self, x, with_details=False):
        feature = self.encoder(x)
        feature = normalize(feature)
        y = self.classifier(feature * 11)
        if with_details:
            return y, feature
        return y

    def encoder_to_decoder(self, x):
        feature = self.encoder(x)
        feature = normalize(feature)
        pseudo_x = self.decoder(feature * 11)
        return pseudo_x

    def decoder_to_encoder(self, feature, decoder):
        # feature = normalize(feature)
        # pseudo_x = enhance_result(decoder(feature * 11))
        pseudo_x = decoder(feature * 11)
        decoder_outputs_numpy = pseudo_x.view([-1, 28, 28]).data.cpu().numpy()
        pseudo_feature = self.encoder(pseudo_x)
        pseudo_feature = normalize(pseudo_feature)
        return pseudo_feature

    def cluster_statics(self, inputs, labels, task_p):
        self.train()
        self.zero_grad()
        logits, feature = self.forward(inputs, with_details=True)
        self.cluster.compute_static(feature, labels)


    # def train_decoder(self, ):


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        torch.cuda.empty_cache()
        loss = 0

        # torch.autograd.set_detect_anomaly(True)

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            self.cluster = Cluster(4, self.n_classes, models_setting['mnist'].encoder[-1])
            self.task_clusters[task_p] = self.cluster
            # self.cluster_optimizer = torch.optim.SGD(self.cluster.get_parameters(), lr=args.lr * 0.1)

            self.decoder = deepcopy(self.decoder)
            self.task_decoders[task_p] = self.decoder
            self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.lr)

        class_offset = get_class_offset(task_p)
        logits, feature = self.forward(inputs, with_details=True)

        # cluster_distance = self.cluster.compute_distance(feature, labels.data.detach())
        log_probs_sum, resp, log_resp = self.cluster.e_step(feature, labels.data.detach()) # + self.cluster.orthogonal_loss()
        cluster_loss = -(resp * log_resp).sum() # -log_probs_sum.mean()
        # loss += cluster_loss * 0.1

        logits, labels = compute_output_offset(logits, labels, class_offset)
        loss += self.classifier_loss_fn(logits, labels.detach())

        # decoder_outputs = self.encoder_to_decoder(inputs)
        # loss += self.decoder_loss_fn(decoder_outputs, inputs) * 10
        #
        # loss += self.cluster_loss_fn(project_dis) * 0.1

        if task_p > 0:
            pt = np.random.choice(self.observed_tasks[:-1])
            pt_inputs, pt_labels, pts = self.buffer.get_buffer_samples([pt], labels.shape[0])
            pt_class_offset = get_class_offset(pts)

            pt_logits, pt_feature = self.forward(pt_inputs, with_details=True)
            loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
            loss += 1.0 * loss_

        # decoder_outputs_numpy = decoder_outputs.view([-1,28,28]).data.cpu().numpy()
        # inputs_numpy = inputs.data.cpu().numpy()
        # loss += torch.mean((class_feature) ** 2)# * 0.1

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        # with torch.autograd.detect_anomaly():
        loss.backward(retain_graph=True)

        # if task_p > 0:
        #     gradient_project(dict(self.classifier.named_parameters()), ref_classifier_param_dicts)
        #     gradient_project(dict(self.feature_net.named_parameters()), ref_feature_net_param_dicts)
        # self.projector.update_centers(feature.data.detach(), project_dis.data.detach())
        # self.projector.update_centers(class_feature.data.detach(), project_dis.data.detach(), labels.detach(), self.n_classes)

        # m = self.cluster.m_step()
        with torch.no_grad():
            self.cluster.update(feature, labels.data.detach(), resp)

        # if task_p == 0:
        #     self.classifier_optimizer.step()
        #     self.encoder_optimizer.step()
        #     self.decoder_optimizer.step()
        #     # self.cluster_optimizer.step()
        # else:
        #     self.classifier_optimizer.step()
        #     self.encoder_optimizer.step()
        #     self.decoder_optimizer.step()
            # self.cluster_optimizer.step()

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

    def train(self):
        total = self.data_loader.task_n_sample[0]
        losses, train_acc = [], []
        self.metric(-1)
        for task_p in range(self.data_loader.n_task):
            self.class_offset = self.get_class_offset(task_p)
            self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
            for epoch in range(self.data_loader.n_epoch):
                if epoch < 2:
                    bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
                    for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(task_p, epoch=epoch)):
                        if self.cuda:
                            batch_inputs = batch_inputs.cuda()
                            batch_labels = batch_labels.to(torch.int64).cuda()
                        loss, acc = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
                        losses.append(loss)
                        train_acc.append(acc)
                        bar.update(batch_labels.size(0))
                    bar.close()
                    print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
                    self.metric(task_p)
                    self.logger.info(f'task {task_p} has trained over ,eval result is shown below.')
                elif epoch == 2:
                    bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
                    for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(task_p, epoch=epoch)):
                        if self.cuda:
                            batch_inputs = batch_inputs.cuda()
                            batch_labels = batch_labels.to(torch.int64).cuda()
                        self.model.cluster_statics(batch_inputs, batch_labels, task_p)
                        bar.update(batch_labels.size(0))
                    self.model.cluster.compute_trans_mat_pinv()
                    bar.close()
                    self.logger.info(f'task {task_p} cluster has been over')







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
    parser.add_argument('--n_epoch', default=3, type=int, help='number of epochs')
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

    trainer = Decoder_train(data_loader, model, args, logger)
        #base_train(data_loader, model, args, logger)
    trainer.train()

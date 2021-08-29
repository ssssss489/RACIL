



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


def cluster_loss(q):
    # Unsupervised Deep Embedding for Clustering Analysis
    # q = torch.softmax(q, dim=1)
    # q = torch.exp(q)
    # if torch.min(q) < 0:
    #     raise ValueError('cluster distances must be greater than 0')
    # f = torch.sum(q, dim=0).view([1, -1])
    # t = q ** 2 / f
    # p = (t / torch.sum(t, dim=1).view([-1, 1])).detach()
    # l = torch.sum(p * torch.log(p / q))
    if q.shape[1] == 1:
        p = q.abs() ** 0.5
        l = torch.sum((p.detach() - q.abs()) ** 2)
    else:
        # q = torch.softmax(q, dim=1)
        p = torch.argmax(q, dim=1)
        l = torch.nn.CrossEntropyLoss()(q, p)

    return l


def enhance_result(x):
    threshold = 0.4
    # return torch.clip_(x - threshold, 0.0, 1.0 - threshold) ** 0.5 / 0.5 ** 0.5
    return torch.where(x > threshold, torch.ones_like(x), torch.zeros_like(x))
# def compute_cluster_distance(x, c):
#     # Unsupervised Deep Embedding for Clustering Analysis
#     l2 = torch.sqrt(((x.view([x.shape[0],x.shape[1],1]) - c.view([1, c.shape[0], c.shape[1]]))**2).sum(dim=1))
#     t = 1 / (l2 + 1)
#     d = t / torch.sum(t, dim=1).view([-1, 1])
#     return d



class model(torch.nn.Module):
    def __init__(self, data, args):
        super(model, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.n_classes = models_setting[data].classifier[-1]

        self.encoder = MLP_Encoder(models_setting[data].encoder)
        self.decoder = MLP_Decoder(models_setting[data].decoder)
        self.classifier = MLP_Classifier(models_setting[data].classifier)
        self.projector = Feature_Project(2, models_setting[data].encoder[-1])

        self.buffer = Buffer(args.n_task, args.n_memory, self.n_classes)
        self.task_decoders = {0: self.decoder}
        self.task_projectors = {0: self.projector}

        self.classifier_loss_fn = torch.nn.CrossEntropyLoss()
        self.cluster_loss_fn = cluster_loss
        self.decoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))
        self.encoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))
        #self.encoder_loss_fn = lambda x, y: torch.sum(y * torch.log(y / x))# torch.nn.KLDivLoss()(torch.log(x), y)

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
        self.projector.init_centers(feature)
        class_feature, mu_feature, project_dis = self.projector(feature)
        y = self.classifier(class_feature * 11)
        if with_details:
            return y, class_feature, mu_feature, project_dis, feature
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

    def augment_feature(self, class_feature, projector):
        pseudo_mu_feature, pseudo_project_dis = projector.pseudo_project(class_feature.shape[0])
        pseudo_class_feature = sub_project(class_feature, pseudo_mu_feature)
        mod_pmf_sq = torch.sum(pseudo_mu_feature ** 2, dim=-1, keepdim=True)
        mod_pcf_sq = torch.sum(pseudo_class_feature ** 2, dim=-1, keepdim=True)
        t = torch.sqrt((1 - mod_pcf_sq) / mod_pmf_sq)
        pseudo_feature = pseudo_mu_feature * t + pseudo_class_feature
        return pseudo_feature, pseudo_class_feature

    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0
        # torch.autograd.set_detect_anomaly(True)

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            self.projector = deepcopy(self.projector)
            self.task_projectors[task_p] = self.projector
            self.projector.center_vectors = None

            self.decoder = deepcopy(self.decoder)
            self.task_decoders[task_p] = self.decoder
            self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.lr)

        class_offset = get_class_offset(task_p)
        logits, class_feature, mu_feature, project_dis, feature = self.forward(inputs, with_details=True)

        logits, labels = compute_output_offset(logits, labels, class_offset)
        loss += self.classifier_loss_fn(logits, labels.detach())

        decoder_outputs = self.encoder_to_decoder(inputs)
        loss += self.decoder_loss_fn(decoder_outputs, inputs) * 10

        loss += self.cluster_loss_fn(project_dis)

        if task_p > 0:
            pt = np.random.choice(self.observed_tasks[:-1])
            pt_inputs, pt_labels, pts = self.buffer.get_buffer_samples([pt], labels.shape[0])
            pt_class_offset = get_class_offset(pts)

            pt_logits, pt_class_feature, pt_mu_feature, pt_project_dis, pt_feature = self.forward(pt_inputs, with_details=True)
            loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
            loss += 1.0 * loss_

            pt_projector = self.task_projectors[pt]
            pseudo_pt_feature, pseudo_pt_class_feature = self.augment_feature(class_feature, pt_projector)
            # pseudo_pt_inputs = pt_decoder(pseudo_pt_feature * 11).view([-1,28,28])
            pseudo_pt_logits = self.classifier(pseudo_pt_class_feature * 11)
            loss_ = self.classifier_loss_fn(*compute_output_offset(pseudo_pt_logits, labels.data.detach(), pt_class_offset))
            # loss += loss_ * 1.0

            pt_decoder = self.task_decoders[pt]

            pseudo_pt_inputs = pt_decoder(pt_feature * 11)
            loss_ = self.decoder_loss_fn(pseudo_pt_inputs, pt_inputs)
            loss += loss_ * 100.0

            pt_feature = torch.cat([pt_feature, pseudo_pt_feature], dim=0)
            t_feature = self.decoder_to_encoder(pt_feature, pt_decoder)
            loss_ = self.encoder_loss_fn(pt_feature, t_feature)
            # loss += loss_ * 100.0

        decoder_outputs_numpy = decoder_outputs.view([-1,28,28]).data.cpu().numpy()
        inputs_numpy = inputs.data.cpu().numpy()
        # loss += torch.mean((class_feature) ** 2)# * 0.1

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        # with torch.autograd.detect_anomaly():
        loss.backward(retain_graph=True)

        # if task_p > 0:
        #     gradient_project(dict(self.classifier.named_parameters()), ref_classifier_param_dicts)
        #     gradient_project(dict(self.feature_net.named_parameters()), ref_feature_net_param_dicts)
        # self.projector.update_centers(feature.data.detach(), project_dis.data.detach())
        self.projector.update_centers(class_feature.data.detach(), project_dis.data.detach(), labels.detach(), self.n_classes)

        if task_p == 0:
            self.classifier_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        else:
            self.classifier_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

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

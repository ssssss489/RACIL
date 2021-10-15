



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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def draw_tSNE(feature, mu, k_assign):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(to_numpy(torch.cat([feature, mu], 0)))
    # plt.plot(data[:, 0], data[:, 1])
    mu = data[-int(mu.shape[0]):]
    data = data[:-int(mu.shape[0])]
    plt.scatter(data[:, 0], data[:, 1], c=to_numpy(k_assign), cmap='rainbow')
    plt.scatter(mu[:, 0], mu[:, 1], marker='D')
    # plt.scatter(data[int(feature.shape[0]):, 0], data[int(feature.shape[0]):, 1], )
    # plt.colorbar()
    plt.show()
    return





class Prototype(torch.nn.Module):
    def __init__(self, n_top_eig, n_class, hidden_size,
                 scale_upper_bound=1, scale_lower_bound=10, init_mu=2, gama=16, n_topk=128):
        super(Prototype, self).__init__()
        self.n_top_eig = n_top_eig
        self.n_center = 2 ** n_top_eig
        self.n_class = n_class
        self.hidden_size = hidden_size

        self.scale_upper_bound = scale_upper_bound
        self.scale_lower_bound = scale_lower_bound
        self.init_mu = init_mu
        self.gama = gama
        self.n_topk = n_topk
        self.init_parameters()

        self.class_multivar_normals = []

    def init_parameters(self):

        self.M = torch.nn.Parameter(torch.randn([self.n_class, self.hidden_size, self.hidden_size]) * 0.1)
        self.M_eig = None
        self.M_evc = None

        self.class_mu = torch.zeros([self.n_class, self.n_center, self.hidden_size]).cuda()
        self.class_cov = torch.zeros([self.n_class, self.n_center, self.hidden_size, self.hidden_size]).cuda()
        self.class_n_sample = torch.zeros([self.n_class, self.n_center]).cuda()
        self.cuda()

    def get_sym_M(self):
        M = self.M.triu()
        diag = M * torch.eye(self.hidden_size).cuda().unsqueeze(0)
        M = M + torch.nn.ReLU()(- (diag - 1e-3)).detach()
        M = torch.matmul(M, M.transpose(-1,-2))#  + torch.eye(self.hidden_size).cuda() * 1e-3
        return M

    def forward(self, x, y):
        y_ = one_hot(y, self.n_class).unsqueeze(-1)
        class_M = y_.unsqueeze(-1) * self.get_sym_M().unsqueeze(0)
        feature = torch.matmul(x.view(x.shape[0], 1, 1, x.shape[1]), class_M).squeeze(2)
        feature = feature.sum(1)
        return feature

    def multi_eig_M(self, x, y):
        if self.M_eig is None:
            sym_M = self.get_sym_M()
            M_eig, M_evc = torch.linalg.eig(sym_M)
            self.M_eig = M_eig.float()
            self.M_evc = M_evc.float()

        y_ = one_hot(y, self.n_class).unsqueeze(-1)
        class_M_eig = y_ * self.M_eig.unsqueeze(0)
        class_M_evc = y_.unsqueeze(-1) * self.M_evc.unsqueeze(0)

        weight = x.view(x.shape[0], 1, 1, x.shape[1]).matmul(class_M_evc).squeeze(2) * class_M_eig
        feature = weight.unsqueeze(2).matmul(class_M_evc.transpose(-1,-2))
        feature = feature.sum(1).squeeze(1)
        weight = weight.sum(1)
        return weight, feature


    def scan_samples(self, x, y, type):
        logits, features = self.multi_eig_M(x, y)
        x = features
        logits = logits[:, :self.n_top_eig]
        y_ = one_hot(y, self.n_class).view([y.shape[0], self.n_class, 1, 1])

        # k_ = one_hot(logits.max(-1)[1], self.n_center).view([y.shape[0], 1, self.n_center, 1])
        k_ = one_hot(binary_decimal((logits > 0).float()).long(), self.n_center).view([y.shape[0], 1, self.n_center, 1])

        if type=='mu':
            x_ = x.view([x.shape[0], 1, 1, x.shape[1]]) * y_ * k_
            self.class_mu += x_.sum(0)
            self.class_n_sample += (y_ * k_).sum(0).squeeze(-1)
        elif type=='cov':
            diff = x.view([x.shape[0], 1, 1, x.shape[1]]) - self.class_mu.unsqueeze(0)
            diff_ = diff * y_ * k_
            self.class_cov += torch.matmul(diff_.unsqueeze(-1), diff_.unsqueeze(-2)).sum(0)


    def compute_cov(self):
        self.class_cov = self.class_cov / (self.class_n_sample.unsqueeze(-1).unsqueeze(-1) + 1e-6)

    def compute_mu(self):
        self.class_mu = self.class_mu / (self.class_n_sample.unsqueeze(-1) + 1e-6)

    def set_multivar_normals(self):
        for cls in range(self.n_class):
            t = []
            for center in range(self.n_center):
                t.append(torch.distributions.multivariate_normal.MultivariateNormal(self.class_mu[cls, center],
                                                                                    self.class_cov[cls, center]
                                                                                    + torch.eye(self.hidden_size).cuda() * 1e-3))
            self.class_multivar_normals.append(t)

    def sample(self, y):
        y_ = one_hot(y, self.n_class)
        class_n_sample = y_.sum(0)
        new_x, new_y = [], []
        for cls in range(self.n_class):
            if class_n_sample[cls] > 0:
                for center in range(self.n_center):
                    new_x.append(self.class_multivar_normals[cls][center].sample([int(class_n_sample[cls])]))
                    new_y.append(torch.empty([int(class_n_sample[cls])]).fill_(cls))
        new_x = torch.cat(new_x, dim=0).cuda()
        new_y = torch.cat(new_y, dim=0).long().cuda()
        return new_x, new_y


class model(torch.nn.Module):
    def __init__(self, data, args):
        super(model, self).__init__()
        self.data_name = data
        self.n_tasks = args.n_tasks
        self.n_classes = parameters[data].classifier[-1]

        self.encoder = MLP_Encoder(parameters[data].encoder)
        self.classifier = MLP_Classifier(parameters[data].classifier)
        self.Prototypes = []
        self.Decoders = []
        for _ in range(self.n_tasks):
            self.Prototypes.append(Prototype(1, self.n_classes, parameters[data].encoder[-1]))
            self.Decoders.append(MLP_Decoder(parameters[data].decoder).cuda())

        self.buffer = Buffer(args.n_tasks, args.n_memory, self.n_classes)

        self.classifier_loss_fn = torch.nn.CrossEntropyLoss()
        self.prototype_loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
        self.decoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))

        self.lr = args.lr
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.decoder_optimizer = None
        self.prototype_optimizer = None

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

    def compute_prototype(self,inputs, labels, get_class_offset, task_p, type):
        with torch.no_grad():
            class_offset = get_class_offset(task_p)
            _, feature = self.forward(inputs, with_details=True)
            _, labels = compute_output_offset(None, labels, class_offset)
            self.Prototypes[task_p].scan_samples(feature, labels, type)


    def learn_step(self, inputs, labels, get_class_offset, task_p):
        decoder = self.Decoders[task_p]
        prototype = self.Prototypes[task_p]
        decoder.train()
        decoder.zero_grad()
        prototype.train()
        prototype.zero_grad()
        loss = 0

        class_offset = get_class_offset(task_p)
        _, labels = compute_output_offset(None, labels, class_offset)
        _, feature = self.forward(inputs, with_details=True)
        prototype_feature = prototype(feature, labels)
        labels_ = one_hot(labels, self.n_classes)
        feature_labels = torch.cat([prototype_feature, labels_], axis=-1)
        decoder_outputs = decoder(feature_labels)
        loss += self.decoder_loss_fn(decoder_outputs, inputs) * 100

        decoder_outputs_numpy = decoder_outputs.view([-1,28,28]).data.cpu().numpy()
        inputs_numpy = inputs.data.cpu().numpy()

        # if task_p > 0:
        #     pt = np.random.choice(self.observed_tasks[:-1])
        #     prototype = self.Prototypes[pt]
        #     decoder = self.task_decoders[task_p-1]
        #     pt_class_offset = get_class_offset(pt)
        #     _, pt_labels = compute_output_offset(None, labels, pt_class_offset)
        #     pt_feature, pt_labels = prototype.sample(pt_labels)
        #     pt_inputs = decoder(pt_feature).view([-1, 28, 28])
        #     pt_decoder_outputs = self.decoder(pt_feature)
        #     loss_ = self.decoder_loss_fn(pt_decoder_outputs, pt_inputs)
        #     # loss += loss_

        if (torch.isnan(loss).sum() > 0):
            print("here!")

        loss.backward(retain_graph=True)

        self.decoder_optimizer.step()
        self.prototype_optimizer.step()

        return float(loss.item())


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0

        if task_p not in self.observed_tasks:
            decoder = self.Decoders[task_p]
            if len(self.observed_tasks) != 0:
                pre_decoder = self.Decoders[self.observed_tasks[-1]]
                decoder.load_state_dict(dict(pre_decoder.named_parameters()))
            self.decoder_optimizer = torch.optim.SGD(self.Decoders[task_p].parameters(), lr=self.lr)
            self.prototype_optimizer = torch.optim.SGD(self.Prototypes[task_p].parameters(), lr=self.lr)
            self.observed_tasks.append(task_p)

        class_offset = get_class_offset(task_p)
        logits, feature = self.forward(inputs, with_details=True)
        logits, labels = compute_output_offset(logits, labels, class_offset)

        classifier_loss = self.classifier_loss_fn(logits, labels)
        loss += classifier_loss

        if (torch.isnan(loss).sum() > 0):
            print("here!")

        if task_p > 0:
            pass
            # pt = np.random.choice(self.observed_tasks[:-1])
            # pt_inputs, pt_labels, pts = self.buffer.get_buffer_samples([pt], labels.shape[0])
            # pt_class_offset = get_class_offset(pts)
            #
            # pt_logits, pt_feature = self.forward(pt_inputs, with_details=True)
            # loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
            # loss += 1.0 * loss_

            pt = np.random.choice(self.observed_tasks[:-1])
            prototype = self.Prototypes[pt]
            pt_decoder = self.Decoders[pt]
            pt_class_offset = get_class_offset(pt)
            pt_feature, pt_labels = prototype.sample(labels)
            pt_labels_ = one_hot(pt_labels, self.n_classes)
            pt_feature_labels = torch.cat([pt_feature, pt_labels_], dim=-1)
            pt_inputs = pt_decoder(pt_feature_labels).view([-1,28,28])
            pt_logits, new_pt_feature = self.forward(pt_inputs, with_details=True)
            loss_ = self.classifier_loss_fn(*compute_output_offset(pt_logits, pt_labels.data.detach(), pt_class_offset))
            loss += 1.0 * loss_

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        loss.backward(retain_graph=True)

        self.classifier_optimizer.step()
        self.encoder_optimizer.step()
        # self.decoder_optimizer.step()


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

    def show_feature_distrubtion(self, inputs, labels, p_task, cls):
        _, feature = self.forward(inputs, with_details=True)
        cls_feature = feature[labels == cls]
        cls_mu = self.Prototypes[p_task].class_mu[cls]
        logits, cls_feature = self.Prototypes[p_task].multi_eig_M(cls_feature, torch.LongTensor([cls]).cuda().repeat(cls_feature.shape[0]))
        logits = logits[:, :self.Prototypes[p_task].n_top_eig]
        # k_assign = logits.max(-1)[1]
        k_assign = binary_decimal((logits > 0).float()).long()
        draw_tSNE(cls_feature, cls_mu, k_assign)




class Decoder_train(base_train):
    def __init__(self, data_loader, model, args, logger):
        super(Decoder_train, self).__init__(data_loader, model, args, logger)
        self.n_epoch_train = args.n_epoch_train
        self.n_epoch_learn = args.n_epoch_learn
        self.total = data_loader.task_n_sample[0]
        self.task_p =None
        self.epoch = None
        self.cuda = True
        self.train_process = [self.train_encoder_classifier] * self.n_epoch_train \
                            +[self.train_decoder_prototype] * self.n_epoch_learn \
                            +[lambda : self.compute_prototype('mu')] \
                            +[lambda : self.compute_prototype('cov')] \
                            +[self.show_tSNE]


    def train_encoder_classifier(self):
        losses, train_acc = [], []
        self.logger.info(f'task {self.task_p} is beginning to train.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            loss, acc = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset, self.task_p)
            losses.append(loss)
            train_acc.append(acc)
            bar.update(batch_labels.size(0))
        bar.close()
        print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
        self.metric(self.task_p)

    def train_decoder_prototype(self):
        self.logger.info(f'task {self.task_p} is beginning to learn.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        losses = []
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            loss = self.model.learn_step(batch_inputs, batch_labels, self.get_class_offset, self.task_p)
            losses.append(loss)
            bar.update(batch_labels.size(0))
        bar.close()
        print(f'    loss = {np.mean(losses)}')

    def compute_prototype(self, type):
        self.logger.info(f'task {self.task_p} is beginning to compute prototype {type}.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            self.model.compute_prototype(batch_inputs, batch_labels, self.get_class_offset, self.task_p, type)
            bar.update(batch_labels.size(0))
        bar.close()
        if type == 'mu':
            self.model.Prototypes[self.task_p].compute_mu()
        else:
            self.model.Prototypes[self.task_p].compute_cov()
            self.model.Prototypes[self.task_p].set_multivar_normals()

    def show_tSNE(self):
        self.model.show_feature_distrubtion(
            self.data_loader.task_datasets[self.task_p].train_images[:5000].cuda(),
            self.data_loader.task_datasets[self.task_p].train_labels[:5000].cuda(),
            self.task_p,
            4)

    def train(self):
        self.metric(-1)
        for self.task_p in range(self.data_loader.n_tasks):
            self.class_offset = self.get_class_offset(self.task_p)
            self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
            for self.epoch, func in enumerate(self.train_process) :
                func()
            self.logger.info(f'task {self.task_p} decoder has learned over')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--result_path', default='result/', help='input directory')
    parser.add_argument('--dataset', default='mnist_rot_10', help='learn task')
    parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
    parser.add_argument('--samples_per_task', type=int, default=-1, help='training samples per task (all if negative)')
    parser.add_argument('--seed', default=5, type=int, help='random seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_epoch_train', default=1, type=int, help='number of epochs for train')
    parser.add_argument('--n_epoch_learn', default=3, type=int, help='number of epochs for train')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU')
    parser.add_argument('--n_memory', default=100, help='number of memories per task')
    args = parser.parse_args()

    args.n_epochs = args.n_epoch_train + args.n_epoch_learn + 2

    torch.manual_seed(args.seed)

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))

    task = args.dataset.split('_')[0]
    # model = MLP(task, args)
    model = model(task, args)

    data_loader = task_data_loader(task_datasets, args)

    trainer = Decoder_train(data_loader, model, args, logger)
        #base_train(data_loader, model, args, logger)
    trainer.train()

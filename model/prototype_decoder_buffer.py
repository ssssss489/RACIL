
from model.base_model import *
from train.base_train import *
from model.buffer import Buffer
from scipy.special import comb
from itertools import combinations
from scipy import stats

def prototype_update(buffer, xs, ys, probs):
    new_instance_size = np.sum([i.shape[0] for i in ys])
    new_prob = torch.cat([buffer.memory_keep_prob] + probs, dim=0)
    new_x = torch.cat([buffer.memory_inputs] + xs, dim=0)
    new_y = torch.cat([buffer.memory_labels] + ys, dim=0)
    prob_topk, idx = torch.topk(new_prob, k=buffer.n_memories, dim=0)
    buffer.memory_keep_prob = prob_topk
    buffer.memory_inputs = new_x[idx]
    buffer.memory_labels = new_y[idx]
    buffer.current_index += new_instance_size
    buffer.current_index = min(buffer.current_index, buffer.n_memories)

def prob_retrieve(buffer, n_retrieve, excl_indices=None):
    filled_indices = np.arange(buffer.current_index)
    if excl_indices is not None:
        excl_indices = list(excl_indices)
    else:
        excl_indices = []
    valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    n_retrieve = min(n_retrieve, valid_indices.shape[0])
    prob = buffer.memory_keep_prob[valid_indices]
    prob = to_numpy(torch.softmax(prob, dim=0))
    indices = torch.from_numpy(np.random.choice(valid_indices, n_retrieve, replace=False)).long()

    x = buffer.memory_inputs[indices]
    y = buffer.memory_labels[indices]
    return x.cuda(), y.cuda().long()


class TreePartition():
    def __init__(self, data, min_size):
        self.data = data #2d
        self.partition_data = []
        self.min_size_cluster = min_size

    def partition(self, data):
        if len(data) < self.min_size_cluster:
            return [data]
        pos = (data >= 0).sum(0)
        neg = len(data) - pos
        partition_idx = np.argmin(np.abs(pos - neg))
        pos_idx = data[:, partition_idx] >= 0
        neg_idx = data[:, partition_idx] < 0

        pos_part_data = data[pos_idx]
        neg_part_data = data[neg_idx]
        if len(pos_part_data) == 0 or len(neg_part_data) == 0:
            return [data]

        pos_part_data = self.partition(pos_part_data)
        neg_part_data = self.partition(neg_part_data)

        return pos_part_data + neg_part_data

    def fit(self):
        self.partition_data = self.partition(self.data)

    def compute_statistic_parameter(self, data_list):
        mus, covs = [], []
        for cluster in data_list:
            mus.append(np.mean(cluster))
            covs.append(np.cov(cluster))
        return mus, covs

    def evaluate_normal(self, data):
        skew = stats.skew(data, axis=0, bias=False)
        kurtosis = stats.kurtosis(data, axis=0, bias=False)
        # normal_test = stats.normaltest(data, axis=0)

        gamma_test = [stats.kstest(data[i], 'gamma', args=stats.gamma.fit(data[i])) for i in range(data.shape[1])]
        return skew, kurtosis, gamma_test



class Prototype(torch.nn.Module):
    def __init__(self, n_top_eigs, n_classes, hidden_size, input_dims):
        super(Prototype, self).__init__()
        self.n_top_eigs = n_top_eigs
        self.select_dims = 3
        self.n_centers = int(comb(n_top_eigs, self.select_dims)) * 2 ** self.select_dims   #2 * n_top_eigs
        self.centers_dict = {c :i for i, c in enumerate(combinations(range(n_top_eigs), self.select_dims)) }
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.input_dims = input_dims
        self.statistic_dims = n_top_eigs

        self.init_parameters()

        self.class_multivar_normals = []

    def init_parameters(self):

        self.M = torch.nn.Parameter(torch.randn([self.n_classes, self.hidden_size, self.hidden_size]) * 0.1)
        self.bias = torch.nn.Parameter(torch.zeros([self.n_classes, self.hidden_size]))

        self.M_eig_sqrt = None
        self.M_evc = None

        self.class_mu = torch.zeros([self.n_classes, self.n_centers, self.statistic_dims])
        self.class_cov = torch.zeros([self.n_classes, self.n_centers, self.statistic_dims, self.statistic_dims])
        self.class_n_samples = torch.zeros([self.n_classes, self.n_centers])

        self.cuda()

    def get_sym_M(self):
        M = self.M.triu()
        diag = M * torch.eye(self.hidden_size).cuda().unsqueeze(0)
        M = M + torch.nn.ReLU()(- (diag - 1e-3)).detach()
        M = torch.matmul(M, M.transpose(-1,-2)) + torch.eye(self.hidden_size).cuda() * 1e-3
        return M

    def forward(self, x, y):
        class_M = torch.index_select(self.get_sym_M(), dim=0, index=y)
        class_bias = torch.index_select(self.bias, dim=0, index=y)
        x = x - class_bias
        feature = torch.matmul(x.unsqueeze(1), class_M)
        weight_dis = torch.matmul(feature, x.unsqueeze(-1))
        return feature.squeeze(1), x.mean(0), weight_dis.squeeze() / self.hidden_size

    def multi_eig_M(self, x, y):
        if self.M_eig_sqrt is None:
            sym_M = self.get_sym_M()
            M_eig, M_evc = torch.linalg.eig(sym_M)
            self.M_eig_sqrt = M_eig.float().sqrt()
            self.M_evc = M_evc.float()

        class_M_eig_sqrt = torch.index_select(self.M_eig_sqrt, dim=0, index=y)[:, :self.statistic_dims]
        class_M_evc = torch.index_select(self.M_evc, dim=0, index=y)[:, :, :self.statistic_dims]
        class_bias = torch.index_select(self.bias, dim=0, index=y)

        x = x - class_bias
        weight = x.unsqueeze(1).matmul(class_M_evc) * class_M_eig_sqrt.unsqueeze(1)
        trans_feature = (weight * class_M_eig_sqrt.unsqueeze(1)).matmul(class_M_evc.transpose(-1, -2))
        return weight.squeeze(1), trans_feature.squeeze(1)

    def compute_centers(self, x, y):
        weights, features = self.multi_eig_M(x, y)
        top_weigths = weights[:, :self.n_top_eigs].cpu()
        features = features.cpu()

        #max_idx = top_weigths.abs().max(1)[1].cpu()
        # k = torch.where(top_weigths.gather(dim=1, index=max_idx.unsqueeze(1)).squeeze() > 0, max_idx, max_idx + self.n_top_eigs)
        top2_idx = top_weigths.abs().topk(self.select_dims, dim=1)[1].cpu().sort()[0]
        ms = binary_decimal((top_weigths.gather(dim=1, index=top2_idx) > 0).float()).long()

        k = torch.Tensor([self.centers_dict[tuple(idx.numpy())] * (2 ** self.select_dims) + m for idx, m in zip(top2_idx, ms)]).long()

        return k, weights # features #

    def compute_parameters(self, x, y, type):
        k, features = self.compute_centers(x, y)

        for cls in set(to_numpy(y)):
            cls_idx = (y == cls).nonzero().squeeze(-1)
            cls_k = k[cls_idx]
            for cen in set(to_numpy(cls_k)):
                cls_cen_idx = cls_idx[(cls_k == cen).nonzero().squeeze(-1)]
                f = features[cls_cen_idx][:, :self.statistic_dims].cpu()
                if type == 'mu':
                    self.class_mu[cls, cen] += f.sum(0)
                    self.class_n_samples[cls, cen] += f.shape[0]
                elif type == 'cov':
                    diff = f - self.class_mu[cls, cen].unsqueeze(0)
                    diff = diff.cuda()
                    self.class_cov[cls, cen] += torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2)).sum(0).cpu()
        return k


    def compute_cov(self, classes):
        self.class_cov[classes] = self.class_cov[classes] / (self.class_n_samples[classes].unsqueeze(-1).unsqueeze(-1) + 1e-6)

    def compute_mu(self, classes):
        self.class_mu[classes] = self.class_mu[classes] / (self.class_n_samples[classes].unsqueeze(-1) + 1e-6)

    def set_multivar_normals(self, classes, scale_cov=1):
        for cls in classes:
            t = []
            for cen in range(self.n_centers):
                if self.class_n_samples[cls, cen] == 0 :
                    t.append(None)
                    continue
                t.append(torch.distributions.multivariate_normal.MultivariateNormal(self.class_mu[cls, cen],
                                                                                    self.class_cov[cls, cen] * scale_cov + \
                                                                                    torch.eye(self.statistic_dims) * 1e-3))
            self.class_multivar_normals.append(t)

    def deprecated_sample(self, y):
        y_ = one_hot(y, self.n_classes)
        class_n_samples = y_.sum(0)
        new_x, new_y = [], []
        for cls in range(self.n_classes):
            if class_n_samples[cls] > 0:
                for center in range(self.n_centers):
                    new_x.append(self.class_multivar_normals[cls][center].sample([int(class_n_samples[cls])]))
                    new_y.append(torch.empty([int(class_n_samples[cls])]).fill_(cls))
                # center = np.random.randint(0,self.n_centers)
                # new_x.append(self.class_multivar_normals[cls][center].sample([int(class_n_samples[cls])]))
                # new_y.append(torch.empty([int(class_n_samples[cls])]).fill_(cls))
        new_x = torch.cat(new_x, dim=0).cuda()
        new_y = torch.cat(new_y, dim=0).long().cuda()
        return new_x, new_y

    def compute_instance_prob(self, x, y):
        k, features = self.compute_centers(x, y)
        f = features[:, :self.statistic_dims].cpu()
        log_probs = []
        for i, (j, cls) in enumerate(zip(to_numpy(k), to_numpy(y))):
            log_prob = (self.class_n_samples[cls].sum() / self.class_n_samples[cls][j]).log()
            normal_dist = self.class_multivar_normals[cls][j]
            log_prob += normal_dist.log_prob(f[i])  #.exp()
            log_probs.append(log_prob)
        return  torch.Tensor(log_probs)

    def depreciated_compute_instance_prob(self, x, y):
        k, features = self.compute_centers(x, y)
        f = features[:, :self.statistic_dims].cpu()
        log_cps, log_nps = torch.zeros([f.shape[0], self.n_centers]), torch.zeros([f.shape[0], self.n_centers])
        for i, cls in enumerate(to_numpy(y)):
            log_cps[i] = (self.class_n_samples[cls] / self.class_n_samples[cls].sum()).log()
            log_cps[i][self.class_n_samples[cls] == 0] = 0 # -1e10
            for j, norm_dist in enumerate(self.class_multivar_normals[cls]):
                if norm_dist:
                    log_nps[i, j] = norm_dist.log_prob(f[i])
        log_probs_tmp = log_cps * log_nps
        log_probs = log_probs_tmp.prod(1)
        return log_probs










class prototype_decoder(base_model):
    def __init__(self, args, model_type):
        super(prototype_decoder, self).__init__(args)
        self.decoder_update = args.decoder_update
        self.eps_mem_batch = args.eps_mem_batch
        self.buffer = Buffer(self.n_tasks, args.n_memories, self.n_classes, parameters[self.data_name].input_dims)
        self.Decoders = []
        self.model_type = model_type
        if model_type == 'mlp':
            self.encoder = MLP_Encoder(parameters[self.data_name].encoder)
            self.classifier = MLP_Classifier(parameters[self.data_name].classifier)

            for _ in range(self.n_tasks):
                self.Decoders.append(MLP_Decoder(parameters[self.data_name].decoder, parameters[self.data_name].input_dims).cuda())

        elif model_type == 'resnet':
            self.encoder = ResNet18_Encoder(parameters[self.data_name].input_dims,
                                            parameters[self.data_name].nf,
                                            parameters[self.data_name].pool_size)
            self.classifier = MLP_Classifier(parameters[self.data_name].classifier, [])

            for _ in range(self.n_tasks):

                self.Decoders.append(ResNet18_Decoder(parameters[self.data_name].pool_size,
                                                      parameters[self.data_name].nf,
                                                      parameters[self.data_name].hidden_size,
                                                      parameters[self.data_name].input_dims).cuda())

        self.prototypes = Prototype(args.top_n_eigs, self.n_classes, parameters[self.data_name].hidden_size,
                                    parameters[self.data_name].input_dims)

        def ladder_loss(ens, des, weights=[1.0, 0.0, 0.0, 0.00, 0.00, 0.0]):
            if isinstance(weights, int) or isinstance(weights, float):
                weights = [weights] * len(ens)
            loss = 0
            for en, de, w in zip(ens, des, weights):
                loss += torch.nn.MSELoss()(en, de) * w
            return loss
        self.decoder_loss_fn = ladder_loss

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.decoder_lr = args.lr_decoder
        self.decoder_optimizer = None
        self.prototype_optimizer = torch.optim.SGD(self.prototypes.parameters(), lr=args.lr_decoder)

        self.decoder_loss_weight = args.decoder_loss_weight
        self.weight_l2_loss_weight = args.weight_l2_loss_weight
        self.sub_bias_weight = args.sub_bias_weight


        self.observed_tasks = []
        self.cuda()

        self.show_all_parameters()

    def forward(self, x, with_details=False):
        if self.model_type == 'mlp':
            cls_feature = self.encoder(x)
            hidden_feature = [x, cls_feature]
        elif self.model_type == 'resnet':
            cls_feature, hidden_feature = self.encoder(x, with_hidden=True)
        y = self.classifier(cls_feature)
        if with_details:
            return y, hidden_feature
        return y


    def compute_prototype(self,inputs, labels, type):
        with torch.no_grad():
            _, en_features = self.forward(inputs, with_details=True)
            self.prototypes.compute_parameters(normalize(en_features[-1]), labels, type)


    def compute_instance_prob(self, inputs, labels):
        with torch.no_grad():
            _, en_features = self.forward(inputs, with_details=True)
            probs = self.prototypes.compute_instance_prob(normalize(en_features[-1]), labels)
            k, weights = self.prototypes.compute_centers(normalize(en_features[-1]), labels)
            return probs, k, weights.cpu()


    def train_step(self, inputs, labels, get_class_offset, task_p, classifier_train_flag=True):
        self.train()
        self.zero_grad()
        decoder = self.Decoders[task_p]
        decoder.train()
        decoder.zero_grad()

        loss = 0

        if task_p not in self.observed_tasks:
            if len(self.observed_tasks) != 0:
                last_decoder = self.Decoders[self.observed_tasks[-1]]
                decoder.load_state_dict(dict(last_decoder.named_parameters()), strict=False)
            self.decoder_optimizer = torch.optim.SGD(self.Decoders[task_p].parameters(), lr=self.decoder_lr)
            self.observed_tasks.append(task_p)

        logits, en_features = self.forward(inputs, with_details=True)
        prototype_feature, sub_bias_mean, M_weight_l2 = self.prototypes(normalize(en_features[-1]), labels)
        en_features = en_features[:-1]
        decoder_outputs, de_features = decoder(en_features + [prototype_feature])

        def greater_mask(a, theshold=1.0):
            idx = (a < theshold).float()
            b = torch.empty_like(a).fill_(theshold)
            return a * idx + b * (1 - idx)

        M_weight_l2_loss = - greater_mask(M_weight_l2).mean()
        sub_bias_mean_loss = sub_bias_mean.square().mean()
        decoder_loss = self.decoder_loss_fn(en_features, de_features)
        classifier_loss = self.classifier_loss_fn(logits, labels)

        loss += decoder_loss * self.decoder_loss_weight\
                + sub_bias_mean_loss * self.sub_bias_weight\
                + M_weight_l2_loss * self.weight_l2_loss_weight

        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        pt_classifier_loss, pt_sub_bias_mean_loss, pt_M_weight_l2_loss, pt_decoder_loss, pt_acc = torch.zeros(5)
        if task_p > 0:
            pt_inputs, pt_labels = prob_retrieve(self.buffer, self.eps_mem_batch)
            pt_logits, pt_en_features = self.forward(pt_inputs, with_details=True)
            pt_prototype_feature, pt_sub_bias_mean, pt_M_weight_l2 = self.prototypes(normalize(pt_en_features[-1]), pt_labels)
            pt_en_features = pt_en_features[:-1]
            pt_decoder_outputs, pt_de_features = decoder(pt_en_features + [pt_prototype_feature])

            pt_sub_bias_mean_loss = pt_sub_bias_mean.square().mean()
            pt_M_weight_l2_loss = - greater_mask(pt_M_weight_l2).mean()
            pt_decoder_loss = self.decoder_loss_fn(pt_en_features, pt_de_features)
            pt_classifier_loss = self.classifier_loss_fn(pt_logits, pt_labels)

            loss += pt_decoder_loss + pt_sub_bias_mean_loss + pt_M_weight_l2_loss * 0.01
            pt_acc = torch.eq(torch.argmax(pt_logits, dim=1), pt_labels).float().mean()

        # if classifier_train_flag:
        loss += classifier_loss + pt_classifier_loss

        if (torch.isnan(loss).sum() > 0):
            print("here!")

        loss.backward(retain_graph=True)

        # if classifier_train_flag:
        self.classifier_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.prototype_optimizer.step()

        return {'classifier_loss': float(classifier_loss.item()), 'acc':float(acc.item()),
                'decoder_loss': float(decoder_loss.item()), 'sub_bias_loss': float(sub_bias_mean_loss),
                'weight_l2_loss': float(M_weight_l2_loss)}, \
               {'classifier_loss': float(pt_classifier_loss.item()), 'acc': float(pt_acc.item()),
                'decoder_loss': float(pt_decoder_loss.item()), 'sub_bias_loss': float(pt_sub_bias_mean_loss),
                'weight_l2_loss': float(pt_M_weight_l2_loss)}



class prototype_decoder_train(base_train):
    def __init__(self, data_loader, model, args, logger):
        super(prototype_decoder_train, self).__init__(data_loader, model, args, logger)
        self.total = data_loader.task_n_sample[0]
        self.n_epochs_learn = args.n_epochs_learn
        self.task_p =None
        self.epoch = None
        self.scale_cov = args.scale_cov
        # self.train_process = [lambda: [self.train_step(),self.learn_step()]] * (self.n_epochs - 2) \
        #                     +[lambda : self.compute_prototype('mu')] \
        #                     +[lambda : self.compute_prototype('cov')]
        self.train_process = [self.train_step] * (self.n_epochs - 3) \
                            +[lambda : self.compute_prototype('mu')] \
                            +[lambda: self.compute_prototype('cov')] \
                            +[self.save_buffer_instance]
                          #  +[self.prototype_predict]


    def train_step(self):
        current_infos = []
        pt_infos = []
        self.logger.info(f'task {self.task_p} is beginning to train.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            current_info, pt_info = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset,
                                                                        self.task_p, self.epoch >= self.n_epochs_learn)
            current_infos.append(current_info)
            pt_infos.append(pt_info)
            bar.update(batch_labels.size(0))
        bar.close()
        for k in current_infos[0].keys():
            print(f'    {k} = {np.mean([d[k] for d in current_infos])}, pt_{k} = {np.mean([d[k] for d in pt_infos])}')
        self.metric(self.task_p)


    def compute_prototype(self, type):
        self.logger.info(f'task {self.task_p} is beginning to compute prototype {type}.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            self.model.compute_prototype(batch_inputs, batch_labels, type)
            bar.update(batch_labels.size(0))
        bar.close()
        classes = np.arange(*self.get_class_offset(self.task_p))
        if type == 'mu':
            self.model.prototypes.compute_mu(classes)
        else:
            self.model.prototypes.compute_cov(classes)
            self.model.prototypes.set_multivar_normals(classes, self.scale_cov)
            print('    ', to_numpy(self.model.prototypes.class_n_samples[classes]))

    def save_buffer_instance(self):
        self.logger.info(f'task {self.task_p} is beginning to select_prototype_sample.')
        inputs, labels, probs, centers, weights = [], [], [], [], []
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            inputs += [batch_inputs]
            labels += [batch_labels]
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            batch_probs, batch_centers, batch_weights = self.model.compute_instance_prob(batch_inputs, batch_labels)
            probs += [batch_probs]
            centers += [batch_centers]
            weights += [batch_weights]
            bar.update(batch_labels.size(0))
        bar.close()
        labels_ = torch.cat(labels, dim=0)
        centers = torch.cat(centers, dim=0)
        weights = torch.cat(weights, dim=0)[:,:self.model.prototypes.statistic_dims]



        cls = 0

        cls_idx = labels_ == cls
        cluster = TreePartition(to_numpy(weights[cls_idx]), 20)
        cluster.fit()

        select_center = torch.topk(self.model.prototypes.class_n_samples[cls], 6)[1]
        c_idx = centers == select_center[0]
        for i in select_center[1:]:
            c_idx = torch.logical_or(c_idx, centers == i)

        idx = torch.logical_and(labels_ == cls, c_idx)
        # draw_tSNE(weights[idx], self.model.prototypes.class_mu[cls][select_center], centers[idx])

        # imshow(self.model.Decoders[0](
        #     (self.model.prototypes.class_mu[cls] * self.model.prototypes.M_eig_sqrt[cls:cls+1, :self.model.prototypes.statistic_dims].cpu()).matmul(
        #         self.model.prototypes.M_evc[cls, :, :self.model.prototypes.statistic_dims].T.cpu()).cuda())[0][select_center])

        prototype_update(self.model.buffer, inputs, labels, probs)




    def train(self):
        self.metric(-1)
        for self.task_p in range(self.data_loader.n_tasks):
            self.class_offset = self.get_class_offset(self.task_p)
            self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
            for self.epoch, func in enumerate(self.train_process) :
                func()
            self.logger.info(f'task {self.task_p} decoder has learned over')

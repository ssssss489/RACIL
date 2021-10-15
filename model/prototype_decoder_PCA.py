
from model.base_model import *
from train.base_train import *
from model.buffer import Buffer
from scipy.special import comb
from itertools import combinations, product
from scipy import stats
from sklearn.decomposition import PCA
from collections import defaultdict

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

def retrieve(buffer, n_retrieve, excl_indices=None, base_class_n_sample=500):
    # filled_indices = np.arange(buffer.current_index)
    # if excl_indices is not None:
    #     excl_indices = list(excl_indices)
    # else:
    #     excl_indices = []
    # valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
    # n_retrieve = min(n_retrieve, valid_indices.shape[0])
    # prob = buffer.memory_keep_prob[valid_indices]
    # prob = to_numpy(torch.softmax(prob, dim=0))
    # indices = torch.from_numpy(np.random.choice(valid_indices, n_retrieve, replace=False)).long()
    new_index = buffer.current_index + n_retrieve
    indices = np.arange(buffer.current_index, new_index)
    indices = indices % buffer.n_memories
    buffer.current_index = new_index
    if 0 in indices:
        buffer.shuffer()
    indices = buffer.random_idx[indices]
    x = buffer.memory_inputs[indices]
    y = buffer.memory_labels[indices]
    t = buffer.memory_tasks[indices]
    weights = [(base_class_n_sample + 0.0) / buffer.class_n_samples[l] for l in to_numpy(y)]
    return x.cuda(), y.cuda().long(), torch.Tensor(weights).cuda(), t.cuda().long()


class cluster:
    def __init__(self, dims, signs, indices, x):
        self.dims = dims
        self.signs = signs
        self.indices = indices
        self.l2_dis = np.sqrt((x[:,dims] ** 2).sum(1)) * indices
        self.gamma = []
        self.probs = []
        self.size = indices.sum()


class Prototype(torch.nn.Module):
    def __init__(self, n_top_eigs, n_classes, hidden_size, input_dims):
        super(Prototype, self).__init__()
        self.n_top_eigs = n_top_eigs
        self.select_dims = 5

        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.input_dims = input_dims
        self.statistic_dims = n_top_eigs

        self.init_parameters()


    def init_parameters(self):

        self.bias = torch.nn.Parameter(torch.zeros([self.n_classes, self.hidden_size]))

        self.class_pca = [PCA(n_components=self.n_top_eigs) for i in range(self.n_classes)]
        self.class_clusters = {i: None for i in range(self.n_classes)}
        self.class_pca_var = torch.zeros([self.n_classes, self.n_top_eigs])
        self.class_mean = torch.FloatTensor(self.n_classes, self.hidden_size)
        self.class_trans_mat = torch.FloatTensor(self.n_classes, self.hidden_size, self.n_top_eigs)


        self.cuda()

    def forward(self, x, y):
        return x

    def transform(self, x, y):
        class_mean = torch.index_select(self.class_mean, dim=0, index=y).cuda()
        class_trans_mat = torch.index_select(self.class_trans_mat, dim=0, index=y).cuda()
        return (x - class_mean).unsqueeze(1).matmul(class_trans_mat).squeeze()

    def anti_transform(self, x, y):
        class_mean = torch.index_select(self.class_mean, dim=0, index=y.cpu()).cuda()
        class_trans_mat = torch.index_select(self.class_trans_mat, dim=0, index=y.cpu()).cuda()
        return x.unsqueeze(1).matmul(class_trans_mat.transpose(1, 2)).squeeze() + class_mean


    def apply_PCA(self, x, y):
        def sign_partition(x, x_sign, dims, signs, indices):
            if len(dims) == self.select_dims:
                return [cluster(dims, signs, indices, x)]
            idx_x = x[indices]
            var = idx_x.var(0)
            var[dims] = -1e5
            cur_dim = var.argmax()
            dims = dims + [cur_dim]

            pos_signs = signs + [1]
            neg_signs = signs + [0]
            pos_indices = np.logical_and(x_sign[0][:, cur_dim], indices)
            neg_indices = np.logical_and(x_sign[1][:, cur_dim], indices)

            pos_partition = sign_partition(x, x_sign, dims, pos_signs, pos_indices)
            neg_partition = sign_partition(x, x_sign, dims, neg_signs, neg_indices)
            return pos_partition + neg_partition

        classes = set(y)
        for cls in classes:
            cls_idx = (y == cls)
            cls = int(cls)
            pca_cls_x = self.class_pca[cls].fit_transform(x[cls_idx])
            mean = torch.from_numpy(self.class_pca[cls].mean_)
            components = torch.from_numpy(self.class_pca[cls].components_)
            self.class_mean[cls] = mean
            self.class_trans_mat[cls] = components.T
            pca_cls_x_sign = (pca_cls_x >= 0.0, pca_cls_x < 0)
            self.class_pca_var[int(cls)] = torch.from_numpy(pca_cls_x.var(0))
            self.class_clusters[cls] = sign_partition(pca_cls_x, pca_cls_x_sign, [], [], np.ones(pca_cls_x.shape[0]) > 0)
            print(cls, ":", [clu.size for clu in self.class_clusters[cls]] )
            pass

    def buffer_update(self, inputs, labels, buffer, task_p):
        buffer_classes = set(to_numpy(buffer.memory_labels))
        current_classes = set(to_numpy(labels).astype(np.int))
        memory_classes = set.union(buffer_classes, current_classes)

        class_vars = to_numpy(self.class_pca_var[[int(cls) for cls in memory_classes]])
        class_prorate = class_vars.sum(1) / class_vars.sum()

        class_inputs = {}
        class_task_map = {}

        for b_cls in buffer_classes:
            class_inputs[b_cls] = buffer.memory_inputs[buffer.memory_labels == b_cls]
            class_task_map[b_cls] = buffer.memory_tasks[buffer.memory_labels == b_cls][0].item()
        for t_cls in current_classes:
            class_inputs[t_cls] = inputs[labels == t_cls]
            class_task_map[t_cls] = task_p

        class_size = np.zeros(len(class_inputs), dtype=np.int)
        left_memories = buffer.n_memories
        for cls in np.argsort(class_prorate)[::-1]:
            class_size[cls] = min(int(np.ceil(left_memories * class_prorate[cls])), len(class_inputs[cls]))
            left_memories -= class_size[cls]
            class_prorate[cls] = 0
            class_prorate = class_prorate / (class_prorate.sum() + 1e-7)

        new_buffer_inputs, new_buffer_labels, new_buffer_keep_prob, new_buffer_tasks = [], [], [], []

        for i, cls in enumerate(class_inputs.keys()):
            choose_size = class_size[i]
            clusters = self.class_clusters[cls]
            cluster_size = np.array([clu.indices.sum() for clu in clusters])
            cluster_choose_size = np.round((cluster_size + 0.0) / cluster_size.sum() * choose_size).astype(np.int)
            if choose_size > cluster_choose_size.sum():
                while choose_size > cluster_choose_size.sum():
                    add_1_idx = np.arange(len(cluster_choose_size))[cluster_choose_size < cluster_size][choose_size - cluster_choose_size.sum()]
                    cluster_choose_size[add_1_idx] += 1
            elif choose_size < cluster_choose_size.sum():
                sub_1_idx = np.argsort(cluster_choose_size)[::-1][np.arange(cluster_choose_size.sum() - choose_size)]
                cluster_choose_size[sub_1_idx] -= 1

            if cluster_choose_size.sum() != choose_size:
                print('debug')

            choose_idx = []
            cur_i = 0
            for j, clu in enumerate(clusters):
                if clu.l2_dis.sum() == 0:
                    continue
                p = clu.l2_dis / clu.l2_dis.sum()
                idx = np.random.choice(np.arange(len(p)), cluster_choose_size[j], p=p, replace=False)
                # idx = np.random.choice(np.arange(len(p)), cluster_choose_size[j], replace=False)
                choose_idx.append(idx)
                new_indices = np.zeros(choose_size).astype(np.bool)
                new_indices[cur_i:cur_i+cluster_choose_size[j]] = True
                new_l2_dis = np.zeros(choose_size)
                new_l2_dis[cur_i:cur_i+cluster_choose_size[j]] = clu.l2_dis[idx]
                clu.indices = new_indices
                clu.l2_dis = new_l2_dis
                cur_i += cluster_choose_size[j]
            choose_idx = np.concatenate(choose_idx, 0)
            new_buffer_inputs.append(class_inputs[cls][choose_idx])
            new_buffer_tasks.append(torch.LongTensor(len(choose_idx)).fill_(class_task_map[cls]))
            new_buffer_labels.append(torch.LongTensor(len(choose_idx)).fill_(int(cls)))
            buffer.abandoned_class_inputs[cls].append(class_inputs[cls][np.setdiff1d(np.arange(len(class_inputs[cls])), choose_idx)])
        buffer.memory_inputs = torch.cat(new_buffer_inputs, 0)
        buffer.memory_labels = torch.cat(new_buffer_labels, 0)
        buffer.memory_tasks = torch.cat(new_buffer_tasks, 0)

        buffer.class_n_samples = {cls: class_size[i] for i, cls in enumerate(class_inputs.keys())}
        buffer.shuffer()

    def pca_loss(self, features, labels):
        pca_features = self.transform(features, labels.cpu())
        l2_dis = pca_features ** 2
        class_var = torch.index_select(self.class_pca_var, dim=0, index=labels.cpu()).cuda()
        loss = (l2_dis - class_var).square().mean(-1)
        return loss



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
            self.task_discriminater = MLP_Classifier(parameters[self.data_name].task_discriminater, [])

            for _ in range(self.n_tasks):

                self.Decoders.append(ResNet18_Decoder(parameters[self.data_name].pool_size,
                                                      parameters[self.data_name].nf,
                                                      parameters[self.data_name].hidden_size,
                                                      parameters[self.data_name].input_dims).cuda())

        self.prototypes = Prototype(args.top_n_eigs, self.n_classes, parameters[self.data_name].hidden_size,
                                    parameters[self.data_name].input_dims)

        def ladder_loss(ens, des, weights=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]):
            if isinstance(weights, int) or isinstance(weights, float):
                weights = [weights] * len(ens)
            loss = 0
            for en, de, w in zip(ens, des, weights):
                if w > 0:
                    # en1 = torch.sigmoid(en) *2-1
                    # de1 = torch.sigmoid(de)*2-1
                    # de1 += 1e-6
                    # loss += torch.nn.KLDivLoss(reduction='mean')(de1.log(), en1.detach()) * w
                    loss += torch.nn.MSELoss()(en.detach(), de[:, -en.shape[1]:]) * w
                    pass

            return loss

        def var_l2_loss(ens, weights=[0.0, 0.01, 0.01, 0.2, 0.3, 0.5]):
            loss = 0
            for en, w in zip(ens, weights):
                if w > 0:
                    loss -= w * en.var()
            return loss


        self.decoder_loss_fn = ladder_loss
        self.classifier_loss_fn = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.var_loss_fn = var_l2_loss

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)

        self.decoder_lr = args.lr_decoder
        self.decoder_optimizer = None
        self.prototype_optimizer = torch.optim.SGD(self.prototypes.parameters(), lr=args.lr_decoder)

        self.decoder_loss_weight = args.decoder_loss_weight
        self.weight_l2_loss_weight = args.weight_l2_loss_weight
        self.sub_bias_weight = args.sub_bias_weight

        self.task_encoder_state_dict = defaultdict(list)
        self.observed_tasks = []
        self.cuda()

        self.show_all_parameters()

    def forward(self, x, tasks=None, with_details=False):
        if self.model_type == 'mlp':
            cls_feature = self.encoder(x)
            hidden_feature = [x, cls_feature]
        elif self.model_type == 'resnet':
            cls_feature, hidden_feature = self.encoder(x, tasks, with_hidden=True)
        y = self.classifier(cls_feature)
        if with_details:
            return y, hidden_feature
        return y


    def compute_prototype(self,inputs, labels, type):
        with torch.no_grad():
            _, en_features = self.forward(inputs, with_details=True)
            self.prototypes.compute_parameters(normalize(en_features[-1]), labels, type)


    def compute_feature(self, inputs, labels):
        with torch.no_grad():
            _, en_features = self.forward(inputs, with_details=True)
            return normalize(en_features[-1])

    def memory_bn_statistic(self, task_p):
        pre_state_dict = deepcopy(self.encoder.state_dict())
        for n, v in dict(pre_state_dict).items():
            self.task_encoder_state_dict[n].append(v)
        # self.task_encoder_state_dict[task_p] = pre_state_dict
        pass


    def update_bn_statistic(self):
        current_state = self.encoder.state_dict()
        for name, param in current_state.items():
            if 'running_mean' in name or 'running_var' in name:
                new_data = current_state[name].data.clone()
                last_data = new_data.clone()
                for pt, pre_state_dict in self.task_encoder_state_dict.items():
                    last_data = pre_state_dict[name]
                new_data = (new_data + last_data) * 0.5
                # new_data /= (len(self.task_encoder_state_dict) + 1)
                param.data.copy_(new_data)




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


        pt_classifier_loss, pt_decoder_loss, pt_pre_decoder_loss,  pt_acc, pre_decoder_loss, var_l2_loss = torch.zeros(6)

        if task_p != 0:
            pt_inputs, pt_labels, pt_weights, pt_tasks = retrieve(self.buffer, labels.shape[0] * task_p) #
            pt_sample_size = pt_inputs.shape[0]
            cmb_inputs = torch.cat([inputs, pt_inputs], 0)
            cmb_labels = torch.cat([labels, pt_labels], 0)
            cmb_tasks = torch.cat([torch.zeros_like(labels).long().fill_(task_p), pt_tasks], 0)
            cmb_logits, cmb_en_features = self.forward(cmb_inputs, cmb_tasks, with_details=True)
            pt_en_features = [f[-pt_sample_size:] for f in cmb_en_features]

            prototype_features = self.prototypes(normalize(cmb_en_features[-1]), cmb_labels)
            cmb_de_outputs, cmb_de_features = decoder(prototype_features)
            pre_decoder = self.Decoders[task_p-1]
            pt_pre_de_outpus, pt_pre_de_features = pre_decoder(prototype_features[-pt_sample_size:])

            classifier_loss = self.classifier_loss_fn(cmb_logits, cmb_labels)
            classifier_loss_mean = classifier_loss.mean()
            decoder_loss = self.decoder_loss_fn(cmb_en_features[:-1], cmb_de_features)
            pt_pre_decoder_loss = self.decoder_loss_fn(pt_en_features[:-1], pt_pre_de_features)
            var_l2_loss = self.var_loss_fn(pt_en_features[:-1])

            loss += decoder_loss * self.decoder_loss_weight\
                    + pre_decoder_loss * self.decoder_loss_weight\
                    + classifier_loss_mean\
                    # + var_l2_loss * 0.1

            pt_acc = torch.eq(torch.argmax(cmb_logits[-pt_sample_size:], dim=1), pt_labels).float().mean()
            acc = torch.eq(torch.argmax(cmb_logits[:-pt_sample_size], dim=1), labels).float().mean()
            pt_classifier_loss = classifier_loss[-pt_sample_size:].mean()
            classifier_loss = classifier_loss[:-pt_sample_size].mean()

        if task_p == 0:
            tasks = torch.zeros_like(labels).long().fill_(task_p)
            logits, en_features = self.forward(inputs, tasks, with_details=True)
            prototype_feature = self.prototypes(normalize(en_features[-1]), labels)

            en_features = en_features[:-1]
            decoder_outputs, de_features = decoder(prototype_feature)

            decoder_loss = self.decoder_loss_fn(en_features, de_features)
            classifier_loss = self.classifier_loss_fn(logits, labels).mean()

            loss += decoder_loss * self.decoder_loss_weight\
                    + classifier_loss\

            acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()


        if (torch.isnan(loss).sum() > 0):
            print("here!")

        loss.backward(retain_graph=True)

        self.classifier_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.prototype_optimizer.step()


        return {'classifier_loss': float(classifier_loss.item()), 'acc':float(acc.item()),
                'decoder_loss': float(decoder_loss.item()), 'var_l2_loss':float(var_l2_loss.item())}, \
               {'classifier_loss': float(pt_classifier_loss.item()), 'acc': float(pt_acc.item()),
                'decoder_loss': float(pt_pre_decoder_loss.item()), 'var_l2_loss':float(var_l2_loss.item())}


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
        self.train_process = [self.train_step] * (self.n_epochs - 1) \
                            +[self.apply_PCA]
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
        self.model.memory_bn_statistic(self.task_p)
        self.metric(self.task_p)

    def apply_PCA(self):
        self.logger.info(f'task {self.task_p} is beginning to apply PCA and select_prototype_sample.')
        inputs, labels, features, centers, weights = [], [], [], [], []
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            inputs += [batch_inputs]
            labels += [batch_labels]
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            batch_features = self.model.compute_feature(batch_inputs, batch_labels)
            features += [to_numpy(batch_features)]
            bar.update(batch_labels.size(0))
        bar.close()


        labels_np = to_numpy(torch.cat(labels, dim=0))
        features = np.concatenate(features, axis=0)

        self.model.prototypes.apply_PCA(features, labels_np)
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)


        # draw_tSNE(weights[idx], self.model.prototypes.class_mu[cls][select_center], centers[idx])

        # imshow(self.model.Decoders[0](
        #     (self.model.prototypes.class_mu[cls] * self.model.prototypes.M_eig_sqrt[cls:cls+1, :self.model.prototypes.statistic_dims].cpu()).matmul(
        #         self.model.prototypes.M_evc[cls, :, :self.model.prototypes.statistic_dims].T.cpu()).cuda())[0][select_center])
        # self.model.memory_bn_statistic(self.task_p)
        self.model.prototypes.buffer_update(inputs, labels, self.model.buffer, self.task_p)
        pass
        # prototype_update(self.model.buffer, inputs, labels, probs)




    def train(self):
        self.metric(-1)
        for self.task_p in range(self.data_loader.n_tasks):
            self.class_offset = self.get_class_offset(self.task_p)
            self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
            for self.epoch, func in enumerate(self.train_process) :
                func()
            self.logger.info(f'task {self.task_p} decoder has learned over')

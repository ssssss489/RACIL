
from model.base_model import *
from train.base_train import *

class Prototype(torch.nn.Module):
    def __init__(self, n_top_eigs, n_classes, hidden_size):
        super(Prototype, self).__init__()
        self.n_top_eigs = n_top_eigs
        self.n_centers = 2 ** n_top_eigs
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        self.init_parameters()

        self.class_multivar_normals = []

    def init_parameters(self):

        self.M = torch.nn.Parameter(torch.randn([self.n_classes, self.hidden_size, self.hidden_size]) * 0.1)
        self.M_eig = None
        self.M_evc = None
        self.M_inv = None

        self.class_mu = torch.zeros([self.n_classes, self.n_centers, self.hidden_size])
        self.class_cov = torch.zeros([self.n_classes, self.n_centers, self.hidden_size, self.hidden_size])
        self.class_n_samples = torch.zeros([self.n_classes, self.n_centers])
        self.cuda()

    def get_sym_M(self):
        M = self.M.triu()
        diag = M * torch.eye(self.hidden_size).cuda().unsqueeze(0)
        M = M + torch.nn.ReLU()(- (diag - 1e-3)).detach()
        M = torch.matmul(M, M.transpose(-1,-2))#  + torch.eye(self.hidden_size).cuda() * 1e-3
        return M

    def forward(self, x, y):
        y_ = one_hot(y, self.n_classes).unsqueeze(-1)
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

        y_ = one_hot(y, self.n_classes).unsqueeze(-1)
        class_M_eig = y_ * self.M_eig.unsqueeze(0)
        class_M_evc = y_.unsqueeze(-1) * self.M_evc.unsqueeze(0)

        weight = x.view(x.shape[0], 1, 1, x.shape[1]).matmul(class_M_evc).squeeze(2) * class_M_eig
        feature = weight.unsqueeze(2).matmul(class_M_evc.transpose(-1,-2))
        feature = feature.sum(1).squeeze(1)
        weight = weight.sum(1)
        return weight, feature


    def scan_samples(self, x, y, type):
        with torch.no_grad():
            logits, features = self.multi_eig_M(x, y)
            x = features
            logits = logits[:, :self.n_top_eigs]
            y_ = one_hot(y, self.n_classes).view([y.shape[0], self.n_classes, 1, 1]).cpu()

            k_ = one_hot(binary_decimal((logits > 0).float()).long(), self.n_centers).view([y.shape[0], 1, self.n_centers, 1]).cpu()

            if type=='mu':
                x_ = x.view([x.shape[0], 1, 1, x.shape[1]]).cpu() * y_ * k_
                self.class_mu += x_.sum(0)
                self.class_n_samples += (y_ * k_).sum(0).squeeze(-1).cpu()
            elif type=='cov':
                diff = x.view([x.shape[0], 1, 1, x.shape[1]]).cpu() - self.class_mu.unsqueeze(0)
                diff_ = diff * y_ * k_
                # diff_ = diff_.cuda().sum(1, keepdim=True).sum(2, keepdim=True)
                # cov = torch.matmul(diff_.unsqueeze(-1), diff_.unsqueeze(-2)).sum(0)
                # class_cov = cov.cpu() * y_.unsqueeze(-1) * k_.unsqueeze(-1)
                # self.class_cov += class_cov.sum(0)
                diff_ = diff_.cuda()
                self.class_cov += torch.matmul(diff_.unsqueeze(-1), diff_.unsqueeze(-2)).sum(0).cpu()


    def compute_cov(self):
        self.class_cov = self.class_cov / (self.class_n_samples.unsqueeze(-1).unsqueeze(-1) + 1e-6)

    def compute_mu(self):
        self.class_mu = self.class_mu / (self.class_n_samples.unsqueeze(-1) + 1e-6)

    def set_multivar_normals(self, scale_cov=1):
        for cls in range(self.n_classes):
            t = []
            for center in range(self.n_centers):
                t.append(torch.distributions.multivariate_normal.MultivariateNormal(self.class_mu[cls, center],
                                                                                    self.class_cov[cls, center] * scale_cov
                                                                                    + torch.eye(self.hidden_size) * 1e-3))
            self.class_multivar_normals.append(t)

    def sample(self, y):
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


def ladder_loss(ens, des, weights=[1.0, 0.0, 0.0, 0.00, 0.00, 0.0]):
    if isinstance(weights, int) or isinstance(weights, float):
        weights = [weights] * len(ens)
    loss = 0
    for en, de, w in zip(ens, des, weights):
        loss += torch.nn.MSELoss()(en, de) * w
    return loss




class prototype_decoder(base_model):
    def __init__(self, args, model_type):
        super(prototype_decoder, self).__init__(args)
        self.decoder_update = args.decoder_update

        self.Prototypes = []
        self.Decoders = []
        self.model_type = model_type
        if model_type == 'mlp':
            self.encoder = MLP_Encoder(parameters[self.data_name].encoder)
            self.classifier = MLP_Classifier(parameters[self.data_name].classifier)

            for _ in range(self.n_tasks):
                self.Prototypes.append(Prototype(args.top_n_eigs, self.n_classes, parameters[self.data_name].encoder[-1]))
                self.Decoders.append(MLP_Decoder(parameters[self.data_name].decoder, parameters[self.data_name].input_dims).cuda())

        elif model_type == 'resnet':
            self.encoder = ResNet18_Encoder(parameters[self.data_name].input_dims,
                                            parameters[self.data_name].nf,
                                            parameters[self.data_name].pool_size)
            self.classifier = MLP_Classifier(parameters[self.data_name].classifier, [])

            for _ in range(self.n_tasks):
                self.Prototypes.append(
                    Prototype(args.top_n_eigs, 10, parameters[self.data_name].hidden_size))
                self.Decoders.append(ResNet18_Decoder(parameters[self.data_name].pool_size,
                                                      parameters[self.data_name].nf,
                                                      parameters[self.data_name].hidden_size,
                                                      parameters[self.data_name].input_dims).cuda())
                # self.Decoders.append(ndpm_Deocder(parameters[self.data_name].nf,
                #                                       parameters[self.data_name].classifier[0],
                #                                       parameters[self.data_name].input_dims).cuda())

        # self.decoder_loss_fn = lambda x, y: torch.nn.MSELoss()(x.view([x.shape[0], -1]), y.view([x.shape[0], -1]))
        self.decoder_loss_fn = ladder_loss

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.classifier_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=args.lr)
        self.decoder_lr = args.lr_decoder
        self.decoder_optimizer = None
        self.prototype_optimizer = None

        self.observed_tasks = []
        self.cuda()

        self.show_all_parameters()

    def forward(self, x, with_details=False):
        if self.model_type == 'mlp':
            cls_feature = self.encoder(x)
            # cls_feature = normalize(cls_feature)
            decoder_feature = cls_feature
        elif self.model_type == 'resnet':
            cls_feature, hidden_feature = self.encoder(x, with_hidden=True)
            # decoder_feature = normalize(decoder_feature)
        y = self.classifier(cls_feature)
        if with_details:
            return y, hidden_feature
        return y

    def compute_prototype(self,inputs, labels, get_class_offset, task_p, type):
        with torch.no_grad():
            class_offset = get_class_offset(task_p)
            _, feature = self.forward(inputs, with_details=True)
            _, labels = compute_output_offset(None, labels, class_offset)
            self.Prototypes[task_p].scan_samples(feature[-1], labels, type)

    def learn_step(self, inputs, labels, get_class_offset, task_p):

        loss = 0

        class_offset = get_class_offset(task_p)
        _, labels = compute_output_offset(None, labels, class_offset)
        with torch.no_grad():
            _, feature = self.forward(inputs, with_details=True)



        # add label to features
        # labels_ = one_hot(labels, self.n_classes)
        # feature_labels = torch.cat([prototype_feature, labels_], axis=-1)
        # decoder_outputs = decoder(feature_labels)


        if self.decoder_update > 0 and task_p > 0:
            pt = np.random.choice(self.observed_tasks[:-1])
            pt_prototype = self.Prototypes[pt]
            last_decoder = self.Decoders[task_p-1]
            pt_class_offset = get_class_offset(pt)
            _, pt_labels = compute_output_offset(None, labels, pt_class_offset)
            pt_feature, pt_labels = pt_prototype.sample(pt_labels)
            pt_pesudo_inputs = last_decoder(pt_feature)
            pt_decoder_outputs = decoder(pt_feature)
            loss_ = self.decoder_loss_fn(pt_decoder_outputs, pt_pesudo_inputs)
            loss += loss_ * 100

        if (torch.isnan(loss).sum() > 0):
            print("here!")

        loss.backward(retain_graph=True)



        return float(loss.item())


    def train_step(self, inputs, labels, get_class_offset, task_p, classifier_train_flag=True):
        self.train()
        self.zero_grad()
        decoder = self.Decoders[task_p]
        prototype = self.Prototypes[task_p]
        decoder.train()
        decoder.zero_grad()
        prototype.train()
        prototype.zero_grad()
        loss = 0

        if task_p not in self.observed_tasks:
            if len(self.observed_tasks) != 0:
                last_decoder = self.Decoders[self.observed_tasks[-1]]
                decoder.load_state_dict(dict(last_decoder.named_parameters()), strict=False)
            self.decoder_optimizer = torch.optim.SGD(self.Decoders[task_p].parameters(), lr=self.decoder_lr)
            self.prototype_optimizer = torch.optim.SGD(self.Prototypes[task_p].parameters(), lr=self.decoder_lr)
            self.observed_tasks.append(task_p)


        class_offset = get_class_offset(task_p)
        logits, en_features = self.forward(inputs, with_details=True)
        _, labels_offset = compute_output_offset(None, labels, class_offset)

        prototype_feature = prototype(en_features[-1], labels_offset)
        en_features = en_features[:-1]
        # decoder_outputs = decoder(feature)
        decoder_outputs, de_features = decoder(prototype_feature)

        _, pseudo_feature = self.forward(decoder_outputs, with_details=True)

        decoder_loss = self.decoder_loss_fn(en_features, de_features) # + self.decoder_loss_fn(pseudo_feature, feature)

        classifier_loss = self.classifier_loss_fn(logits, labels)
        loss += classifier_loss + decoder_loss

        pt_classifier_loss, pt_acc = torch.zeros(2)
        if task_p > 0:
            pt = np.random.choice(self.observed_tasks[:-1])
            prototype = self.Prototypes[pt]
            if self.decoder_update == 0:
                pt_decoder = self.Decoders[pt]
            else:
                pt_decoder = self.Decoders[task_p - 1]
            pt_class_offset = get_class_offset(pt)
            pt_prototype_feature, pt_labels_offset = prototype.sample(labels_offset)
            pt_labels = anti_compute_output_offset(pt_labels_offset, pt_class_offset)

            with torch.no_grad():
                pt_inputs = pt_decoder(pt_prototype_feature)

            # add label to featuers
            # pt_labels_ = one_hot(pt_labels, self.n_classes)
            # pt_feature_labels = torch.cat([pt_feature, pt_labels_], dim=-1)
            # pt_inputs = pt_decoder(pt_feature_labels).view([-1,28,28])

            pt_logits, new_pt_feature = self.forward(pt_inputs.detach(), with_details=True)

            # pt_logits, _ = compute_output_offset(pt_logits, None, pt_class_offset)
            pt_classifier_loss = self.classifier_loss_fn(pt_logits, pt_labels)
            loss += 1.0 * pt_classifier_loss
            pt_acc = torch.eq(torch.argmax(pt_logits, dim=1), pt_labels).float().mean()


        if (torch.isnan(loss).sum() > 0):
            print("here!")

        loss.backward(retain_graph=True)

        if classifier_train_flag:
            self.classifier_optimizer.step()
            self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.prototype_optimizer.step()

        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()
        return float(classifier_loss.item()), float(pt_classifier_loss.item()),\
               float(acc.item()), float(pt_acc.item()), float(decoder_loss.item())


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
        self.train_process = [self.train_step] * (self.n_epochs - 2) \
                            +[lambda : self.compute_prototype('mu')] \
                            +[lambda : self.compute_prototype('cov')]


    def train_step(self):
        classifier_losses, train_acc = [], []
        pt_classifier_losses, pt_train_acc = [], []
        decoder_losses = []
        self.logger.info(f'task {self.task_p} is beginning to train.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            loss, pt_loss, acc, pt_acc, de_loss = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset,
                                                                        self.task_p, self.epoch < self.n_epochs - self.n_epochs_learn - 2)
            classifier_losses.append(loss)
            train_acc.append(acc)
            pt_classifier_losses.append(pt_loss)
            pt_train_acc.append(pt_acc)
            decoder_losses.append(de_loss)
            bar.update(batch_labels.size(0))
        bar.close()
        print(f'    classifier_loss = {np.mean(classifier_losses)}, train_acc = {np.mean(train_acc)}')
        print(f'    pt_classifier_loss = {np.mean(pt_classifier_losses)}, pt_train_acc = {np.mean(pt_train_acc)}')
        print(f'    decoder_loss = {np.mean(decoder_losses)}')
        self.metric(self.task_p)


    def compute_prototype(self, type):
        self.logger.info(f'task {self.task_p} is beginning to compute prototype {type}.')
        bar = tqdm(total=self.total, desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            self.model.compute_prototype(batch_inputs, batch_labels, self.get_class_offset, self.task_p, type)
            bar.update(batch_labels.size(0))
        bar.close()
        if type == 'mu':
            self.model.Prototypes[self.task_p].compute_mu()
        else:
            self.model.Prototypes[self.task_p].compute_cov()
            self.model.Prototypes[self.task_p].set_multivar_normals(self.scale_cov)

    # def show_tSNE(self):
    #     self.model.show_feature_distrubtion(
    #         self.data_loader.task_datasets[self.task_p].train_images[:5000].cuda(),
    #         self.data_loader.task_datasets[self.task_p].train_labels[:5000].cuda(),
    #         self.task_p,
    #         4)

    def train(self):
        self.metric(-1)
        for self.task_p in range(self.data_loader.n_tasks):
            self.class_offset = self.get_class_offset(self.task_p)
            self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
            for self.epoch, func in enumerate(self.train_process) :
                func()
            self.logger.info(f'task {self.task_p} decoder has learned over')

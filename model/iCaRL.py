

from model.base_model import *
from memory.iCaRL_buffer import iCaRL_buffer, random_retrieve
from model.regularization import decoder_regularization




class iCaRL(ResNet18):
    def __init__(self, args):
        super(iCaRL, self).__init__(args)
        self.buffer = iCaRL_buffer(self.n_tasks, args.n_memories, self.n_classes, parameters[self.data_name].hidden_size)
        self.last_encoder = None
        self.last_classifier = None
        self.regularization = None
        self.nme_classifier = False
        self.classifier_loss_fn = nn.BCELoss()
        if args.regular_type == 'decoder':
            self.regularization = decoder_regularization(self.data_name, lr=args.lr_decoder, loss_weight=args.decoder_loss_weight)

    def last_forward(self, x, t=None):
        # self.last_encoder.eval()
        # self.last_classifier.eval()
        with torch.no_grad():
            return self.last_classifier(self.last_encoder(x, t)).sigmoid()

    def forward(self, x, t=None, with_hidden=False):
        x, en_features = self.encoder(x, t, with_hidden=True)
        y = self.classifier(x)
        y = y.sigmoid()
        if with_hidden:
            return y, en_features
        else:
            return y

    def unit_encoder_feature(self, inputs):
        self.eval()
        self.zero_grad()
        features = self.encoder(inputs)
        features = unit_vector(features)
        return features

    def predict(self, inputs, class_offset=None):
        self.eval()
        self.zero_grad()
        if self.nme_classifier:
            features = self.encoder_feature(inputs)
            logits = torch.matmul(features, self.buffer.prototypes.T.cuda())
        else:
            logits = self.forward(inputs)
        if class_offset is not None:
            offset = torch.zeros_like(logits).cuda().fill_(-1e10)
            offset[:, class_offset] = 0
            predicts = torch.argmax(logits + offset, dim=1)
        else:
            predicts = torch.argmax(logits, dim=1)
        return predicts

    def train_step(self, inputs, labels, current_class_offset, task_p, old_class_offset=None):
        self.train()
        self.zero_grad()
        loss = 0
        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            if task_p > 0:
                self.last_classifier = deepcopy(self.classifier)
                self.last_encoder = deepcopy(self.encoder)

        if task_p > 0:
            mem_inputs, mem_labels, mem_tasks, mem_logits = random_retrieve(self.buffer, n_retrieve=inputs.shape[0] * task_p)
        else:
            mem_inputs, mem_labels, mem_tasks, mem_logits = torch.FloatTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.FloatTensor().cuda()

        tasks = torch.cat([torch.zeros_like(labels).fill_(task_p).cuda(), mem_tasks], dim=0)
        inputs = torch.cat([inputs, mem_inputs], dim=0)
        labels = torch.cat([labels, mem_labels], dim=0)

        logits, en_features = self.forward(inputs, tasks, with_hidden=True)

        pt_sample_idx = tasks != task_p
        cur_sample_idx = tasks == task_p

        cur_logits, pt_logits = logits[cur_sample_idx], logits[pt_sample_idx]
        cur_labels, pt_labels = labels[cur_sample_idx], labels[pt_sample_idx]

        labels_ = one_hot(labels, self.n_classes)
        if task_p > 0:
            last_logits = self.last_forward(inputs, tasks)
            labels_[:, old_class_offset] = last_logits[:, old_class_offset]

        cur_labels_, pt_labels_ = labels_[cur_sample_idx], labels_[pt_sample_idx]

        all_obverse_classes = old_class_offset + current_class_offset

        cur_classifier_loss = self.classifier_loss_fn(cur_logits[:, all_obverse_classes], cur_labels_[:, all_obverse_classes])
        pt_classifier_loss = self.classifier_loss_fn(pt_logits[:, all_obverse_classes], pt_labels_[:, all_obverse_classes])
        loss += cur_classifier_loss + pt_classifier_loss

        regularize_loss, pt_regularize_loss, pt_distill_loss = torch.FloatTensor([0, 0, 0])

        if self.regularization:
            regularize_loss, pt_regularize_loss = self.regularization(en_features, tasks, task_p)
            loss += regularize_loss + pt_regularize_loss

        acc = torch.eq(torch.argmax(cur_logits, dim=1), cur_labels).float().mean()
        pt_acc = torch.eq(torch.argmax(pt_logits, dim=1), pt_labels).float().mean()

        loss.backward()
        self.optimizer.step()
        if self.regularization:
            self.regularization.optimizer.step()

        return {'loss': float(cur_classifier_loss.item()), 'acc': float(acc.item()),
                'regular_loss': float(regularize_loss.item()), 'distill_loss': float(pt_distill_loss.item())}, \
               {'loss': float(pt_classifier_loss.item()), 'acc': float(pt_acc.item()),
                'regular_loss': float(pt_regularize_loss.item()), 'distill_loss': float(pt_distill_loss.item())}

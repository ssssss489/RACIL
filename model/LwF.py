


from model.base_model import *
from memory.base_buffer import *
from model.regularization import decoder_regularization

class LwF(ResNet18):
    def __init__(self, args):
        super(LwF, self).__init__(args)
        self.last_classifier = None
        self.last_encoder = None
        self.regularization = None
        self.distill_loss_weight = args.distill_loss_weight
        if args.regular_type == 'decoder':
            self.regularization = decoder_regularization(self.data_name,
                                                         lr=args.lr_decoder,
                                                         loss_weight=args.decoder_loss_weight)

    def last_forward(self, x, t=None):
        # self.last_encoder.eval()
        # self.last_classifier.eval()
        with torch.no_grad():
            return self.last_classifier(self.last_encoder(x, t))

    def distill_loss(self, logits, labels, T=2.0):
        log_scores_norm = (logits / T).softmax(dim=1).log()
        targets_norm =(labels / T).softmax(dim=1)
        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        kd_loss = (-1 * targets_norm * log_scores_norm).sum(dim=1).mean() * T ** 2
        return kd_loss

    def train_step(self, inputs, labels, class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0
        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            if task_p > 0:
                self.last_classifier = deepcopy(self.classifier)
                self.last_encoder = deepcopy(self.encoder)

        tasks = torch.zeros_like(labels).fill_(task_p).cuda()
        logits, en_features = self.forward(inputs, tasks, with_hidden=True)

        classifier_loss = self.classifier_loss_fn(logits, labels)
        loss += (1/ (task_p + 1)) * classifier_loss

        if task_p > 0:
            dis_logits = self.last_forward(inputs, tasks)
            distill_loss = self.distill_loss(logits, dis_logits.detach())
            loss += (1 - (1/ (task_p + 1))) * distill_loss * self.distill_loss_weight

        regularize_loss, pt_regularize_loss = torch.FloatTensor([0,0])
        if self.regularization:
            regularize_loss, pt_regularize_loss = self.regularization(en_features, tasks, task_p)
            loss += regularize_loss + pt_regularize_loss

        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        loss.backward()
        self.optimizer.step()
        if self.regularization:
            self.regularization.optimizer.step()

        return {'loss': float(classifier_loss.item()), 'acc': float(acc.item()), 'regular_loss': float(regularize_loss.item())}, \
               {'loss': 0, 'acc': 0, 'regular_loss': 0}




from model.base_model import *
from model.regularization import decoder_regularization

class EWC(ResNet18):
    def __init__(self, args):
        super(EWC, self).__init__(args)
        self.running_fisher = self.init_fisher()
        self.normalized_fisher = self.init_fisher()
        self.tmp_fisher = self.init_fisher()
        self.prev_params = {}
        self.lambda_ = args.ewc_lambda
        self.alpha_ = args.ewc_alpha
        self.running_fisher_after = args.running_fisher_after
        self.running_idx = 1
        self.weights = {n: p for n, p in self.named_parameters() if p.requires_grad}



    def init_fisher(self):
        return {n: p.clone().detach().fill_(0) for n, p in self.named_parameters() if p.requires_grad}

    def accum_fisher(self):
        for n, p in self.tmp_fisher.items():
            p += self.weights[n].grad ** 2

    def update_running_fisher(self):
        for n, p in self.running_fisher.items():
            self.running_fisher[n] = (1. - self.alpha_) * p \
                                     + 1. / self.running_fisher_after * self.alpha_ * self.tmp_fisher[n]
            # reset the accumulated fisher
        self.tmp_fisher = self.init_fisher()

    def fisher_loss(self):
        reg_loss = 0
        if len(self.prev_params) > 0:
            # add regularization loss
            for n, p in self.weights.items():
                reg_loss += (self.normalized_fisher[n] * (p - self.prev_params[n]) ** 2).sum()
                # reg_loss += (1 * (p - self.prev_params[n]) ** 2).sum()
        return reg_loss

    def train_step(self, inputs, labels, class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0
        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            if task_p > 0:
                self.prev_params = deepcopy(self.weights)
                max_fisher = max([torch.max(m) for m in self.running_fisher.values()])
                min_fisher = min([torch.min(m) for m in self.running_fisher.values()])
                for n, p in self.running_fisher.items():
                    self.normalized_fisher[n] = (p - min_fisher) / (max_fisher - min_fisher + 1e-32)

        if self.running_idx % self.running_fisher_after == 0:
            self.update_running_fisher()
        self.running_idx += 1

        tasks = torch.zeros_like(labels).fill_(task_p).cuda()
        logits, en_features = self.forward(inputs, tasks, with_hidden=True)

        classifier_loss = self.classifier_loss_fn(logits, labels)
        fisher_loss = self.fisher_loss()
        loss += classifier_loss + self.lambda_ * fisher_loss

        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        loss.backward()
        self.accum_fisher()
        self.optimizer.step()


        return {'loss': float(classifier_loss.item()), 'acc': float(acc.item()), 'regular_loss': 0}, \
               {'loss': 0, 'acc': 0, 'regular_loss': 0}


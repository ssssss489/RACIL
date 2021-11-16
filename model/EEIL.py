



from model.base_model import *
from memory.base_buffer import *
from model.regularization import decoder_regularization

class EEIL(ResNet18):
    def __init__(self, args):
        super(EEIL, self).__init__(args)
        self.buffer = None
        self.next_buffer = Buffer(self.n_tasks, args.n_memories, self.n_classes)
        self.pre_classifiers = []
        self.task_class_offset = []
        self.eps_mem_batch = args.eps_mem_batch
        self.distill_weight = args.eeil_distll_weight
        self.regularization = None
        if args.regular_type == 'decoder':
            self.regularization = decoder_regularization(self.data_name,
                                                         lr=args.lr_decoder,
                                                         loss_weight=args.decoder_loss_weight)

    def distill_loss(self, logits, labels, class_offset, T=2.0):
        log_scores_norm = (logits ** 1/T).softmax(dim=1).log()
        targets_norm =(labels ** 1/ T).softmax(dim=1).detach()
        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        kd_loss = -1 * (targets_norm * log_scores_norm)[:, class_offset].sum(dim=1).mean()
        return kd_loss


    def train_step(self, inputs, labels, class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0
        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            self.buffer = deepcopy(self.next_buffer)
            self.task_class_offset.append(class_offset)
            if task_p > 0:
                self.pre_classifiers.append(deepcopy(self.classifier))

        if task_p > 0:
            mem_inputs, mem_labels, mem_tasks = random_retrieve(self.buffer, n_retrieve=inputs.shape[0] * task_p)
        else:
            mem_inputs, mem_labels, mem_tasks = torch.FloatTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda()

        total_class_average_update(self.next_buffer, inputs, labels, task_p)

        tasks = torch.cat([torch.zeros_like(labels).fill_(task_p).cuda(), mem_tasks], dim=0)
        inputs = torch.cat([inputs, mem_inputs], dim=0)
        labels = torch.cat([labels, mem_labels], dim=0)

        logits, en_features = self.forward(inputs, tasks, with_hidden=True)

        regularize_loss, pt_regularize_loss , = torch.FloatTensor([0,0])

        distill_loss = 0
        for pre_task in range(task_p):
            pre_task_logits = self.pre_classifiers[pre_task](en_features[-1])
            # distill_loss += self.distill_loss(logits, pre_task_logits, np.concatenate(self.task_class_offset[:pre_task + 1]))
            distill_loss += self.distill_loss(logits, pre_task_logits, self.task_class_offset[pre_task]) * self.distill_weight

        loss += distill_loss

        obversed_class_offset = np.concatenate(self.task_class_offset, axis=0)
        logits = mask(logits, obversed_class_offset)

        pt_sample_idx = tasks != task_p
        cur_sample_idx = tasks == task_p

        cur_logits, pt_logits = logits[cur_sample_idx], logits[pt_sample_idx]
        cur_labels, pt_labels = labels[cur_sample_idx], labels[pt_sample_idx]

        cur_classifier_loss = self.classifier_loss_fn(cur_logits, cur_labels)
        pt_classifier_loss = self.classifier_loss_fn(pt_logits, pt_labels)
        loss += cur_classifier_loss + pt_classifier_loss


        if self.regularization:
            regularize_loss, pt_regularize_loss = self.regularization(en_features, tasks, task_p)
            loss += regularize_loss + pt_regularize_loss

        acc = torch.eq(torch.argmax(cur_logits, dim=1), cur_labels).float().mean()
        pt_acc = torch.eq(torch.argmax(pt_logits, dim=1), pt_labels).float().mean()

        loss.backward()
        self.optimizer.step()
        if self.regularization:
            self.regularization.optimizer.step()

        return {'loss': float(cur_classifier_loss.item()), 'acc': float(acc.item()), 'distill_loss': float(distill_loss), 'regular_loss': float(regularize_loss.item())}, \
               {'loss': float(pt_classifier_loss.item()), 'acc': float(pt_acc.item()), 'distill_loss': float(distill_loss), 'regular_loss': float(pt_regularize_loss.item())}


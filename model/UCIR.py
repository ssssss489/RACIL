



from model.base_model import *
from memory.base_buffer import *
from model.regularization import decoder_regularization

class UCIL_classifier(nn.Module):
    def __init__(self, hidden_sizes):
        super(UCIL_classifier, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.weights = nn.Parameter(torch.randn(self.hidden_sizes))
        self.eta = nn.Parameter(torch.ones(self.hidden_sizes[-1]) * 5 , requires_grad=True)

    def forward(self, x):
        weightsT = unit_vector(self.weights.T)
        x = unit_vector(x)
        cos_sim = torch.matmul(x, weightsT.T)
        logits = cos_sim * self.eta.unsqueeze(0)
        return logits

    def cos_sim(self, x):
        weightsT = unit_vector(self.weights.T)
        x = unit_vector(x)
        cos_sim = torch.matmul(x, weightsT.T)
        return cos_sim



class UCIR(ResNet18):
    def __init__(self, args):
        super(UCIR, self).__init__(args)
        self.buffer = None
        self.next_buffer = Buffer(self.n_tasks, args.n_memories, self.n_classes)
        self.last_encoder = None
        self.last_classifier = None

        self.classifier = UCIL_classifier(parameters[self.data_name].classifier).cuda()
        self.regularization = None
        self.task_class_offset = []
        self.distill_loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.lambda_base = args.ucir_lambda
        if args.regular_type == 'decoder':
            self.regularization = decoder_regularization(self.data_name,
                                                         lr=args.lr_decoder,
                                                         loss_weight=args.decoder_loss_weight)

    def distill_loss(self, new_feature, org_feature):
        f1 = unit_vector(new_feature)
        f2 = unit_vector(org_feature).detach()
        return (1 - (f1 * f2).sum(-1)).mean()

    def margin_loss(self, new_feature, labels, margin=0.5, K=2):
        cos_sim = self.classifier.cos_sim(new_feature)
        labels_ = one_hot(labels, self.n_classes)
        target_cos_sim = (cos_sim * labels_).sum(-1)
        neg_cos_sim_k = torch.topk(cos_sim -1e10 * labels_, K, dim=-1)[0]
        margin_sub = margin - target_cos_sim.unsqueeze(-1) + neg_cos_sim_k
        loss = nn.ReLU()(margin_sub).sum(-1).mean()
        return loss


    def train_step(self, inputs, labels, class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0
        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            self.buffer = deepcopy(self.next_buffer)
            self.task_class_offset.append(class_offset)
            if task_p > 0:
                self.last_classifier = deepcopy(self.classifier)
                self.last_encoder = deepcopy(self.encoder)


        if task_p > 0:
            mem_inputs, mem_labels, mem_tasks = random_retrieve(self.buffer, n_retrieve=inputs.shape[0])
            pass
        else:
            mem_inputs, mem_labels, mem_tasks = torch.FloatTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda()

        total_class_average_update(self.next_buffer, inputs, labels, task_p)

        tasks = torch.cat([torch.zeros_like(labels).fill_(task_p).cuda(), mem_tasks], dim=0)
        inputs = torch.cat([inputs, mem_inputs], dim=0)
        labels = torch.cat([labels, mem_labels], dim=0)

        logits, en_features = self.forward(inputs, tasks, with_hidden=True)

        distill_loss, margin_loss = 0.0, 0.0

        if task_p > 0:
            last_features = self.last_encoder(inputs, tasks)
            distill_loss = self.distill_loss(en_features[-1], last_features)
            loss += distill_loss * self.lambda_base * np.sqrt(1 / (task_p))

        margin_loss = self.margin_loss(en_features[-1], labels)
        loss += margin_loss

        regularize_loss, pt_regularize_loss = torch.FloatTensor([0,0])
        if self.regularization:
            # en_features[-1] = unit_vector(en_features[-1])
            regularize_loss, pt_regularize_loss = self.regularization(en_features, tasks, task_p)
            loss += regularize_loss + pt_regularize_loss

        obversed_class_offset = np.concatenate(self.task_class_offset, axis=0)
        logits = mask(logits, obversed_class_offset)

        pt_sample_idx = tasks != task_p
        cur_sample_idx = tasks == task_p

        cur_logits, pt_logits = logits[cur_sample_idx], logits[pt_sample_idx]
        cur_labels, pt_labels = labels[cur_sample_idx], labels[pt_sample_idx]

        cur_classifier_loss = self.classifier_loss_fn(cur_logits, cur_labels)
        pt_classifier_loss = self.classifier_loss_fn(pt_logits, pt_labels)
        loss += cur_classifier_loss + pt_classifier_loss

        acc = torch.eq(torch.argmax(cur_logits, dim=1), cur_labels).float().mean()
        pt_acc = torch.eq(torch.argmax(pt_logits, dim=1), pt_labels).float().mean()

        loss.backward()
        self.optimizer.step()
        if self.regularization:
            self.regularization.optimizer.step()

        return {'loss': float(cur_classifier_loss.item()), 'acc': float(acc.item()),
                'margin_loss': float(margin_loss), 'distll_loss': float(distill_loss),
                'regular_loss': float(regularize_loss.item())}, \
               {'loss': float(pt_classifier_loss.item()), 'acc': float(pt_acc.item()),
                'margin_loss': float(margin_loss), 'distll_loss': float(distill_loss),
                'regular_loss': float(pt_regularize_loss.item())}


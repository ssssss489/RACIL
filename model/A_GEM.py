
from model.base_model import *
from memory.base_buffer import *
from model.regularization import decoder_regularization

def avg_project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()

    avg_memory = np.mean(memories_np, axis=0)
    new_grad = gradient_np - np.dot(avg_memory, gradient_np) / np.dot(gradient_np, gradient_np) * avg_memory
    # t = memories_np.shape[0]
    # P = np.dot(memories_np, memories_np.transpose())
    # P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    # q = np.dot(memories_np, gradient_np) * -1
    # G = np.eye(t)
    # h = np.zeros(t) + margin
    # v = quadprog.solve_qp(P, q, G, h)[0]
    # x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(new_grad).view(-1, 1))

class A_GEM(ResNet18):
    def __init__(self, args):
        super(A_GEM, self).__init__(args)
        self.grad_parameters = [p for p in self.parameters() if p.requires_grad]
        self.buffer = None
        self.next_buffer = Buffer(self.n_tasks, args.n_memories, self.n_classes)
        self.eps_mem_batch = args.eps_mem_batch
        self.regularization = None
        if args.regular_type == 'decoder':
            self.regularization = decoder_regularization(self.data_name,
                                                         lr=args.lr_decoder,
                                                         loss_weight=args.decoder_loss_weight)

    def project_grad(self, pt_classifier_loss):
        grad = [p.grad.clone() for p in self.grad_parameters]
        self.encoder.zero_grad()
        self.classifier.zero_grad()
        pt_classifier_loss.backward(retain_graph=True)
        grad_ref = [p.grad.clone() for p in self.grad_parameters]
        prod = sum([torch.sum(g * g_r) for g, g_r in zip(grad, grad_ref)])
        if prod < 0:
            prod_ref = sum([torch.sum(g_r ** 2) for g_r in grad_ref])
            # do projection
            grad = [g - prod / prod_ref * g_r for g, g_r in zip(grad, grad_ref)]
        # replace params' grad
        for g, p in zip(grad, self.grad_parameters):
            p.grad.data.copy_(g)


    def train_step(self, inputs, labels, class_offset, task_p):
        self.train()
        self.zero_grad()
        loss = 0
        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)
            self.buffer = deepcopy(self.next_buffer)

        if task_p > 0:
            mem_inputs, mem_labels, mem_tasks = random_retrieve(self.buffer, n_retrieve=inputs.shape[0] * task_p)
        else:
            mem_inputs, mem_labels, mem_tasks = torch.FloatTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda()

        total_class_average_update(self.next_buffer, inputs, labels, task_p)

        tasks = torch.cat([torch.zeros_like(labels).fill_(task_p).cuda(), mem_tasks], dim=0)
        inputs = torch.cat([inputs, mem_inputs], dim=0)
        labels = torch.cat([labels, mem_labels], dim=0)

        logits, en_features = self.forward(inputs, tasks, with_hidden=True)

        pt_sample_idx = tasks != task_p
        cur_sample_idx = tasks == task_p

        cur_logits, pt_logits = logits[cur_sample_idx], logits[pt_sample_idx]
        cur_labels, pt_labels = labels[cur_sample_idx], labels[pt_sample_idx]

        cur_classifier_loss = self.classifier_loss_fn(cur_logits, cur_labels)
        pt_classifier_loss = self.classifier_loss_fn(pt_logits, pt_labels)

        loss += cur_classifier_loss # + pt_classifier_loss

        regularize_loss, pt_regularize_loss = torch.FloatTensor([0,0])
        if self.regularization:
            regularize_loss, pt_regularize_loss = self.regularization(en_features, tasks, task_p)
            loss += regularize_loss + pt_regularize_loss

        acc = torch.eq(torch.argmax(cur_logits, dim=1), cur_labels).float().mean()
        pt_acc = torch.eq(torch.argmax(pt_logits, dim=1), pt_labels).float().mean()

        loss.backward(retain_graph=True)

        if self.regularization:
            self.regularization.optimizer.step()

        self.project_grad(pt_classifier_loss)

        self.optimizer.step()


        return {'loss': float(cur_classifier_loss.item()), 'acc': float(acc.item()), 'regular_loss': float(regularize_loss.item())}, \
               {'loss': float(pt_classifier_loss.item()), 'acc': float(pt_acc.item()), 'regular_loss': float(pt_regularize_loss.item())}



from model.base_model import *
import numpy as np
import quadprog

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
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
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))




class GEM(nn.Module):
    def __init__(self, model, data, args):
        super(GEM, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.model = model(data, args)

        self.n_memory = args.n_memory
        self.margin = args.memory_strength
        self.cuda = args.cuda
        self.output_size = self.model.output_size


        # allocate episodic memory
        self.memory_inputs = torch.FloatTensor(self.n_task, self.n_memory, *models_setting[self.data].input_size)
        self.memory_labels = torch.LongTensor(self.n_task, self.n_memory)
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_task)

        if self.cuda:
            self.memory_inputs = self.memory_inputs.cuda()
            self.memory_labels = self.memory_labels.cuda()
            self.grads = self.grads.cuda()

        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.data == 'cifar100':
            self.nc_per_task = int(self.model.hidden_sizes[-1] / self.n_task)
        else:
            self.nc_per_task = self.model.hidden_sizes[-1]

    def forward(self, x):
        y = self.model(x)
        return y


    def train_step(self, inputs, labels, get_class_offset, t):
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        self.train()
            # Update ring buffer storing examples from current task
        bsz = labels.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memory)
        effbsz = endcnt - self.mem_cnt
        self.memory_inputs[t, self.mem_cnt: endcnt].copy_(
            inputs.data[: effbsz])
        if bsz == 1:
            self.memory_labels[t, self.mem_cnt] = labels.data[0]
        else:
            self.memory_labels[t, self.mem_cnt: endcnt].copy_(labels.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt >= self.n_memory:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                ptlogits = self.forward(self.memory_inputs[past_task])

                pt_class_offset = get_class_offset(past_task)

                ptloss = self.model.loss_fn(*self.model.compute_output_offset(ptlogits, self.memory_labels[past_task], *pt_class_offset))
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()
        logits = self.forward(inputs)
        class_offset = get_class_offset(t)
        logits, labels = self.model.compute_output_offset(logits, labels, *class_offset)
        loss = self.model.loss_fn(logits, labels)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.cuda \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)
        self.model.optimizer.step()
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        return float(loss.item()), float(acc.item())

    def predict(self, inputs, t):
        return self.model.predict(inputs, t)
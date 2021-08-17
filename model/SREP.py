

from model.base_model import *
from copy import deepcopy
from collections import defaultdict

class SREP(nn.Module):
    def __init__(self, model, data, args):
        super(SREP, self).__init__()
        self.data = data
        self.n_task = args.n_task

        # self.n_memory = args.n_memory
        # self.reg = args.memory_strength

        self.beta = args.beta

        self.model = model(data, args)
        self.output_size = self.model.output_size

        self.memory_inputs = defaultdict(torch.FloatTensor)
        self.memory_labels = defaultdict(torch.LongTensor)
        self.n_memory = args.n_memory
        self.task_grad = {}

        self.observed_tasks = []
        self.current_task = -1



    def forward(self, x):
        return self.model(x)

    def predict(self, inputs, class_offset=None):
        return self.model.predict(inputs, class_offset)

    def compute_output_offset(self, logits, labels, output_offset_start, output_offset_end):
        if logits is not None:
            logits = logits[:, output_offset_start: output_offset_end]
        if labels is not None:
            labels = labels - output_offset_start
        return logits, labels

    def save_memory_samples(self, inputs, labels, t):
        if t != self.current_task:
            self.current_task = t
            self.observed_tasks.append(t)

        task_memory_inputs = self.memory_inputs[t]
        task_memory_labels = self.memory_labels[t]
        task_memory_labels = torch.cat((task_memory_labels, labels.cpu().data.clone()))
        task_memory_inputs = torch.cat((task_memory_inputs, inputs.cpu().data.clone()))
        self.memory_inputs[t] = task_memory_inputs[-self.n_memory:]
        self.memory_labels[t] = task_memory_labels[-self.n_memory:]


    def get_memory_samples(self, t, size):
        task_memory_inputs = self.memory_inputs[t]
        task_memory_labels = self.memory_labels[t]
        idx = torch.from_numpy(np.random.choice(np.arange(self.n_memory), size, replace=False))
        inputs = task_memory_inputs[idx]
        labels = task_memory_labels[idx]
        return inputs.cuda(), labels.cuda()


    def compute_grad(self, inputs_list, labels_list):
        logits = self.model(inputs_list)
        # opt = torch.optim.SGD(self.model.parameters(), lr=-self.beta)
        loss = self.model.loss_fn(logits, labels_list)
        loss.backward()
        o1_grad = {name: parm.grad.clone() for name, parm in self.model.named_parameters()}
        self.model.load_state_dict({name: parm.data - parm.grad * self.beta if parm.grad is not None else parm.data
                                    for name, parm in self.model.named_parameters()})
        for k in range(1):
            self.model.zero_grad()
            logits = self.model(inputs_list)
            loss = self.model.loss_fn(logits, labels_list)
            loss.backward()
        o2_grad = {name: parm.grad.clone() for name, parm in self.model.named_parameters()}
        return o1_grad, o2_grad



    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.model.train()
        self.zero_grad()

        class_offset = get_class_offset(task_p)

        before_model = deepcopy(self.model.state_dict())

        # with torch.no_grad():
        logits = self.forward(inputs)
        logits, flabels = self.compute_output_offset(logits, labels, *class_offset)
        loss = self.model.loss_fn(logits, flabels)
        acc = torch.eq(torch.argmax(logits, dim=1), flabels).float().mean()

        self.save_memory_samples(inputs, labels, task_p)

        if task_p > 0 and False:
            pt = np.random.choice(self.observed_tasks[:-1])
            pt_inputs, pt_labels = self.get_memory_samples(pt, 10)
            # inputs = torch.cat([pre_inputs, inputs], dim=0)
            # labels = torch.cat([pre_labels, labels], dim=0)

            pt_class_offset = get_class_offset(pt)
            pt_logits = self.forward(pt_inputs)
            loss_ = self.model.loss_fn(*self.compute_output_offset(pt_logits, pt_labels, *pt_class_offset))
            loss += loss_

        loss.backward()
        self.model.optimizer.step()


        # o1_grad, o2_grad = self.compute_grad(inputs, labels)
        # # print({name: grad.var() for name, grad in o1_grad.items()})
        # mu = 0.1 # * torch.log(loss - 0.1 + 1)
        # self.model.load_state_dict({name: before_model[name] - (o1_grad[name] +
        #                                                         mu * (o2_grad[name] - o1_grad[name]) / self.beta
        #                                                         ) * self.model.lr
        #                             for name in o2_grad})

        return float(loss.item()), float(acc.item())

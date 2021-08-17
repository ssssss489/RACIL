


from model.base_model import *
from copy import deepcopy
from collections import defaultdict





def cat_grad(grads):
    shapes = {}
    flatten_grads = []
    for name, grad in grads.items():
        shapes[name] = grad.size()
        flatten_grads.append(grad.view(-1))
    flatten_grad = torch.cat(flatten_grads, dim=-1)
    return flatten_grad, shapes

def split_grads(flatten_grad, shapes):
    start = 0
    grads = {}
    for name, shape in shapes.items():
        size = 1
        for s in shape:
            size *= s
        end = start + size
        grads[name] = flatten_grad[start:end].view(shape)
        start = end

    return grads


class Dual_parm(nn.Module):
    def __init__(self, model, data, args):
        super(Dual_parm, self).__init__()
        self.data = data
        self.n_task = args.n_task

        self.memory_inputs = defaultdict(torch.FloatTensor)
        self.memory_labels = defaultdict(torch.LongTensor)
        self.n_memory = args.n_memory

        self.beta = args.beta

        self.model = model(data, args)
        self.output_size = self.model.output_size

        self.task_over_parameters = []
        self.task_parameters = []

        self.over_train = False

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

    def compute_pt_avg_grad(self):
        task_weights = np.ones(len(self.task_over_parameters))
        task_weights = task_weights / task_weights.sum()

        current_parameters = dict(self.named_parameters())
        avg_grads = {name: torch.zeros_like(current_parameters[name].grad.data) for name in current_parameters}

        for pt in range(0, len(self.task_over_parameters)):
            pt_parameters = self.task_parameters[pt]
            pt_over_parameters = self.task_over_parameters[pt]
            # avg_grads = {name: avg_grads[name].data + task_weights[pt] *
            #                   (pt_parameters[name].data - pt_over_parameters[name].data) for name in avg_grads}
            avg_grads = {name: avg_grads[name].data + task_weights[pt] *
                               (current_parameters[name].data - pt_over_parameters[name].data) for name in avg_grads}
        return avg_grads


    def project_grad(self):
        task_weights = np.arange(len(self.task_over_parameters))
        task_weights = task_weights / task_weights.sum()

        current_parameters = dict(self.named_parameters())

        # avg_grads = {name: torch.zeros_like(current_parameters[name].grad.data) for name in current_parameters}

        for pt in range(0, len(self.task_over_parameters)):
            pt_parameters = self.task_parameters[pt]
            pt_over_parameters = self.task_over_parameters[pt]
            # avg_grads = {name: avg_grads[name].data + task_weights[pt] *
            #                   (pt_parameters[name].data - pt_over_parameters[name].data) for name in avg_grads}
            avg_grads = {name: avg_grads[name].data + task_weights[pt] *
                               (current_parameters[name].data - pt_over_parameters[name].data) for name in avg_grads}

        flatten_avg_grad, grad_shapes = cat_grad(avg_grads)
        flatten_current_grad, _ = cat_grad({name: current_parameters[name].grad.data for name in current_parameters})

        if torch.dot(flatten_current_grad, flatten_avg_grad) < 0:
            project_grad = flatten_current_grad - (torch.dot(flatten_current_grad, flatten_avg_grad) /
                                                   torch.dot(flatten_avg_grad, flatten_avg_grad)) * flatten_avg_grad
            for name, grad in split_grads(project_grad, grad_shapes).items():
                current_parameters[name].grad.data.copy_(grad)

        # for name, parameter in self.named_parameters():
        #     org_grad = parameter.grad.data.view(-1)
        #     avg_grad = avg_grads[name].view(-1)
        #     if torch.dot(org_grad, avg_grad) < 0:
        #         project_grad = org_grad - (torch.dot(org_grad, avg_grad) / torch.dot(avg_grad, avg_grad)) * avg_grad
        #         parameter.grad.data.copy_(project_grad.view(parameter.grad.data.size()))



    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.model.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        named_parameter_dict = dict(self.named_parameters())
        org_param = {name: param.clone() for name, param in named_parameter_dict.items()}
        self.save_memory_samples(inputs, labels, task_p)
        if task_p > 0:
            pt_loss = 0
            for pt in self.observed_tasks[:-1]:
                pt_inputs, pt_labels = self.get_memory_samples(pt, 10)
                pt_class_offset = get_class_offset(pt)
                pt_logits = self.forward(pt_inputs)
                pt_loss += self.model.loss_fn(*self.compute_output_offset(pt_logits, pt_labels, *pt_class_offset))

            pt_loss.backward()
            self.model.optimizer.step()
            # self.load_state_dict({name: param.data - param.grad.data * 0.01 for name, param in named_parameter_dict.items()})

        self.zero_grad()
        logits = self.forward(inputs)
        logits, flabels = self.compute_output_offset(logits, labels, *class_offset)

        if not self.over_train:
            loss = self.model.loss_fn(logits, flabels)
        else:
            loss = self.model.loss_fn(logits, flabels, 10.0)
        acc = torch.eq(torch.argmax(logits, dim=1), flabels).float().mean()
        loss.backward()
        self.model.optimizer.step()
        self.save_memory_samples(inputs, labels, task_p)

        if task_p > 0:
            self.load_state_dict({name: org_param[name].data + (param.data - org_param[name].data) * 0.5 for name, param in named_parameter_dict.items()})
            # for name, param in named_parameter_dict.items():
            #     param.data.copy_(org_param[name].data)
            # self.load_state_dict(org_param)
#            self.project_grad()
        else:
            self.model.optimizer.step()




        return float(loss.item()), float(acc.item())
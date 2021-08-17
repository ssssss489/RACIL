

from model.base_model import *
from copy import deepcopy
from collections import defaultdict
from model.buffer import Buffer
import numpy as np

class ER(nn.Module):
    def __init__(self, model, data, args, buffer=Buffer):
        super(ER, self).__init__()
        self.data = data
        self.n_task = args.n_task

        # self.beta = args.beta

        self.model = model(data, args)
        self.n_classes = self.model.n_classes

        self.buffer = buffer(args.n_task, args.n_memory, self.model.n_classes)
        self.task_grad = {}

        self.observed_tasks = []
        self.current_task = -1


    def forward(self, x):
        return self.model(x)


    def predict(self, inputs, class_offset=None):
        return self.model.predict(inputs, class_offset)

    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.model.train()
        self.model.zero_grad()

        class_offset = get_class_offset(task_p)

        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, *class_offset)
        loss = self.model.loss_fn(logits, labels)
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)

        if len(self.observed_tasks) > 1:
            pt = np.random.choice(self.observed_tasks[:-1])
            pt_inputs, pt_labels = self.buffer.get_buffer_samples([pt], labels.shape[0])
            pt_class_offset = get_class_offset(pt)
            pt_logits = self.forward(pt_inputs)
            loss_ = self.model.loss_fn(*compute_output_offset(pt_logits, pt_labels, *pt_class_offset))
            loss += 1.0 * loss_

        loss.backward()
        self.model.optimizer.step()

        return float(loss.item()), float(acc.item())

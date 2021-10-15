

from model.base_model import *
from copy import deepcopy
from collections import defaultdict
from model.buffer import *
import numpy as np

class ER(base_model):
    def __init__(self, args, model_type):
        super(ER, self).__init__(args)
        if model_type == 'resnet':
            self.model = ResNet18(args)
        elif model_type == 'mlp':
            self.model = MLP(args)


        self.eps_mem_batch = args.eps_mem_batch
        self.buffer = Buffer(self.n_tasks, args.n_memories, self.n_classes, self.input_dims)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)

    def forward(self, x):
        return self.model(x)


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.model.train()
        self.model.zero_grad()

        idx = random_update(self.buffer, inputs, labels)

        class_offset = get_class_offset(task_p)

        logits = self.forward(inputs)
        # logits, labels = compute_output_offset(logits, labels, class_offset)
        loss = self.classifier_loss_fn(logits, labels)
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        if task_p not in self.observed_tasks:
            self.observed_tasks.append(task_p)

        # if len(self.observed_tasks) > 1:

        pt_inputs, pt_labels = random_retrieve(self.buffer, self.eps_mem_batch)
        pt_logits = self.forward(pt_inputs)
        loss_ = self.classifier_loss_fn(pt_logits, pt_labels)
        loss += 1.0 * loss_

        loss.backward()
        self.optimizer.step()

        return float(loss.item()), float(acc.item())

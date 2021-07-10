

from model.base_model import *

class EWC(nn.Module):
    def __init__(self, model, data, args):
        super(EWC, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.n_memory = args.n_memory
        self.reg = args.memory_strength
        self.model = model(data, args)
        self.output_size = self.model.output_size

        self.fisher = {}
        self.current_task = 0
        self.optpar = {}
        self.memx = None
        self.memy = None





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

    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.model.train()
        if task_p != self.current_task:
            self.model.zero_grad()
            old_class_offset = get_class_offset(self.current_task)
            mem_logits = self.model.forward(self.memx)
            mem_logits, mem_labels = self.compute_output_offset(mem_logits, self.memy, *old_class_offset)
            self.model.loss_fn(mem_logits, mem_labels).backward()
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []
            for p in self.model.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher[self.current_task].append(pg)
            self.current_task = task_p
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = inputs.data.clone()
            self.memy = labels.data.clone()
        else:
            if self.memx.size(0) < self.n_memory:
                self.memx = torch.cat((self.memx, inputs.data.clone()))
                self.memy = torch.cat((self.memy, labels.data.clone()))
                if self.memx.size(0) > self.n_memory:
                    self.memx = self.memx[:self.n_memory]
                    self.memy = self.memy[:self.n_memory]

        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = self.compute_output_offset(logits, labels, *class_offset)
        loss = self.model.loss_fn(logits, labels)
        for tt in range(task_p):
            for i, p in enumerate(self.model.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss.backward()
        self.model.optimizer.step()
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()
        return float(loss.item()), float(acc.item())






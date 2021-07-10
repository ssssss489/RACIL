

from model.base_model import *
from copy import deepcopy

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

    def compute_grad(self, inputs, labels, class_offset):
        logits = self.model(inputs)
        # opt = torch.optim.SGD(self.model.parameters(), lr=-self.beta)
        logits, labels = self.compute_output_offset(logits, labels, *class_offset)
        loss = self.model.loss_fn(logits, labels)
        loss.backward()
        self.model.load_state_dict({name: parm.data - parm.grad * self.beta
                                    for name, parm in self.model.named_parameters()})
        for k in range(1):
            self.model.zero_grad()
            logits = self.model(inputs)
            logits, labels = self.compute_output_offset(logits, labels, *class_offset)
            loss = self.model.loss_fn(logits, labels)
            loss.backward()
        return self.model.named_parameters()


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.model.train()
        self.zero_grad()

        class_offset = get_class_offset(task_p)

        before_model = deepcopy(self.model.state_dict())

        with torch.no_grad():
            logits = self.forward(inputs)
            logits, labels = self.compute_output_offset(logits, labels, *class_offset)
            loss = self.model.loss_fn(logits, labels)

            acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        new_parameters = list(self.compute_grad(inputs, labels, class_offset))

        self.model.load_state_dict({name: before_model[name] - parm.grad * self.model.lr
                                    for name, parm in new_parameters})


        return float(loss.item()), float(acc.item())

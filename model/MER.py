
from model.base_model import *
import numpy as np
from torch.autograd import Variable
import random
from copy import deepcopy

class MER(nn.Module):
    def __init__(self, model, data, args):
        super(MER, self).__init__()

        self.model = model(data, args)

        self.batchSize = int(args.batch_size)

        self.memories = args.n_memory
        self.steps = int(args.batches_per_example)

        self.beta = args.beta
        self.gamma = args.gamma

        # allocate buffer
        self.M = []
        self.age = 0

        # handle gpus if specified
        self.cuda = args.cuda


    def forward(self, x, t):
        output = self.model(x, t)
        return output

    def getBatch(self, x, y, t):
        xi = Variable(torch.from_numpy(np.array(x))).float().view(1, -1)
        yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
        if self.cuda:
            xi = xi.cuda()
            yi = yi.cuda()
        bxs = [xi]
        bys = [yi]
        bts = [t]

        if len(self.M) > 0:
            order = [i for i in range(0, len(self.M))]
            osize = min(self.batchSize, len(self.M))
            for j in range(0, osize):
                random.shuffle(order)
                k = order[j]
                x, y, t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x))).float().view(1, -1)
                yi = Variable(torch.from_numpy(np.array(y))).long().view(1)
                # handle gpus if specified
                if self.cuda:
                    xi = xi.cuda()
                    yi = yi.cuda()
                bxs.append(xi)
                bys.append(yi)
                bts.append(t)

        return bxs, bys, bts

    def train_step(self, x, y, t):
        ### step through elements of x
        losses = 0.0
        train_acc = []
        for i in range(0, x.size()[0]):
            self.age += 1
            xi = x[i].data.cpu().numpy()
            yi = y[i].data.cpu().numpy()
            self.model.zero_grad()

            before = deepcopy(self.model.state_dict())
            for step in range(0, self.steps):
                weights_before = deepcopy(self.model.state_dict())
                # Draw batch from buffer:
                bxs, bys, bts = self.getBatch(xi, yi, t)
                loss = 0.0
                for idx in range(len(bxs)):
                    self.model.zero_grad()
                    bx = bxs[idx]
                    by = bys[idx]
                    bt = bts[idx]
                    logits = self.forward(bx, bt)
                    logits, labels = self.model.compute_output_offset(logits, by, bt)
                    loss = self.model.loss_fn(logits, labels)
                    loss.backward()
                    self.model.optimizer.step()
                    losses += loss
                    train_acc.append(torch.eq(torch.argmax(logits.cpu(), dim=1).view(-1), labels.cpu().view(-1)).numpy())
                weights_after = self.model.state_dict()

                # Within batch Reptile meta-update:
                self.model.load_state_dict(
                    {name: weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in
                     weights_before})

            after = self.model.state_dict()

            # Across batch Reptile meta-update:
            self.model.load_state_dict(
                {name: before[name] + ((after[name] - before[name]) * self.gamma) for name in before})

            # Reservoir sampling memory update:

            if len(self.M) < self.memories:
                self.M.append([xi, yi, t])

            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [xi, yi, t]

        return float(losses)/len(train_acc), np.mean(train_acc)

    def predict(self, inputs, t):
        return self.model.predict(inputs, t)
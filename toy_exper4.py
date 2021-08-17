

import torch
import numpy as np
from torch.autograd import Variable

# loss_fn = lambda x, y: (1 - x / 2 + x ** 5 + y ** 3) * torch.exp(-x ** 2 - y ** 2)
loss_fn = lambda x, y: 3 *x ** 2 + 2 * y ** 3

x1= Variable(torch.from_numpy(np.array(0.7)), requires_grad=True)
y1 = Variable(torch.from_numpy(np.array(1.2)), requires_grad=True)

x2 = x1 + 0.001
y2 = y1 - 0.002
loss1 = loss_fn(x1, y1)
loss2 = loss_fn(x2, y2)

print((loss2 - loss1).data)

loss1.backward()
grad = torch.FloatTensor(2)
grad[0] = x1.grad.data
grad[1] = y1.grad.data

print(torch.matmul(grad, torch.from_numpy(np.array([0.001, -0.002])).to(torch.float)))

print(x1.grad.data, y1.grad.data)
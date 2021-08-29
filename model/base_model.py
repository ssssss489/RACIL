
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from easydict import EasyDict
from utils import *

models_setting = {
    'mnist': EasyDict(
        encoder=[28 * 28, 128, 128],
        classifier=[128, 128, 10],
        decoder=[128, 128, 28 * 28],
        # hidden_sizes=[28 * 28, 512, 10],
        input_dims=[28, 28],
        # n_classes=10
    ),
    'cifar100': EasyDict(
        hidden_sizes=[3 * 32 * 32, 512, 512, 100],
        input_size=[3, 32, 32],
        n_classes=100,
        pool_size= 4
    ),
    'tinyimageNet': EasyDict(
        hidden_sizes=[3 * 64 * 64, 200, 200, 200],
        input_size=[3, 64, 64],
        n_classes=200,
        pool_size = 8
    ),


}

class MLP_Encoder(nn.Module):
    def __init__(self, hidden_sizes):
        super(MLP_Encoder, self).__init__()
        layers = [nn.Flatten()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Feature_Project(nn.Module):
    def __init__(self, n_center, hidden_size, beta=1, gama=0.8): #16 * 128 *
        super(Feature_Project, self).__init__()
        self.n_center = n_center
        self.center_vectors = None
        self.samplers = None
        self.beta = beta
        self.gama = gama
        # self.augment_d = torch.autograd.Variable(torch.zeros([100, n_center]).cuda())

    def forward(self, x):
        d = torch.matmul(x, self.center_vectors.T)
        mu = torch.matmul(d, self.center_vectors)
        c = sub_project(x, mu)
        return c, mu, d

    def pseudo_project(self, shape):
        if self.samplers is None:
            self.samplers = []
            matmul = torch.matmul(self.center_vectors, self.center_vectors.T)
            for v in matmul:
                self.samplers.append(torch.distributions.Dirichlet(torch.exp(v * self.beta)))
        samplers = np.random.choice(self.samplers, size=shape)
        self.augment_d = torch.stack([s.sample() for s in samplers], dim=0).cuda()
        mu = torch.matmul(self.augment_d, self.center_vectors)
        return mu, self.augment_d.detach()

    def init_centers(self, x):
        if self.center_vectors is None:
            rand_choice = (torch.rand([self.n_center, x.shape[0]]) > 0.8).to(torch.float).cuda()
            init_vector = torch.matmul(rand_choice, x)
            self.center_vectors = torch.autograd.Variable(unit_vector(init_vector))
            # self.center_vectors = torch.autograd.Variable(unit_vector(torch.randn([self.n_center, x.shape[1]]).cuda()))


    def update_centers(self, x, d, y, n_class):

        y_idx = one_hot(y, n_class)
        y_idx_mean = y_idx / (y.sum(dim=0, keepdim=True) + 1e-8)
        x_class_mean = torch.matmul(y_idx_mean.T, x)
        x_sub_mean = x - torch.matmul(y_idx, x_class_mean)

        d_argmax_idx = one_hot(torch.argmax(d, dim=1), self.n_center)
        d_argmax_idx_mean = d_argmax_idx / (d.sum(dim=0, keepdim=True) + 1e-8)
        new_vector = torch.matmul(d_argmax_idx_mean.T, x_sub_mean)

        # t = d.T
        # t_abs = torch.abs(t)
        # t = torch.where(t_abs == t_abs.max(0, keepdim=True).values, torch.ones_like(t), torch.zeros_like(t)) * t
        # ts = t / (torch.sum(t.abs(), dim=1, keepdim=True) + 1e-8)
        # new_vector = torch.matmul(ts, x)
        self.center_vectors.copy_(unit_vector(self.gama * self.center_vectors + (1-self.gama) * new_vector))





class MLP_Classifier(nn.Module):
    def __init__(self, hidden_sizes):
        super(MLP_Classifier, self).__init__()
        layers = [nn.ReLU()]
        self.hidden_sizes = hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP_Decoder(nn.Module):

    def __init__(self, hidden_sizes):
        super(MLP_Decoder, self).__init__()
        layers = []
        self.hidden_sizes = hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        # return (self.net(x) + 1) * 0.5


class MLP(nn.Module):
    def __init__(self, data, args):
        super(MLP, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.n_classes = models_setting[data].classifier[-1]
        self.feature_net = MLP_Encoder(models_setting[data].feature_extractor)
        self.classifier = MLP_Classifier(models_setting[data].classifier)
        if args.cuda:
            self.cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)
        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")



    def forward(self, x):
        y = self.classifier(self.feature_net(x))
        return y

    def compute_grad(self, inputs, labels, get_class_offset, task_p):
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, *class_offset)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        return [p.grad.data.clone() for p in self.parameters()]


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, *class_offset)
        loss = self.loss_fn(logits, labels)
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), float(acc.item())


    def predict(self, inputs, class_offset=None):
        self.eval()
        self.zero_grad()
        logits = self.forward(inputs)
        if class_offset:
            logits = logits[:, class_offset[0]: class_offset[1]]
            predicts = torch.argmax(logits, dim=1) + class_offset[0]
        else:
            predicts = torch.argmax(logits, dim=1)
        return predicts


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, data, args):
        super(ResNet, self).__init__()

        self.n_classes = num_classes
        self.data = data
        self.pool_size = models_setting[self.data].pool_size
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)

        if args.cuda:
            self.cuda()

        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, *models_setting[self.data].input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, self.pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



    def train_step(self, inputs, labels, get_class_offset, task_p=None):
        self.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, *class_offset)
        loss = self.loss_fn(logits, labels)
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), float(acc.item())

    def predict(self, inputs, class_offset=None):
        self.eval()
        self.zero_grad()
        logits = self.forward(inputs)
        if class_offset:
            logits = logits[:, class_offset[0]: class_offset[1]]
            predicts = torch.argmax(logits, dim=1) + class_offset[0]
        else:
            predicts = torch.argmax(logits, dim=1)
        return predicts

def ResNet18(data, args, nf=20):
    nclasses = models_setting[data].n_classes
    model = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, data, args)
    return model





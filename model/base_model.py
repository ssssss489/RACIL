
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from easydict import EasyDict

models_setting = {
    'mnist': EasyDict(
        hidden_sizes=[28 * 28, 100, 100, 10],
        input_size=[28, 28],
        output_size=10
    ),
    'cifar100': EasyDict(
        hidden_sizes=[3 * 32 * 32, 100, 100, 100],
        input_size=[3, 32, 32],
        output_size=100,
        pool_size= 4
    ),
    'tinyimageNet': EasyDict(
        hidden_sizes=[3 * 64 * 64, 200, 200, 200],
        input_size=[3, 64, 64],
        output_size=200,
        pool_size = 8
    ),


}



class MLP(nn.Module):
    def __init__(self, data, args):
        super(MLP, self).__init__()
        self.data = data
        self.n_task = args.n_task
        layers = [nn.Flatten()]
        self.hidden_sizes = models_setting[data].hidden_sizes
        self.output_size = models_setting[data].output_size
        for i in range(len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            if i < len(self.hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        if args.cuda:
            self.model.cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)

        for name, param in self.model.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")

    def compute_output_offset(self, logits, labels, output_offset_start, output_offset_end):
        if logits is not None:
            logits = logits[:, output_offset_start: output_offset_end]
        if labels is not None:
            labels = labels - output_offset_start
        return logits, labels

    def forward(self, x):
        y = self.model(x)
        return y

    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = self.compute_output_offset(logits, labels, *class_offset)
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

        self.output_size = num_classes
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

    def compute_output_offset(self, logits, labels, output_offset_start, output_offset_end):
        if logits is not None:
            logits = logits[:, output_offset_start: output_offset_end]
        if labels is not None:
            labels = labels - output_offset_start
        return logits, labels

    def train_step(self, inputs, labels, get_class_offset, task_p=None):
        self.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = self.compute_output_offset(logits, labels, *class_offset)
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
    nclasses = models_setting[data].output_size
    model = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, data, args)
    return model





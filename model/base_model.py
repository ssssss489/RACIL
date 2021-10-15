
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, interpolate
from easydict import EasyDict
from utils import *
from copy import deepcopy
from functools import partial
from model.batch_norm import *

parameters = {
    'mnist': EasyDict(
        encoder=[28 * 28, 128, 128],
        classifier=[128, 128, 10],
        decoder=[128, 128, 28 * 28],
        hidden_size=128,
        input_dims=[28, 28],
        n_classes=10,
    ),
    'cifar100': EasyDict(
        input_dims=[3, 32, 32],
        nf=20,
        classifier=[20 * 8, 100],
        task_discriminater=[20 * 8, 128,  10],
        hidden_size=160,
        n_classes=100,
        pool_size=4,
    ),
    'tinyimageNet': EasyDict(
        input_dims=[3, 64, 64],
        n_classes=200,
        pool_size=8
    ),
}


class base_model(nn.Module):
    def __init__(self, args):
        super(base_model, self).__init__()
        self.data_name = args.data_name
        self.n_tasks = args.n_tasks
        self.n_classes = parameters[self.data_name].n_classes
        self.input_dims = parameters[self.data_name].input_dims
        self.classifier_loss_fn = torch.nn.CrossEntropyLoss()
        self.observed_tasks = []
        self.lr = args.lr

    def show_all_parameters(self):
        for name, param in self.named_parameters():
            print(f"    Layer: {name} | Size: {param.size()} ")


    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, class_offset)
        loss = self.classifier_loss_fn(logits, labels)
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


class MLP_Classifier(nn.Module):
    def __init__(self, hidden_sizes, layers=[nn.ReLU()]):
        super(MLP_Classifier, self).__init__()
        self.hidden_sizes = hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP_Decoder(nn.Module):
    def __init__(self, hidden_sizes, outputs_dims):
        super(MLP_Decoder, self).__init__()
        layers = []
        self.hidden_sizes = hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
        self.outputs_dims = outputs_dims
        self.net = nn.Sequential(*layers)

    def forward(self, xs):
        input, x = xs
        f = self.net(x).view(x.size(0), *self.outputs_dims)
        return f, [f]


class MLP(base_model):
    def __init__(self, args):
        super(MLP, self).__init__(args)
        self.encoder = MLP_Encoder(parameters[self.data_name].encoder)
        self.classifier = MLP_Classifier(parameters[self.data_name].classifier)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)
        self.show_all_parameters()
        self.cuda()


    def forward(self, x):
        y = self.classifier(self.encoder(x))
        return y



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockEnc(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockEnc, self).__init__()
        bn_func = noiseBatchNorm2d
        # bn_func = nn.BatchNorm2d
        # bn_func = nn.InstanceNorm2d
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = bn_func(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_func(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_cov = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False);
            self.shortcut_bn = bn_func(self.expansion * planes)
            self.shortcut = None
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            #     bn_func(self.expansion * planes)
            # )

    def forward(self, x, t):
        out = relu(self.bn1(self.conv1(x), t))
        out = self.bn2(self.conv2(out), t)
        if self.shortcut is not None:
            out += x
        else:
            out += self.shortcut_bn(self.shortcut_cov(x), t)
        out = relu(out)
        return out


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super(ResizeConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        if scale_factor == 1:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=scale_factor, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=scale_factor, padding=1, output_padding=1)

    def forward(self, x):
        # x = interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super(BasicBlockDec, self).__init__()

        planes = int(in_planes / stride)
        bn_func = nn.BatchNorm2d
        # bn_func = nn.InstanceNorm2d
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = bn_func(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = bn_func(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = bn_func(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                bn_func(planes)
            )


    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18_Encoder(nn.Module):
    def __init__(self, input_dims, nf, pool_size, num_blocks=[2, 2, 2, 2]):
        super(ResNet18_Encoder, self).__init__()
        self.input_dims = input_dims
        self.pool_size = pool_size
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        bn_func = noiseBatchNorm2d
        # bn_func = nn.BatchNorm2d
        # bn_func = nn.InstanceNorm2d
        self.bn1 = bn_func(nf * 1)
        self.layer1 = self._make_layer(1, BasicBlockEnc, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(2, BasicBlockEnc, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(3, BasicBlockEnc, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(4, BasicBlockEnc, nf * 8, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(nf * 8 * self.pool_size * self.pool_size, 160)


    def _make_layer(self, l, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            bk = block(self.in_planes, planes, stride).cuda()
            self.add_module('layer_{}_{}'.format(l, i), bk)
            layers.append(bk)
            self.in_planes = planes * block.expansion
        def func(layers, x, t):
            for bk in layers:
                x = bk(x, t)
            return x
        return partial(func, layers) # nn.Sequential(*layers)

    def forward(self, x, t=None, with_hidden=False):
        bsz = x.size(0)
        x = x.view(bsz, *self.input_dims)
        out1 = relu(self.bn1(self.conv1(x), t)) # 20 32 32
        out2 = self.layer1(out1, t) # 20 32 32
        out3 = self.layer2(out2, t) # 40 16 16
        out4 = self.layer3(out3, t) # 80 8 8
        out5 = self.layer4(out4, t) # 160 4 4
        out6 = self.linear1(out5.view(bsz, -1)) # 160
        if with_hidden:
            return out6, [x, out1, out2, out3, out4, out5, out6]
        else:
            return out6

    def forward_from_layer3(self, out3, t=None, with_hidden=False):
        bsz = out3.size(0)
        out4 = self.layer3(out3, t) # 80 8 8
        out5 = self.layer4(out4, t) # 160 4 4
        out6 = self.linear1(out5.view(bsz, -1)) # 160
        if with_hidden:
            return out6, [out3, out4, out5, out6]
        else:
            return out6



class ResNet18_Decoder(nn.Module):
    def __init__(self, pool_size, nf, z_dim, output_dims, num_Blocks=[2,2,2,2]):
        super(ResNet18_Decoder, self).__init__()
        self.outputs_dims = output_dims
        self.pool_size = pool_size
        self.linear = nn.Linear(z_dim, z_dim * pool_size * pool_size)
        self.in_planes = z_dim
        self.layer4 = self._make_layer(BasicBlockDec, nf * 4, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, nf * 2, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, nf * 1, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, nf * 1, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(nf, self.outputs_dims[0], kernel_size=3, scale_factor=int(self.outputs_dims[-1]/32))
        pass
        # self.conv1 = nn.Conv2d(64, self.outputs_dims[0], kernel_size=3, stride=int(self.outputs_dims[-1]/32))

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)


    def forward(self, z):
        bsz = z.shape[0]
        out5_ = self.linear(z)
        out5 = out5_.view(bsz, -1, self.pool_size, self.pool_size) # 160 4 4
        out4 = self.layer4(out5) # 80 , 8, 8
        out3 = self.layer3(out4) # 40, 16, 16
        out2 = self.layer2(out3) # 20, 32, 32
        out1 = self.layer1(out2) # 20, 32, 32
        x = torch.sigmoid(self.conv1(out1))# * 0.5 + 0.5 # 3, 32, 32
        x = x.view(x.size(0), *self.outputs_dims)
        return x, [x, out1, out2, out3, out4, out5]

class ndpm_Deocder(nn.Module):
    def __init__(self, nf_base, inputs_dim, outputs_dims):
        super(ndpm_Deocder, self).__init__()
        self.outputs_dim = outputs_dims
        x_c, x_h, x_w = outputs_dims
        self.reshape = lambda x: x.view(x.size(0), 2 * nf_base, x_h // 4, x_w // 4)
        self.dec_z = nn.Sequential(
            nn.Linear(inputs_dim, 16 * nf_base),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Linear(
                16 * nf_base,
                (x_h // 4) * (x_w // 4) * 2 * nf_base),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(2 * nf_base, 1 * nf_base,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.ConvTranspose2d(1 * nf_base, x_c,
                                       kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        h3 = self.dec_z(x)
        h2 = self.dec3(h3)
        h2 = self.reshape(h2)
        h1 = self.dec2(h2)
        x_logit = self.dec1(h1)
        return torch.sigmoid(x_logit)




class ResNet18(base_model):
    def __init__(self, args):
        super(ResNet18, self).__init__(args)
        self.encoder = ResNet18_Encoder(parameters[self.data_name].input_dims,
                                        parameters[self.data_name].nf,
                                        parameters[self.data_name].pool_size)
        self.classifier = MLP_Classifier(parameters[self.data_name].classifier, [])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)
        self.show_all_parameters()
        self.cuda()

    def forward(self, x):
        y = self.classifier(self.encoder(x))
        return y








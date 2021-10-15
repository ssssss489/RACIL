
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def compute_output_offset(logits, labels, output_offset):
    if isinstance(output_offset[0], int):
        output_offset_start, output_offset_end = output_offset
        if logits is not None:
            if len(logits.shape) == 2:
                logits = logits[:, output_offset_start: output_offset_end]
            elif len(logits.shape) == 3:
                logits = logits[:, :, output_offset_start: output_offset_end]
            elif len(logits.shape) == 4:
                logits = logits[:, :, :, output_offset_start: output_offset_end]
        if labels is not None:
            labels = labels - output_offset_start
    else:
        for i, offset in enumerate(output_offset):
            output_offset_start, output_offset_end = offset
            if logits is not None:
                logits[i] = logits[i, output_offset_start: output_offset_end]
            if labels is not None:
                labels[i] = labels[i] - output_offset_start
    return logits, labels

def anti_compute_output_offset(labels, output_offset):
    output_offset_start, output_offset_end = output_offset
    return labels + output_offset_start

def compute_param_cos_similar(v1, v2):
    multi_v1_v2 = torch.sum(torch.Tensor([torch.matmul(v1_.reshape(-1), v2_.reshape(-1)) for v1_, v2_ in zip(v1, v2)]))
    mod_v1 = torch.sqrt(torch.sum(torch.Tensor([torch.sum(v1_ ** 2) for v1_ in v1])))
    mod_v2 = torch.sqrt(torch.sum(torch.Tensor([torch.sum(v2_ ** 2) for v2_ in v2])))
    cos = (multi_v1_v2 + 1e-8) / (mod_v2 * mod_v1 + 1e-8)
    return cos

def compute_feature_cos_similar(v1, v2):
    multi_v1_v2 = torch.sum(v1 * v2, dim=-1)
    mod_v1 = torch.sqrt(torch.sum(v1 ** 2, dim=-1))
    mod_v2 = torch.sqrt(torch.sum(v2 ** 2, dim=-1))
    cos = (multi_v1_v2 + 1e-8) / (mod_v2 * mod_v1 + 1e-8)
    return cos

def compute_euclid_similar(v1, v2):
    return torch.sqrt(torch.sum(torch.cat([(v1_.reshape(-1) - v2_.reshape(-1))**2 for v1_, v2_ in zip(v1, v2)])))

def normalize(var):
    var = var - var.mean(dim=-1).view([-1, 1])
    var = var / var.std(dim=-1).view([-1, 1])
    # var = var / (torch.sqrt((var ** 2).sum(dim=-1, keepdim=True) + 1e-8))
    return var
    #  11.2694

def min_max_normalize(var):
    min = var.min(-1, keepdim=True)[0]
    max = var.max(-1, keepdim=True)[0]
    return (var - min) / (max - min)



def sub_project(var, pro):
    var_pro = torch.sum(var * pro, dim=-1, keepdim=True)
    pro_pro = torch.sum(pro * pro, dim=-1, keepdim=True)
    result = var - (var_pro / (pro_pro + 1e-8)) * pro
    return result


def unit_vector(var):
    var = var / (torch.sqrt((var ** 2).sum(dim=-1, keepdim=True) + 1e-8))
    return var

def to_numpy(x):
    return x.data.cpu().numpy()


def imshow(imgs, n_col=3):
    num = imgs.shape[0]
    imgs = to_numpy(imgs)
    plt.axis('off')
    for i, img in enumerate(imgs):
        ax = plt.subplot(int(np.ceil(num/n_col)), n_col, i+1)
        ax.axis('off')
        if len(imgs.shape) == 3:
            ax.imshow(np.transpose(img), cmap="gray")
        else:
            ax.imshow(np.transpose(img, (1, 2, 0)))
    # plt.imshow()
    plt.show()


def one_hot(y, num):
    one_hot = torch.zeros(y.shape[0], num).cuda().scatter_(1, y.view([-1,1]), 1)
    return one_hot

def binary_decimal(x):
    b = 2 ** torch.from_numpy(int(x.shape[1]) - 1 - np.arange(int(x.shape[1])))
    d = (b.unsqueeze(0) * x).sum(1)
    return d


def draw_tSNE(feature, mu, k_assign):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(to_numpy(torch.cat([feature, mu], 0)))
    # plt.plot(data[:, 0], data[:, 1])
    mu = data[-int(mu.shape[0]):]
    data = data[:-int(mu.shape[0])]
    plt.scatter(data[:, 0], data[:, 1], c=to_numpy(k_assign), cmap=plt.cm.jet)
    plt.scatter(mu[:, 0], mu[:, 1], marker='D')
    # plt.scatter(data[int(feature.shape[0]):, 0], data[int(feature.shape[0]):, 1], )
    # plt.colorbar()
    plt.show()
    return
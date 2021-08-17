
import torch


def compute_output_offset(logits, labels, output_offset_start, output_offset_end):
    if logits is not None:
        logits = logits[:, output_offset_start: output_offset_end]
    if labels is not None:
        labels = labels - output_offset_start
    return logits, labels


def compute_cos_similar(v1, v2):
    multi_v1_v2 = torch.sum(torch.Tensor([torch.matmul(v1_.reshape(-1), v2_.reshape(-1)) for v1_, v2_ in zip(v1, v2)]))
    mod_v1 = torch.sqrt(torch.sum(torch.Tensor([torch.sum(v1_ ** 2) for v1_ in v1])))
    mod_v2 = torch.sqrt(torch.sum(torch.Tensor([torch.sum(v2_ ** 2) for v2_ in v2])))
    cos = (multi_v1_v2 + 1e-8) #/ (mod_v2 * mod_v1 + 1e-8)
    return cos

def compute_euclid_similar(v1, v2):
    return torch.sqrt(torch.sum(torch.cat([(v1_.reshape(-1) - v2_.reshape(-1))**2 for v1_, v2_ in zip(v1, v2)])))
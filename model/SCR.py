from model.base_model import *
import numpy as np
from model.buffer import Buffer

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class SCR(base_model):
    def __init__(self, args):
        super(SCR, self).__init__(args)
        self.encoder = ResNet18_Encoder(parameters[self.data_name].input_dims,
                                        parameters[self.data_name].nf,
                                        parameters[self.data_name].pool_size)
        self.loss_fn = SupConLoss()
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=args.lr)
        self.buffer = Buffer(self.n_tasks, args.n_memories, self.n_classes)

    def forward(self, x, train=False):
        if train:
            return self.encoder(x)
        else:
            feature = self.model.features(x)  # (batch_size, feature_size)
            for j in range(feature.size(0)):  # Normalize
                feature.data[j] = feature.data[j] / feature.data[j].norm()
            feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
            means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)
            means = torch.stack([means] * x.size(0))  # (batch_size, n_classes, feature_size)
            means = means.transpose(1, 2)
            feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
            dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
            _, pred_label = dists.min(1)



    def train_step(self, inputs, labels, get_class_offset, task_p):
        self.train()
        self.zero_grad()
        class_offset = get_class_offset(task_p)
        logits = self.forward(inputs)
        # logits, labels = compute_output_offset(logits, labels, class_offset)

        if task_p > 0:
            pt_inputs, pt_labels = self.buffer.retrieve(inputs, labels)
            pt_logits= self.fo

        loss = self.classifier_loss_fn(logits, labels)


        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), float(acc.item())









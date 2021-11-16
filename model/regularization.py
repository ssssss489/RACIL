
import torch
import torch.nn as nn
from model.base_model import ResNet18_Decoder, parameters, ndpm_Deocder
from copy import deepcopy


class decoder_regularization(nn.Module):
    def __init__(self, data_name, lr, loss_weight, last_loss_weight):
        super(decoder_regularization, self).__init__()
        self.decoder = ResNet18_Decoder(parameters[data_name].pool_size,
                                        parameters[data_name].nf,
                                        parameters[data_name].hidden_size,
                                        parameters[data_name].input_dims,)

        self.last_decoder = None
        self.lr = lr
        # self.optimizer = torch.optim.SGD(self.decoder.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(0.9, 0.999))

        self.observe_tasks = []
        self.loss_weight = loss_weight
        self.last_loss_weight = last_loss_weight
        self.cuda()

    def loss_fn(self, en, de, w=1.0):
        return torch.nn.MSELoss()(en.detach(), de) * w

    def forward(self, en_features, last_en_features, tasks, task_p):
        if task_p not in self.observe_tasks:
            self.observe_tasks.append(task_p)
            self.last_decoder = deepcopy(self.decoder)
        feature_idx = 3  # ER idx = 4 if lr = 0.01 & SGD
        outputs, de_features = self.decoder(en_features[-1])
        loss1 = self.loss_fn(en_features[feature_idx], de_features[feature_idx])
        loss2 = torch.FloatTensor([0]).cuda()[0]
        if task_p > 0:
            pre_sample_idx = tasks != task_p
            pre_en_feature = en_features[-1][pre_sample_idx]
            last_outputs, last_de_features = self.last_decoder(pre_en_feature)
            loss2 = self.loss_fn(last_en_features[feature_idx][pre_sample_idx], last_de_features[feature_idx])
        return loss1 * self.loss_weight, loss2 * self.last_loss_weight









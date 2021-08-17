

from model.base_model import *
from copy import deepcopy
from collections import defaultdict
from model.buffer import Buffer
import numpy as np



class Feature_Augment(nn.Module):
    def __init__(self, model, data, args):
        super(Feature_Augment, self).__init__()
        self.data = data
        self.n_task = args.n_task
        self.model = model(data, args)

        self.n_classes = self.model.n_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.pt_loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)

        self.buffer = Buffer(args.n_task, args.n_memory, self.n_classes)

        self.observed_task_classifiers = {}


    def forward(self, x):
        return self.model(x)


    def predict(self, inputs, class_offset=None):
        return self.model.predict(inputs, class_offset)


    def forward_with_augment(self, x, y, pt, augment_feature):
        feature = self.model.feature_net(x)
        noise = augment_feature * 0.1
        pt_classifier = self.observed_task_classifiers[pt].classifier
        # with torch.no_grad():
        pt_softmax = nn.Softmax(dim=-1)(pt_classifier(feature+noise))
        weights = nn.ReLU()(pt_softmax.gather(1, y.view(1, y.shape[0]).t()) - 0.5) * 2
        logits = deepcopy(self.model.classifier)(feature + noise)
        return logits, weights.detach(), feature

    def compute_feature_sub(self, x1, x2):
        with torch.no_grad():
            feature1 = self.model.feature_net(x1)
            feature2 = self.model.feature_net(x2)
            return feature1 - feature2


    def train_step(self, inputs, labels, get_class_offset, task_p, augment_inputs):
        self.train()
        self.zero_grad()

        class_offset = get_class_offset(task_p)

        logits = self.forward(inputs)
        logits, labels = compute_output_offset(logits, labels, *class_offset)
        loss = self.loss_fn(logits, labels)
        acc = torch.eq(torch.argmax(logits, dim=1), labels).float().mean()

        self.buffer.save_buffer_samples(inputs, labels, task_p)

        if task_p not in self.observed_task_classifiers:
            if task_p != 0:
                self.model = deepcopy(self.model)
            self.observed_task_classifiers[task_p] = self.model

        if len(self.observed_task_classifiers) > 1:
            pt = np.random.choice(list(self.observed_task_classifiers.keys())[:-1])
            pt_inputs, pt_labels = self.buffer.get_buffer_samples([pt], labels.shape[0])
            pt_class_offset = get_class_offset(pt)

            pt_logits = self.forward(pt_inputs)
            loss_ = self.pt_loss_fn(*compute_output_offset(pt_logits, pt_labels, *pt_class_offset))
            loss += 1.0 * torch.mean(loss_)

            # pt_logits, pt_weights = self.forward_with_augment(pt_inputs, pt_labels, pt, torch.randn(feature.shape).cuda() * 1)
            augment_feature = self.compute_feature_sub(inputs, augment_inputs)
            pt_logits, pt_weights, f = self.forward_with_augment(pt_inputs, pt_labels, pt,
                                                              augment_feature)
            loss_ = self.pt_loss_fn(*compute_output_offset(pt_logits, pt_labels, *pt_class_offset))
            loss += 1.0 * torch.mean(loss_ * pt_weights)

            pt_logits, pt_weights, f = self.forward_with_augment(pt_inputs, pt_labels, pt,
                                                              - augment_feature)
            loss_ = self.pt_loss_fn(*compute_output_offset(pt_logits, pt_labels, *pt_class_offset))
            loss += 1.0 * torch.mean(loss_ * pt_weights)

        loss.backward()
        self.model.optimizer.step()

        return float(loss.item()), float(acc.item())

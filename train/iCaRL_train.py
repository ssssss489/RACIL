

from train.base_train import *
from utils import *
from memory.iCaRL_buffer import iCaRL_update

class iCaRL_train(base_train):
    def __init__(self, data_loader, model, args, logger):
        super(iCaRL_train, self).__init__(data_loader, model, args, logger)
        self.train_process = [self.train_encoder_classifier] * (self.n_epochs - 1) +\
                             [self.memory_sample]

    def train_encoder_classifier(self):
        current_infos, pt_infos = [], []
        self.logger.info(f'task {self.task_p} is beginning to train.')
        old_class_offset = list(set(np.nonzero(self.trained_classes)[0]).difference(self.current_classes))
        bar = tqdm(total=self.data_loader.task_n_sample[self.task_p], desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            info, pt_info = self.model.train_step(batch_inputs, batch_labels,  self.current_classes, self.task_p, old_class_offset,)
            current_infos.append(info)
            pt_infos.append(pt_info)
            bar.update(batch_labels.size(0))
        bar.close()
        for k in current_infos[0].keys():
            print(f'    {k} = {np.mean([d[k] for d in current_infos])}, pt_{k} = {np.mean([d[k] for d in pt_infos])}')
        self.metric(self.task_p)

    def memory_sample(self):
        self.logger.info(f'task {self.task_p} is beginning to apply PCA and select_prototype_sample.')
        inputs, labels, features = [], [], [],
        bar = tqdm(total=self.data_loader.task_n_sample[self.task_p], desc=f'task {self.task_p} epoch {self.epoch}')
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch)):
            inputs += [batch_inputs]
            labels += [batch_labels]
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            with torch.no_grad():
                batch_features = self.model.encoder_feature(batch_inputs)
            features += [batch_features]
            bar.update(batch_labels.size(0))
        bar.close()
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        features = torch.cat(features, dim=0)
        with torch.no_grad():
            iCaRL_update(self.model.buffer, inputs, labels, self.task_p, features, self.model)
        self.model.nme_classifier = True
        self.metric(self.task_p)
        self.model.nme_classifier = False
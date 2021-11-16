
import torch
# from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from easydict import EasyDict
import os

class base_train:
    def __init__(self, data_loader, model, args, logger):
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.batch_size = args.batch_size
        self.result_file = os.path.join(args.result_path,
                                        f'{args.model}_{args.dataset}_{args.n_epochs}_{args.n_memories}_'
                                        f'{args.bn_type}_{args.regular_type}.pt')

        self.init_metric_results = None
        self.n_tasks = args.n_tasks
        self.trained_classes = np.zeros(self.model.n_classes).astype(np.bool)

        self.n_epochs = args.n_epochs
        self.n_pretrain_epochs = args.n_pretrain_epochs

        self.task_p =None
        self.epoch = None
        self.current_classes=None
        self.eval_size = 1024

        self.pretrain_process = [self.train_encoder_classifier] * self.n_pretrain_epochs
        self.train_process = [self.train_encoder_classifier] * self.n_epochs

        self.metric_results = []
        self.first_task_features = []


    def train_encoder_classifier(self):
        current_infos, pt_infos = [], []
        self.logger.info(f'task {self.task_p} is beginning to train.')
        bar = tqdm(total=self.data_loader.task_n_sample[self.task_p], desc=f'task {self.task_p} epoch {self.epoch}')
        batch_size = self.batch_size #int(self.batch_size / (self.task_p + 1))
        for i, (batch_inputs, batch_labels) in enumerate(self.data_loader.get_batch(self.task_p, epoch=self.epoch, batch_size=batch_size)):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.to(torch.int64).cuda()
            info, pt_info = self.model.train_step(batch_inputs, batch_labels, self.current_classes , self.task_p)
            current_infos.append(info)
            pt_infos.append(pt_info)
            bar.update(batch_labels.size(0))
        bar.close()
        for k in current_infos[0].keys():
            print(f'    {k} = {np.mean([d[k] for d in current_infos])}, pt_{k} = {np.mean([d[k] for d in pt_infos])}')
        self.metric(self.task_p)


    def train(self):
        self.metric(-1)
        for self.task_p in range(self.data_loader.n_tasks):
            self.current_classes = self.data_loader.task_datasets[self.task_p].classes
            self.trained_classes[self.current_classes] = True
            if self.n_pretrain_epochs > 0 and self.task_p == 0:
                for self.epoch, func in enumerate(self.pretrain_process):
                    func()
            else:
                for self.epoch, func in enumerate(self.train_process):
                    func()
            self.logger.info(f'task {self.task_p} decoder has learned over')
        self.save_train_results()

    def metric(self, task_p=None):
        eval_size = self.eval_size
        task_accuracy = []
        offset_task_accuracy = []

        if task_p == -1:
            observed_classes = np.arange(self.model.n_classes)
        else:
            observed_classes = np.nonzero(self.trained_classes)[0].tolist()

        for i, dataset in enumerate(self.data_loader.task_datasets):
            test_size = dataset.test_labels.size(0)
            start = 0
            predicts = torch.zeros_like(dataset.test_labels)
            offset_predicts = torch.zeros_like(dataset.test_labels)
            features = []
            while start < test_size:
                if start + eval_size < test_size:
                    end = start + eval_size
                else:
                    end = test_size
                inputs = dataset.test_images[start: end]
                inputs = inputs.cuda()
                with torch.no_grad():
                    offset_predicts[start: end] = self.model.predict(inputs, dataset.classes)
                    predicts[start: end] = self.model.predict(inputs, observed_classes)
                    if i == 0 and self.epoch == self.n_epochs - 1:
                        features.append(self.model.encoder_feature(inputs).cpu())
                start = end
            offset_accuray = np.mean(dataset.test_labels.numpy() == offset_predicts.numpy())
            offset_task_accuracy.append(offset_accuray)
            accuray = np.mean(dataset.test_labels.numpy() == predicts.numpy())
            task_accuracy.append(accuray)
            print(f'    task{i}: offset_acc={offset_accuray :.4f} acc = {accuray :.4f}')
            if i == 0 and self.epoch == self.n_epochs - 1:
                self.first_task_features.append((torch.cat(features, dim=0), dataset.test_labels))
        if task_p == -1:
            self.init_metric_results = task_accuracy
        elif task_p is not None:
            pre_task_accuracy = task_accuracy[:task_p+1]
            avg_acc = np.mean(pre_task_accuracy)
            print(f'    average accuracy = {avg_acc}')

            pre_metric_results = np.array(self.metric_results)
            if task_p >= 1:
                max_pre_metric_results = pre_metric_results[:].max(axis=0)[:task_p]
                avg_fgt = np.mean(max_pre_metric_results - pre_task_accuracy[:task_p])
                print(f'    average forgetting = {avg_fgt}')
            self.metric_results.append(task_accuracy)


    def save_train_results(self):
        result = EasyDict(init_accuracy=self.init_metric_results,
                          tasks_accuracy=self.metric_results,
                          first_task_feature=self.first_task_features,)
        torch.save(result, self.result_file)
        self.logger.info(f'experiment result has been saved in {self.result_file}')



















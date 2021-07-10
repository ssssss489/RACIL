
import torch
# from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from easydict import EasyDict

class base_train:
    def __init__(self, data_loader, model, args, logger):
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.cuda = args.cuda
        self.batch_size = args.batch_size
        self.metric_results = []
        self.offset_metric_results = []
        self.init_metric_results = None
        self.n_task = args.n_task
        self.trained_classes = np.zeros(self.model.output_size).astype(np.bool)
        self.class_offset = None

    def train(self):
        last_task_p = -1
        last_epoch = -1
        bar = None
        total = self.data_loader.task_n_sample[0]
        losses, train_acc = [], []
        for i, (task_p, epoch, batch_inputs, batch_labels) in enumerate(self.data_loader):

            if last_epoch != epoch or task_p != last_task_p:
                if bar:
                    bar.close()
                print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')

                if epoch != 0:
                    self.metric()
                else:
                    self.metric(last_task_p)
                    self.logger.info(f'task {last_task_p} has trained over ,eval result is shown below.')
                    self.class_offset = self.get_class_offset(task_p)
                    self.trained_classes[self.class_offset[0]:self.class_offset[1]] = True
                    last_task_p = task_p
                    total = self.data_loader.task_n_sample[task_p]

                bar = tqdm(total=total, desc=f'task {task_p} epoch {epoch}')
                losses, train_acc = [], []
                last_epoch = epoch

            ## train step
            if self.cuda:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.to(torch.int64).cuda()
            loss, acc = self.model.train_step(batch_inputs, batch_labels, self.get_class_offset, task_p)
            losses.append(loss)
            train_acc.append(acc)
            bar.update(batch_labels.size(0))

        if bar:
            bar.close()
        print(f'    loss = {np.mean(losses)}, train_acc = {np.mean(train_acc)}')
        self.metric(last_task_p)
        self.logger.info(f'task {last_task_p} has trained over ,eval result is shown below.')



    def get_class_offset(self, t):
        if self.model.data == 'mnist':
            class_offset = (0, 10)
        elif self.model.data == 'cifar100' or self.model.data == 'tinyimageNet':
            class_offset = self.data_loader.task_datasets[t].classes
        return class_offset



    def metric(self, task_p=None):
        eval_size = 1024
        task_accuracy = []
        offset_task_accuracy = []
        if task_p == -1:
            n_class = self.model.output_size
        else:
            n_class = self.trained_classes.sum()
        for i, dataset in enumerate(self.data_loader.task_datasets):
            test_size = dataset.test_labels.size(0)
            start = 0
            predicts = torch.zeros_like(dataset.test_labels)
            offset_predicts = torch.zeros_like(dataset.test_labels)
            while start < test_size:
                if start + eval_size < test_size:
                    end = start + eval_size
                else:
                    end = test_size
                inputs = dataset.test_images[start: end]
                if self.cuda:
                    inputs = inputs.cuda()
                offset_predicts[start: end] = self.model.predict(inputs, self.get_class_offset(i))
                predicts[start: end] = self.model.predict(inputs, (0, n_class))
                start = end
            offset_accuray = np.mean(dataset.test_labels.numpy() == offset_predicts.numpy())
            offset_task_accuracy.append(offset_accuray)
            accuray = np.mean(dataset.test_labels.numpy() == predicts.numpy())
            task_accuracy.append(accuray)
            print(f'    task{i}: offset_acc={offset_accuray :.4f} acc = {accuray :.4f}')
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

    def save_train_results(self, file_name):
        result = EasyDict(init_accuracy=self.init_metric_results,
                          tasks_accuracy=self.metric_results,)
        torch.save(result, file_name)



















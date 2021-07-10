import os
import logging
import torch
import argparse
from easydict import EasyDict
from dataset import task_data_loader, mkdir
from model.base_model import MLP, ResNet18
from model.GEM import GEM
from model.MER import MER
from model.EWC import EWC
from model.SREP import SREP
from train.base_train import base_train






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--result_path', default='result/', help='input directory')
    parser.add_argument('--dataset', default='mnist_perm_10', help='learn task')
    parser.add_argument('--n_task', default=10, type=int, help='number of tasks')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_epoch', default=1, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--samples_per_task', type=int, default=-1, help='training samples per task (all if negative)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='mlp', help='learner model')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU')
    parser.add_argument('--n_memory', default=1000, help='number of memories per task')
    parser.add_argument('--memory_strength', default=1000, type=float, help='memory strength (meaning depends on memory)')
    parser.add_argument('--beta', default=0.01, type=float, help='beta learning rate parameter')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma learning rate parameter')
    parser.add_argument('--batches_per_example', type=int, default=5, help='the number of batch per incoming example')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))

    task = args.dataset.split('_')[0]
    if args.model == 'mlp':
        logger.info('init mlp model for {}'.format(task))
        model = MLP(task, args)
        # model = EWC(MLP,task, args)
        # model = SREP(MLP, task, args)


    elif args.model == 'resnet':
        logger.info('init resnet model for {}'.format(task))
        model = ResNet18(task, args)


    data_loader = task_data_loader(task_datasets, args)

    trainer = base_train(data_loader, model, args, logger)
    trainer.train()

    mkdir(args.result_path)
    trainer.save_train_results(os.path.join(args.result_path, f'{args.model}_{args.dataset}_{args.batch_size}_{args.n_epoch}_{args.lr}.pt'))












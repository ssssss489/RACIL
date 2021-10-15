import os
import logging
import torch
import argparse
from easydict import EasyDict
from dataset import task_data_loader, mkdir
from model.base_model import MLP, ResNet18
from model.GEM import GEM
from model.A_GEM import A_GEM
from model.MER import MER
from model.EWC import EWC
from model.SREP import SREP
from model.ER import ER
from model.Dual_parm import Dual_parm
from model.Feature_Augment import Feature_Augment
from model.prototype_decoder_PCA_distill import prototype_decoder, prototype_decoder_train
from train.base_train import base_train






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()

    # task setting
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--result_path', default='result/', help='input directory')
    parser.add_argument('--dataset', default='cifar100_10', help='learn task')
    # parser.add_argument('--dataset', default='mnist_rot_10', help='learn task')

    parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')

    # base model setting
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--n_epochs', default=2, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--samples_per_task', type=int, default=-1, help='training samples per task (all if negative)')
    parser.add_argument('--seed', default=3, type=int, help='random seed')

    # parser.add_argument('--model', default='resnet', help='learner model')
    parser.add_argument('--model', default='prototype_decoder', help='learner model')
    # parser.add_argument('--model', default='ER', help='learner model')

    parser.add_argument('--n_memories', default=2000, help='number of memories per task')
    parser.add_argument('--eps_mem_batch', default=100, type=int, help='Episode memory per batch ')

    parser.add_argument('--memory_strength', default=0.5, type=float, help='memory strength (meaning depends on memory)')
    parser.add_argument('--beta', default=0.1, type=float, help='beta learning rate parameter')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma learning rate parameter')
    parser.add_argument('--batches_per_example', type=int, default=5, help='the number of batch per incoming example')

    #prototype_decoder
    parser.add_argument('--top_n_eigs', default=10, help='number of top eigs for partition')
    parser.add_argument('--decoder_update', default=0, help='1 means to update the decoder, otherwise 0')
    parser.add_argument('--n_epochs_learn', default=6, help='train decoder epochs')
    parser.add_argument('--scale_cov', default=1, help='scale cov')
    parser.add_argument('--lr_decoder', default=0.1, type=float, help='learning rate of decoder')
    parser.add_argument('--decoder_loss_weight', default=10, type=float, help='weight of decoder loss')
    parser.add_argument('--weight_l2_loss_weight', default=0.1, type=float, help='weight of weight_l2_loss')
    parser.add_argument('--sub_bias_weight', default=1, type=float, help='weight of sub_bias_weig')


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.data_name = args.dataset.split('_')[0]

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))
    data_loader = task_data_loader(task_datasets, args)

    if args.data_name.startswith('mnist'):
        if args.model == 'mlp':
            logger.info('init mlp model for {}'.format(args.data_name))
            model = MLP(args)
            trainer = base_train(data_loader, model, args, logger)
        elif args.model == 'prototype_decoder':
            logger.info('init prototype_decoder model for {}'.format(args.data_name))
            args.n_epochs += args.n_epochs_learn + 2
            data_loader = task_data_loader(task_datasets, args)
            model = prototype_decoder(args, 'mlp')
            trainer = prototype_decoder_train(data_loader, model, args, logger)
        elif args.model == 'ER':
            logger.info('init ER model for {}'.format(args.data_name))
            model = ER(args, 'mlp')
            trainer = base_train(data_loader, model, args, logger)

        # model = ER(MLP, task, args)

        # model = EWC(MLP,task, args)
        # model = GEM(MLP, task, args)
        # model = SREP(MLP, task, args)
        # model = A_GEM(MLP, task, args)

    if args.data_name.startswith('cifar100') or args.data_name.startswith('tinyimageNet'):
        if args.model == 'resnet':
            logger.info('init resnet model for {}'.format(args.data_name))
            model = ResNet18(args)
            trainer = base_train(data_loader, model, args, logger)
        elif args.model == 'prototype_decoder':
            logger.info('init prototype_decoder model for {}'.format(args.data_name))
            args.n_epochs += args.n_epochs_learn + 1
            data_loader = task_data_loader(task_datasets, args)
            model = prototype_decoder(args, 'resnet')
            trainer = prototype_decoder_train(data_loader, model, args, logger)
        elif args.model == 'ER':
            logger.info('init ER model for {}'.format(args.data_name))
            model = ER(args, 'resnet')
            trainer = base_train(data_loader, model, args, logger)


        # model = EWC(ResNet18, task, args)
        # model = GEM(MLP, task, args)
        # model = SREP(ResNet18, task, args)




    trainer.train()

    mkdir(args.result_path)
    trainer.save_train_results(os.path.join(args.result_path, f'{args.model}_{args.dataset}_{args.seed}_{args.batch_size}_{args.n_epochs}_{args.lr}.pt'))












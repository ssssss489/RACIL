import os
import logging
import torch
import argparse
from easydict import EasyDict
from dataset import task_data_loader, mkdir
from model.base_model import MLP, ResNet18
from model.iCaRL import iCaRL
from train.iCaRL_train import iCaRL_train
from model.LwF import LwF
from model.EEIL import EEIL
from model.A_GEM import A_GEM
from model.MER import MER
from model.UCIR import UCIR
from model.EWC import EWC
from model.SREP import SREP
from model.ER import ER

from model.prototype_decoder_PCA_distill import prototype_decoder, prototype_decoder_train
from train.base_train import base_train



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(os.path.split(__file__)[1])

    parser = argparse.ArgumentParser()

    # task setting
    parser.add_argument('--path', default='dataset/', help='input directory')
    parser.add_argument('--result_path', default='result/', help='result directory')
    # parser.add_argument('--dataset', default='cifar100_10_3', help='learn task')
    parser.add_argument('--dataset', default='cifar100_11_0', help='learn task')

    # parser.add_argument('--dataset', default='miniimageNet64_10_3', help='learn task')

    parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
    parser.add_argument('--samples_per_task', type=int, default=-1, help='training samples per task (all if negative)')

    # base model setting
    parser.add_argument('--model', default='ER',
                        choices=['ER', 'iCaRL', 'EEIL', 'UCIR',  'LwF', 'None', 'EWC', 'AGEM',  'prototype_decoder'],
                        help='learner model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate') #icarl 0.1 UCIR
    parser.add_argument('--n_epochs', default=6, type=int, help='number of epochs')
    parser.add_argument('--n_pretrain_epochs', default=10, type=int, help='number of epochs for pretrain')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')


    # base memory setting
    parser.add_argument('--n_memories', default=2000, type=int, help='number of memories per task')

    # add noisy batch Normalization and decoder based regularization
    parser.add_argument('--bn_type', default='bn', choices=['bn', 'nbn'], help='choice bn types')
    parser.add_argument('--regular_type', default='None', choices=['None', 'decoder',], help='flag of add decoder regularization')
    parser.add_argument('--lr_decoder', default=0.1, type=float, help='learning rate of decoder')
    parser.add_argument('--decoder_loss_weight', default=10, type=float, help='weight of decoder loss') # donot fit iCaRL

    # LwF parameters
    parser.add_argument('--distill_loss_weight', default=1.0, help='weight of knowledge distill loss')


    # EWC parameters
    parser.add_argument('--ewc_lambda', default=100, help= 'parameter lambda in ewc')
    parser.add_argument('--ewc_alpha', default=0.9, help= 'parameter alpha in ewc')
    parser.add_argument('--running_fisher_after', default=5, help= 'parameter lambda in ewc')

    #EEIL parametrs
    parser.add_argument('--eeil_distll_weight', default=1, help= 'parameter lambda in ucir')



    #UCIR parameters
    parser.add_argument('--ucir_lambda', default=1, help= 'parameter lambda in ucir')



    parser.add_argument('--eps_mem_batch', default=100, type=int, help='Episode memory per batch ')

    parser.add_argument('--memory_strength', default=0.5, type=float, help='memory strength (meaning depends on memory)')
    parser.add_argument('--beta', default=0.1, type=float, help='beta learning rate parameter')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma learning rate parameter')
    parser.add_argument('--batches_per_example', type=int, default=5, help='the number of batch per incoming example')

    #prototype_decoder
    parser.add_argument('--top_n_eigs', default=10, help='number of top eigs for partition')
    parser.add_argument('--decoder_update', default=0, help='1 means to update the decoder, otherwise 0')
    parser.add_argument('--n_epochs_learn', default=0, help='train decoder epochs')
    parser.add_argument('--scale_cov', default=1, help='scale cov')
    parser.add_argument('--weight_l2_loss_weight', default=0.1, type=float, help='weight of weight_l2_loss')
    parser.add_argument('--sub_bias_weight', default=1, type=float, help='weight of sub_bias_weig')


    args = parser.parse_args()

    mkdir(args.result_path)

    args.data_name = args.dataset.split('_')[0]
    torch.manual_seed(args.dataset.split('_')[-1])

    task_datasets = torch.load(os.path.join(args.path, args.dataset + '.pt'))
    data_loader = task_data_loader(task_datasets, args)

    print(args)

    if args.data_name.startswith('cifar100') or args.data_name.startswith('miniimageNet'):
        if args.model == 'None':
            logger.info('init resnet model for {}'.format(args.data_name))
            model = ResNet18(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'ER':
            logger.info('init ER model for {}'.format(args.data_name))
            model = ER(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'iCaRL':
            logger.info('init iCaRL model for {}'.format(args.data_name))
            args.n_epochs += 1
            model = iCaRL(args)
            trainer = iCaRL_train(data_loader, model, args, logger)

        elif args.model == 'LwF':
            logger.info('init LwF model for {}'.format(args.data_name))
            model = LwF(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'EWC':
            logger.info('init EWC model for {}'.format(args.data_name))
            model = EWC(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'AGEM':
            logger.info('init AGEM model for {}'.format(args.data_name))
            model = A_GEM(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'EEIL':
            logger.info('init EEIL model for {}'.format(args.data_name))
            model = EEIL(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'UCIR':
            logger.info('init UCIR model for {}'.format(args.data_name))
            model = UCIR(args)
            trainer = base_train(data_loader, model, args, logger)

        elif args.model == 'prototype_decoder':
            logger.info('init prototype_decoder model for {}'.format(args.data_name))
            data_loader = task_data_loader(task_datasets, args)
            model = prototype_decoder(args)
            trainer = prototype_decoder_train(data_loader, model, args, logger)


        # model = EWC(ResNet18, task, args)
        # model = GEM(MLP, task, args)
        # model = SREP(ResNet18, task, args)

    trainer.train()













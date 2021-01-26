import argparse
import os
import torch
from src.train_helper import Trainer
import random
import numpy as np



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# train dir, val dir

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--comments', default='empty description', help='experiment description')
    parser.add_argument('--experiment', default='0.1.0', help='experiment number')
    parser.add_argument('--train-limit', default=1000, help='number of train samples used')
    parser.add_argument('--train-dir', default='', help='train path')
    parser.add_argument('--val-dataset', default='sht-b', help='dataset to evaluate')
    parser.add_argument('--train-dataset', default='blendercam', help='dataset to train')
    parser.add_argument('--val-limit', default=1000, help='number of val samples used')
    parser.add_argument('--val-dir', default='', help='val path')
    parser.add_argument('--model', default='resnet50', help='pretrained model')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-4,
                        help='max cyclic learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,#1,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')

    args = parser.parse_args()

    return args

def setSeed(manualSeed):
    	np.random.seed(manualSeed)
    	random.seed(manualSeed)
    	torch.manual_seed(manualSeed)
    	torch.cuda.manual_seed(manualSeed)
    	torch.cuda.manual_seed_all(manualSeed)

    	torch.backends.cudnn.enabled = False
    	torch.backends.cudnn.benchmark = False
    	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_args()

    setSeed(args.seed)

    #torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

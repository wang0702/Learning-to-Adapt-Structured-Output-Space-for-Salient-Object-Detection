import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import argparse
from datetime import datetime
import os
import os.path as osp

from dataset.testD_dataset import get_loader
from networks.poolnet import build_model, weights_init
from model.discriminator import FCDiscriminator

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 10
EPOCH_SIZE = 5
NUM_STEPS = 10000 # x
NUM_STEPS_STOP = 150000 # x

LEARNING_RATE = 5e-5
LEARNING_RATE_D = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POWER = 0.9 # ???

LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
DATA_DIRECTORY_TARGET = ''
DATA_LIST_PATH_TARGET = ''

SHOW_EVERY = 50
SAVE_PRED_EVERY = 5000
RESNET_PATH = './dataset/pretrained/resnet50_caffe.pth'
RESTORE_FROM = './dataset/pretrained/s8.pth'
D_RESTORE_FROM = './dataset/pretrained/D2.pth'
SNAPSHOT_DIR = './snapshots/'

def get_args():

    parser = argparse.ArgumentParser(description="Tsy-PoolNet Network")

    # size
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--epotch-size", type=int, default=EPOCH_SIZE,
                        help="epotch size.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")

    # detail
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")

    # super canshu
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")    

    # input
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")

    # other
    parser.add_argument("--show-every", type=int, default=SHOW_EVERY,
                        help="How many times show one show.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--pretrained-model", type=str, default=RESNET_PATH,
                        help="pretrained model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--D-restore-from", type=str, default=D_RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    
    return parser.parse_args()

args = get_args()
device = torch.device("cuda")

def pan(x):

    sum = 0
    tot = 0
    for j in x[0][0]:
        for i in j:
            if i < 0.5:
                sum += 1
            tot += 1
    if sum < tot // 2:
        return 1
    else:
        return 0

def main():

    # create the model
    model = build_model()
    model.to(device)
    model.load_state_dict(torch.load(args.restore_from))    

    # create domintor
    model_D1 = FCDiscriminator(num_classes=1)
    model_D1.to(device)
    model_D1.load_state_dict(torch.load(args.D_restore_from))    

    up = torch.nn.Upsample(scale_factor=32, mode='bilinear')
    sig = torch.nn.Sigmoid()

    # labels for adversarial training 两种域的记号
    salLabel = 0
    edgeLabel = 1

    picloader = get_loader(args)
    correct = 0
    tot = 0
    for i_iter, data_batch in enumerate(picloader):
        tot += 2

        sal_image, edge_image = data_batch['sal_image'], data_batch['edge_image']
        sal_image, edge_image = Variable(sal_image), Variable(edge_image)
        sal_image, edge_image = sal_image.to(device), edge_image.to(device)

        sal_pred = model(sal_image)
        edge_pred = model(edge_image)

        # test D
        # for param in model_D1.parameters():
        #     param.requires_grad = True

        ss_out = model_D1(sal_pred)
        se_out = model_D1(edge_pred)
        if pan(ss_out) == salLabel:
            correct += 1
        if pan(se_out) == edgeLabel:
            correct += 1

        if i_iter % 100 == 0:
            print('processing %d: %f' % (i_iter, correct / tot))

    print(correct / tot)


if __name__ == '__main__':
    main()

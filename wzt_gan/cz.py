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

from dataset.joint_dataset import get_loader
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
RESTORE_FROM = './dataset/pretrained/s7.pth'
D_RESTORE_FROM = ''
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

    # super argument
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

def main():

    # make dir
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    run = 0
    while os.path.exists("%s/run-%d" % (args.snapshot_dir, run)):
        run += 1
    os.mkdir("%s/run-%d" % (args.snapshot_dir, run))
    os.mkdir("%s/run-%d/models" % (args.snapshot_dir, run))
    args.file_dir = "%s/run-%d/file.txt" % (args.snapshot_dir, run)
    args.snapshot_dir = "%s/run-%d/models" % (args.snapshot_dir, run)

    # xuan xue you hua
    cudnn.enabled = True
    cudnn.benchmark = True
    
    # create the model
    model = build_model()
    model.to(device)
    model.train()
    model.apply(weights_init)
    model.load_state_dict(torch.load(args.restore_from))    
    # model.base.load_pretrained_model(torch.load(args.pretrained_model))

    # create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, weight_decay=args.weight_decay) # 整个模型的优化器
    optimizer.zero_grad()

    # start time
    with open(args.file_dir, 'a') as f:
        f.write('strat time: ' + str(datetime.now()) + '\n\n')

        f.write('learning rate: ' + str(args.learning_rate) + '\n')
        f.write('learning rate D: ' + str(args.learning_rate_D) + '\n')
        f.write('wight decay: ' + str(args.weight_decay) + '\n')
        f.write('lambda_adv_target2: ' + str(args.lambda_adv_target2) + '\n\n')

        f.write('eptch size: ' + str(args.epotch_size) + '\n')
        f.write('batch size: ' + str(args.batch_size) + '\n')
        f.write('iter size: ' + str(args.iter_size) + '\n')
        f.write('num steps: ' + str(args.num_steps) + '\n\n')

    picloader = get_loader(args)
    iter_num = len(picloader.dataset) // args.batch_size
    aveGrad = 0

    for i_epotch in range(args.epotch_size):
        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0
        model.zero_grad()

        for i_iter, data_batch in enumerate(picloader):
            
            sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
            if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                print('IMAGE ERROR, PASSING```')
                with open(args.file_dir, 'a') as f:
                    f.write('IMAGE ERROR, PASSING```\n')
                continue
            sal_image, sal_label = Variable(sal_image), Variable(sal_label)
            sal_image, sal_label = sal_image.to(device), sal_label.to(device)

            sal_pred = model(sal_image)
            sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
            sal_loss = sal_loss_fuse / (args.iter_size * args.batch_size)
            loss_seg_value1 += sal_loss.data

            sal_loss.backward()

            aveGrad += 1
            if aveGrad % args.iter_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0
            
            if i_iter % (args.show_every // args.batch_size) == 0:
                print(
                'epotch = {5:2d}/{6:2d}, iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}, loss_adv1 = {3:.3f}, loss_D1 = {4:.3f}'.format(
                    i_iter, iter_num, loss_seg_value1, loss_adv_target_value1, loss_D_value1, i_epotch, args.epotch_size))
                with open(args.file_dir, 'a') as f:
                    f.write(
                    'epotch = {5:2d}/{6:2d}, iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}, loss_adv1 = {3:.3f}, loss_D1 = {4:.3f}\n'.format(
                        i_iter, iter_num, loss_seg_value1, loss_adv_target_value1, loss_D_value1, i_epotch, args.epotch_size))    
            
                loss_seg_value1, loss_adv_target_value1, loss_D_value1 = 0, 0, 0
            
            if i_iter == iter_num - 1 or i_iter % args.save_pred_every == 0 and i_iter != 0:
                print('taking snapshot ...')
                with open(args.file_dir, 'a') as f:
                    f.write('taking snapshot ...\n')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'sal_' + str(i_epotch) + '_' + str(i_iter) + '.pth'))
                # torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'sal_' + str(i_epotch) + '_' + str(i_iter) + '_D1.pth'))

        if i_epotch == 7:
            args.learning_rate = args.learning_rate * 0.1
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    # end
    with open(args.file_dir, 'a') as f:
        f.write('end time: ' + str(datetime.now()) + '\n')

if __name__ == '__main__':
    main()

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
import cv2

from dataset.joint_dataset import get_loader
from networks.poolnet import build_model, weights_init

BATCH_SIZE = 1

DATA_DIRECTORY = ''
DATA_LIST_PATH = ''

RESTORE_FROM = './dataset/pretrained/s8.pth'
SNAPSHOT_DIR = './snapshots/ans/'

def get_args():

    parser = argparse.ArgumentParser(description="Tsy-PoolNet Network")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")

	# input
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")

    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    
    return parser.parse_args()

args = get_args()
device = torch.device("cuda")

def main():

    gtdir = args.snapshot_dir + 'gt/1/'
    preddir = args.snapshot_dir + 'pred/tsy1/1/'

    # make dir
    if not os.path.exists(gtdir):
        os.makedirs(gtdir)
    if not os.path.exists(preddir):
        os.makedirs(preddir)

    # xuan xue you hua
    cudnn.enabled = True
    cudnn.benchmark = True
    
    # create the model
    model = build_model()
    model.to(device)
    model.train()
    model.apply(weights_init)
    model.load_state_dict(torch.load(args.restore_from))    

    picloader = get_loader(args, mode='test')

    for i_iter, data_batch in enumerate(picloader):
           
        if i_iter % 50 == 0:
            print(i_iter)
        sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']

        with torch.no_grad():
            sal_image = Variable(sal_image).to(device)
            preds = model(sal_image)
            pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
            label = np.squeeze(sal_label.cpu().data.numpy())
            multi_fuse = 255 * pred
            label = 255 * label
            cv2.imwrite(os.path.join(preddir, str(i_iter) + '.jpg'), multi_fuse)
            cv2.imwrite(os.path.join(gtdir, str(i_iter) + '.png'), label)

		

if __name__ == '__main__':
    main()

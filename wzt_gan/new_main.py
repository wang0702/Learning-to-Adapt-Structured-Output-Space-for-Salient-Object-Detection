import argparse #添加参数
import torch #...
import torch.nn as nn#模型包
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter

from dataset.joint_dataset import get_loader
from networks.poolnet import build_model, weights_init

from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d

from datetime import datetime

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32) # 像素平均值，感觉应该是某种计算上的优化，通过偏移平均值来使得分布近似正态分布

MODEL = 'DeepLab'#模型名称
BATCH_SIZE = 1#每次批数量
ITER_SIZE = 10#迭代批次数
NUM_WORKERS = 4#使用多进程加载的进程数
DATA_DIRECTORY = './data/GTA5'#source训练数据存储路径
DATA_LIST_PATH = './dataset/gta5_list/train.txt'#存储训练数据名称文件的路径
IGNORE_LABEL = 255 #忽略的像素点？
INPUT_SIZE = '1280,720'# source图片像素
DATA_DIRECTORY_TARGET = './data/Cityscapes/data' # target文件地址
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt' # 名称文件路径
INPUT_SIZE_TARGET = '1024,512' # target图片像素
MOMENTUM = 0.9 #动量
NUM_CLASSES = 19 # 用来预测的D的层数
NUM_STEPS = 10000 #训练步数
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9 #？
RANDOM_SEED = 1234 #随机种子

RESTORE_FROM = './dataset/pretrained/s5.pth' #预训练参数
D_RESTORE_FROM = ''

SAVE_NUM_IMAGES = 2 #？
SAVE_PRED_EVERY = 5000 #？
SNAPSHOT_DIR = './snapshots/' #快照路径？
WEIGHT_DECAY = 0.0005 #权重衰减
LOG_DIR = './log' #日志路径

LEARNING_RATE = 5e-5 #学习率
LEARNING_RATE_D = 1e-4 #D的学习率

# 几个超参数的值
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'

EPOCH_SIZE = 5
SHOW_EVERY = 20 # 多少组显示
RESNET_PATH = './dataset/pretrained/resnet50_caffe.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    # 这里解释的很详细
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab") # 模型名称？
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes") # target图片名字
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.") # batchsize
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.") # itersize
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.") # 多线程的线程数
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.") # sourse文件的路径
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.") # sourse list的路径
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.") # 忽略掉的像素点的最低值？
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.") # 输入图片的size
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.") # target的文件路径
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.") # target list的路径
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.") # target的size
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.") # 我好像知道是什么意思了。应该指的是 train模式还是test模式，也就是.eval()和.train()的我问题
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.") # 主程序的学习率
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.") # 判别器的学习率
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.") # 超参数lambda_seg的值，以下三个见论文
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.") # 超参数
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.") # 超参数
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.") # 动量，优化的超参数
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.") #？
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") # 多少种类要预测
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.") # 迭代多少次？
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.") # 多少次提前结束，优化超参数
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.") # 优化超参数
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.") # 随机镜像图像
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.") # 随机伸缩图像
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.") # 随机种子
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.") # pretrained参数存放地点

    parser.add_argument("--D-restore-from", type=str, default=D_RESTORE_FROM,
                        help="Where restore model parameters from.") # pretrained参数存放地点

    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.") # 存多少张图片
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.") # 多少张图片存一次？
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.") # 存放模型的代号的地方？
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.") # 优化超参数
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.") # 使用cpu
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.") # 是否使用tensorboard
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.") # log的存放路径
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.") # ？

    parser.add_argument("--epotch-size", type=int, default=EPOCH_SIZE,
                        help="epotch size.") # ？
    parser.add_argument("--pretrained-model", type=str, default=RESNET_PATH,
                        help="pretrained model.") # ？                       
                        
    return parser.parse_args()


args = get_arguments() #得到变量
'''
#某种优化中的学习率计算 poly衰减？ 这个竟然是手动修改学习率
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

# 修改学习率
def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

# 修改判别器的学习率
def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def bce2d(input, target, reduction='sum'):
    assert(input.size() == target.size()) # 确保图片大小一致
    pos = torch.eq(target, 1).float() # label为1的部分为1，其他部分为0
    neg = torch.eq(target, 0).float() # label为0的部分为1，其他部分为0

    num_pos = torch.sum(pos) # 1像素数量
    num_neg = torch.sum(neg) # 0像素数量
    num_total = num_pos + num_neg # 总和

    alpha = num_neg  / num_total # 0部分比率
    beta = 1.1 * num_pos  / num_total # 1部分比率
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg # 构造了一个有趣的权重矩阵，1部分权值总和和0部分权值总和相同

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
'''

def main():
    """Create the model and start the training."""
    # device放GPU还是CPU
    device = torch.device("cuda" if not args.cpu else "cpu")
    cudnn.enabled = True #一种玄学优化

    # Create network
    # 重要一步 输入类别数
    model = build_model() # 生成一个由resnet组成的语义分割模型
    # 读取pretrained模型
    model.to(device)
    model.train()
    model.apply(weights_init)
    # model.load_state_dict(torch.load(args.restore_from))    
    model.base.load_pretrained_model(torch.load(args.pretrained_model))
    #设置model参数

    
    # 玄学优化
    cudnn.benchmark = True

    # init D 设置D 鉴别器
    '''
    model_D1 = FCDiscriminator(num_classes=1).to(device)

    model_D1.train()
    model_D1.to(device)
    model_D1.apply(weights_init)
    '''
    # 创建存放模型的文件夹
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    picloader = get_loader(args)
    

    # implement model.optim_parameters(args) to handle different models' lr setting
    
    # 优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, weight_decay=args.weight_decay) # 整个模型的优化器
    optimizer.zero_grad()

    #optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99)) # D1的优化器
    # optimizer_D1.zero_grad()

    # 损失函数
    bce_loss = torch.nn.BCEWithLogitsLoss() # sigmoid + BCE的完美组合

    '''
    # 两个改变size的上采样
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True) # 变为source input的上采样
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True) # 变为 target input的上采样 
    '''
    # save folder
    run = 0
    while os.path.exists("%s/run-%d" % (args.snapshot_dir, run)):
        run += 1
    os.mkdir("%s/run-%d" % (args.snapshot_dir, run))
    os.mkdir("%s/run-%d/models" % (args.snapshot_dir, run))
    args.file_dir = "%s/run-%d/file.txt" % (args.snapshot_dir, run)
    args.snapshot_dir = "%s/run-%d/models" % (args.snapshot_dir, run)

    # labels for adversarial training 两种域的记号
    source_label = 0
    target_label = 1

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


    for i_epotch in range(args.epotch_size):
        # 损失值置零
        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        model.zero_grad()
        # model_D1.zero_grad()

        loader_iter = enumerate(picloader)

        for i_iter in range(args.num_steps // args.batch_size // args.iter_size): # 迭代次数 大batch
            
            # 优化器梯度置零 + 调整学习率
            optimizer.zero_grad()
            # adjust_learning_rate(optimizer, i_iter)

            # optimizer_D1.zero_grad()
            # adjust_learning_rate_D(optimizer_D1, i_iter)

            for sub_i in range(args.iter_size): # 迭代次数 小batch
                
                # get picture
                _, data_batch = loader_iter.__next__() # 获取一组图片
                source_images, source_labels, target_images = data_batch['sal_image'], data_batch['sal_label'], data_batch['edge_image']#, data_batch['edge_label']
                source_images, source_labels, target_images = Variable(source_images), Variable(source_labels), Variable(target_images)

                if (source_images.size(2) != source_labels.size(2)) or (source_images.size(3) != source_labels.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    with open(args.file_dir, 'a') as f:
                        f.write('IMAGE ERROR, PASSING```\n')
                    continue

                # 放入GPU
                source_images = source_images.to(device)
                source_labels = source_labels.to(device)
                target_images = target_images.to(device)
                pred1 = model(source_images) # 三层block和四层block之后classify之后的结果（相当于两种层的结果）
                # pred_target1 = model(target_images) # 放入模型
                
                # train G

                # don't accumulate grads in D 不需要D的梯度，因为这里是用D来辅助训练G
                # for param in model_D1.parameters():
                #     param.requires_grad = False

                # train with source
                # 计算损失函数
                loss_seg1 = F.binary_cross_entropy_with_logits(pred1, source_labels, reduction='sum')
                lossG = loss_seg1 / args.iter_size / args.batch_size
                loss_seg_value1 += lossG.item()  # 记录这次的iter的结果，显示相关和训练不相关

                lossG.backward()
                '''
                # D_out1 = model_D1(F.softmax(pred_target1)) # 放入鉴别器（不知道为什么要softmax）
                D_out1 = model_D1(pred_target1)
                # 这里用的是bceloss 训练G的时候，target判别为sourse_label时损失函数低
                loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device)) # 后面一个相当于全部是正确答案的和前一个size相同的tensor
                lossD = loss_adv_target1 / args.iter_size
                loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size # 记录专用
                
                lossD = lossD * args.lambda_adv_target2
                lossD.backward()
                
                # train D
                
                # bring back requires_grad 恢复D的grad
                for param in model_D1.parameters():
                    param.requires_grad = True

                pred1 = pred1.detach()# train with source 脱离grad
                # D_out1 = model_D1(F.softmax(pred1))# sourse的判别结果
                D_out1 = model_D1(pred1)

                # 训练D时sourse判断成sourse损失函数低
                loss_Ds = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
                loss_Ds = loss_Ds / args.iter_size
                loss_D_value1 += loss_Ds.item()# 显示专用

                pred_target1 = pred_target1.detach()# train with target target数据训练 脱离
                # D_out1 = model_D1(F.softmax(pred_target1))# 得到判别结果
                D_out1 = model_D1(pred_target1)# 得到判别结果

                # taget判别为target时损失函数低
                loss_Dt = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
                loss_Dt = loss_Dt / args.iter_size
                loss_D_value1 += loss_Dt.item()# 显示专用

                loss_Ds.backward()
                loss_Dt.backward()
                '''
            # 修改一次参数
            optimizer.step()
            # optimizer_D1.step()

            '''
            # 不管
            if args.tensorboard:
                scalar_info = {
                    'loss_seg1': loss_seg_value1,
                    'loss_seg2': loss_seg_value2,
                    'loss_adv_target1': loss_adv_target_value1,
                    'loss_adv_target2': loss_adv_target_value2,
                    'loss_D1': loss_D_value1,
                    'loss_D2': loss_D_value2,
                }

                if i_iter % 10 == 0:
                    for key, val in scalar_info.items():
                        writer.add_scalar(key, val, i_iter)
            '''
            
            # 显示
            if i_iter*args.batch_size % SHOW_EVERY == 0:
                print('exp = {}'.format(args.snapshot_dir))
                print(
                'epotch = {5:2d}/{6:2d}, iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}, loss_adv1 = {3:.3f}, loss_D1 = {4:.3f}'.format(
                    i_iter, args.num_steps//args.batch_size//args.iter_size, loss_seg_value1, loss_adv_target_value1, loss_D_value1, i_epotch, args.epotch_size))
                with open(args.file_dir, 'a') as f:
                    f.write('exp = {}\n'.format(args.snapshot_dir))
                    f.write(
                    'epotch = {5:2d}/{6:2d}, iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f}, loss_adv1 = {3:.3f}, loss_D1 = {4:.3f}\n'.format(
                        i_iter, args.num_steps//args.batch_size//args.iter_size, loss_seg_value1, loss_adv_target_value1, loss_D_value1, i_epotch, args.epotch_size))    
            
                loss_seg_value1, loss_adv_target_value1, loss_D_value1 = 0, 0, 0
            # 提前终止
            if i_iter >= args.num_steps_stop - 1:
                print('save model ...')
                with open(args.file_dir, 'a') as f:
                    f.write('save model ...\n')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'sal_' + str(i_epotch) + '_' + str(args.num_steps_stop) + '.pth'))
                # torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'sal_' + str(i_epotch) + '_' + str(args.num_steps_stop) + '_D1.pth'))
                break
            
            if i_iter == args.num_steps//args.batch_size//args.iter_size - 1 or i_iter*args.batch_size*args.iter_size % args.save_pred_every == 0 and i_iter != 0:
                print('taking snapshot ...')
                with open(args.file_dir, 'a') as f:
                    f.write('taking snapshot ...\n')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'sal_' + str(i_epotch) + '_' + str(i_iter) + '.pth'))
                # torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'sal_' + str(i_epotch) + '_' + str(i_iter) + '_D1.pth'))
            
    '''
    if args.tensorboard:
        writer.close()
    '''
    with open(args.file_dir, 'a') as f:
        f.write('end time: ' + str(datetime.now()) + '\n')

if __name__ == '__main__':
    main()

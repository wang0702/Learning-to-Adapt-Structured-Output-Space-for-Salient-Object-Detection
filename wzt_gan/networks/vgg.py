import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F

# vgg16 制作指定的vgg卷积层list
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)] #经典池化
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)] #size减半的池化
        else:
            if stage == 6:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) #经典卷积
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) #估计有一些格式化的考量
            if batch_norm: # 做不做规范化
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] #卷积后面都要加上线性整流函数
            in_channels = v
    return layers

# vgg16 类
class vgg16(nn.Module):
    #初始化
    def __init__(self):
        super(vgg16, self).__init__() #父类初始化
        self.cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'tun_ex': [512, 512, 512]} #模型的结构表
        self.extract = [8, 15, 22, 29] # [3, 8, 15, 22, 29]
        self.base = nn.ModuleList(vgg(self.cfg['tun'], 3)) #基础的卷积层list
        # 初始化值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # 加载预训练模型
    def load_pretrained_model(self, model):
        self.base.load_state_dict(model, strict=False)

    # 将参数放入模型中并返回指定几层的结果
    def forward(self, x):
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

class vgg16_locate(nn.Module):
    def __init__(self):
        super(vgg16_locate,self).__init__() # 初始化父类
        self.vgg16 = vgg16() #建立一个vgg16模型
        self.in_planes = 512 #？
        self.out_planes = [512, 256, 128] #？

        ppms, infos = [], []

        # 制作ppm
        for ii in [1, 3, 5]:
            #  ii为边的正方形自适应平均池化 + 核为1x1的卷积（为什么我觉得就是同乘一个数） + 线性整流函数
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms) #组合为modulelist
        # 这个暂时不知道是干什么的 一个4*in_planes -> in_planes的经典卷积 + 线性整流
        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))

        # in_planes -> out_planes[x]的经典卷积 + 线性整流
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos) #组合为modulelist

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # 读取预训练模型
    def load_pretrained_model(self, model):
        self.vgg16.load_pretrained_model(model)

    # 将参数放入模型得到结果
    def forward(self, x):
        x_size = x.size()[2:] #只搞前两个维度
        xs = self.vgg16(x) #先做一遍vgg16
        
        # xs[-1]应该是最后一个extract的元素的层（self.extract[-1]层）的结果
        xls = [xs[-1]]
        # 添加进入了每种ppm的的参数
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs[-1]), xs[-1].size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1)) # 把xls的参数接在一起做一次ppm_cat
        
        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

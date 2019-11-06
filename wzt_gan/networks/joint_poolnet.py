import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from .deeplab_resnet import resnet50_locate
from .vgg import vgg16_locate

# 两种主干的参数
config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 256, 'edgeinfoc':[48,128], 'block': [[512, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16], True]}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True]}

# 隐藏层 把list的每个元素按照顺序换为指定channel + 整流
class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__() #上级初始化
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True))) # 根据参数变换层数，1x1卷积 + 线性整流
        self.convert0 = nn.ModuleList(up)

    # 把list_x的元素按顺序分别放入指定的模型
    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

# 深度池化模型 本体 + 2倍 4倍 8倍 缩小体稀释相加 + 转换为k_out层
class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__() # 上级初始化
        self.pools_sizes = [2,4,8] 
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i)) # 进行平均下采样 2倍 4倍 8倍
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False)) # k层->k层 经典33卷积
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False) # k->k_out 的经典33卷积
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False) #如果需要熔断就再加一个 k_out -> k_out 的经典33卷积

    def forward(self, x, x2=None, x3=None):
        x_size = x.size() # 得到size
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x)) # 先平均池化再33卷积
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True)) # 再稀释回原来的大小并加在x上面
        resl = self.relu(resl) # 线性整流
        # 如果需要把大小变成x2形态
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl) # 变成k_out层
        # 如果需要熔断，把结果和x2与x3相加后 再进行一次卷积
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

# 阻塞层 （除4层 + 33卷积 + 乘四层 + 整流）x 2 + 变为指定层数
class BlockLayer(nn.Module):
    def __init__(self, k_in, k_out_list):
        super(BlockLayer, self).__init__() # 上级初始化
        up_in1, up_mid1, up_in2, up_mid2, up_out = [], [], [], [], []

        for k in k_out_list:
            up_in1.append(nn.Conv2d(k_in, k_in//4, 1, 1, bias=False)) # 层数除四 的经典变层模型
            up_mid1.append(nn.Sequential(nn.Conv2d(k_in//4, k_in//4, 3, 1, 1, bias=False), nn.Conv2d(k_in//4, k_in, 1, 1, bias=False))) # 进行一次经典33卷积再变回原来层数
            up_in2.append(nn.Conv2d(k_in, k_in//4, 1, 1, bias=False)) # 层数除四 的经典变层模型
            up_mid2.append(nn.Sequential(nn.Conv2d(k_in//4, k_in//4, 3, 1, 1, bias=False), nn.Conv2d(k_in//4, k_in, 1, 1, bias=False))) # 进行一次经典33卷积再变回原来层数
            up_out.append(nn.Conv2d(k_in, k, 1, 1, bias=False)) # kin->k的层卷积

        self.block_in1 = nn.ModuleList(up_in1)
        self.block_in2 = nn.ModuleList(up_in2)
        self.block_mid1 = nn.ModuleList(up_mid1)
        self.block_mid2 = nn.ModuleList(up_mid2)
        self.block_out = nn.ModuleList(up_out)
        self.relu = nn.ReLU()

    def forward(self, x, mode=0):
        x_tmp = self.relu(x + self.block_mid1[mode](self.block_in1[mode](x))) # 除4层 + 33卷积 + 乘四层 + 整流
        # x_tmp = self.block_mid2[mode](self.block_in2[mode](self.relu(x + x_tmp)))
        x_tmp = self.relu(x_tmp + self.block_mid2[mode](self.block_in2[mode](x_tmp))) # 除4层 + 33卷积 + 乘四层 + 整流
        x_tmp = self.block_out[mode](x_tmp) # 变为指定输出层

        return x_tmp

# 边缘信息层 将list元素变为指定大小 + 接起来 + 4x 33卷积
class EdgeInfoLayerC(nn.Module):
    def __init__(self, k_in, k_out):
        super(EdgeInfoLayerC, self).__init__() # 上级初始化
        self.trans = nn.Sequential(nn.Conv2d(k_in, k_in, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_in, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False), nn.ReLU(inplace=True)) #4x 33卷积

    def forward(self, x, x_size):
        tmp_x = []
        for i_x in x:
            tmp_x.append(F.interpolate(i_x, x_size[2:], mode='bilinear', align_corners=True)) # 把x中的元素变为xsize大小
        x = self.trans(torch.cat(tmp_x, dim=1)) #接起来进行上面的卷积
        return x

# 熔断层 [把列表元素变为一层 + 变为指定大小 + 接起来 + 变为一层, 列表元素变为一层组成的list]
class FuseLayer1(nn.Module):
    def __init__(self, list_k, deep_sup):
        super(FuseLayer1, self).__init__() # 上级的初始化
        up = []
        for i in range(len(list_k)):
            up.append(nn.Conv2d(list_k[i], 1, 1, 1)) # 变为一层的变层卷积
        self.trans = nn.ModuleList(up)
        self.fuse = nn.Conv2d(len(list_k), 1, 1, 1) # list_k层变一层
        self.deep_sup = deep_sup

    def forward(self, list_x, x_size):
        up_x = []
        for i, i_x in enumerate(list_x):
            up_x.append(F.interpolate(self.trans[i](i_x), x_size[2:], mode='bilinear', align_corners=True)) # 把list_x中的元素变为1层再稀释为指定大小
        out_fuse = self.fuse(torch.cat(up_x, dim = 1)) # 把这些接起来再变成一层
        if self.deep_sup:
            out_all = []
            for up_i in up_x:
                out_all.append(up_i) # copy
            return [out_fuse, out_all] # 应该是有什么顾虑，比如梯度之类的东西所以必须拷贝 这个是重点
        else:
            return [out_fuse]

# 分数层 k->1的33卷积（可以变指定size）
class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__() # 上级初始化
        self.score = nn.Conv2d(k ,1, 3, 1, 1) # k->1 的33卷积

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True) # 可以变为指定size
        return x

# 制造各种需要的layer
def extra_layer(base_model_cfg, base):
    # 选主干 vgg/resnet
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers = [], [], [], [], [], []
    
    # 制造各个层
    convert_layers = ConvertLayer(config['convert'])

    for k in config['block']:
        block_layers += [BlockLayer(k[0], k[1])]

    for i in range(len(config['deep_pool'][0])):
       deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    fuse_layers = FuseLayer1(config['fuse'][0], config['fuse'][1])

    edgeinfo_layers = EdgeInfoLayerC(config['edgeinfoc'][0], config['edgeinfoc'][1])

    score_layers = ScoreLayer(config['score'])

    return base, convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers

# 组装PoolNet
class PoolNet(nn.Module):
    #           自己    主干模型名字     基础？    隐藏卷积层？    深度池化网络？    阻塞卷积层？   融合层？      边缘信息层？       得分层？
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers):
        super(PoolNet, self).__init__() # 上级初始化
        self.base_model_cfg = base_model_cfg # 使用主干的名字
        self.base = base # 主干模型

        # 各种 extra_layer
        self.block = nn.ModuleList(block_layers) #装module类的list类
        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.fuse = fuse_layers
        self.edgeinfo = edgeinfo_layers
        self.score = score_layers

        # vvg没有covert_layer
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

    def forward(self, x, mode):
        x_size = x.size() # 参数的大小
        conv2merge, infos = self.base(x) # 中间变量, ppm + 连接 + 变为指定中间变量size + 33卷积 的参数list

        # resnet变层数
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1] # reverse(info因为本来就是倒着的所以不需要reverse)
        

        edge_merge = [] # 存放中间变量
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        edge_merge.append(merge)
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])
            edge_merge.append(merge)
            
        # 这个mode很迷
        if mode == 0:
            edge_merge = [self.block[i](kk) for i, kk in enumerate(edge_merge)] # 先把那四个参数放到block中
            merge = self.fuse(edge_merge, x_size) # 压缩成一层，size为x原来的size
        elif mode == 1:
            merge = self.deep_pool[-1](merge) # 先再做一次deep_pool
            edge_merge = [self.block[i](kk).detach() for i, kk in enumerate(edge_merge)] # 将四个参数放入block中（这里使用了分离操作，是因为这里不需要学习吗？）
            edge_merge = self.edgeinfo(edge_merge, merge.size()) # 做边缘信息模型
            merge = self.score(torch.cat([merge, edge_merge], dim=1), x_size) # 把上面做的和新做的merge连接起来做一次score
        return merge

# 建立网络 base层: 主干网络(vgg/resnet) + ppm处理 + 连接 + 分发进入每一中间层(变为其size + 33卷积) 返回: 中间参数list + 分发处理的ppm参数list
def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))
# 初始化参数
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

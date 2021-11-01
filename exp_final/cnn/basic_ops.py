import torch
import torch.nn as nn
from torch.nn import Conv2d
import math
import torch.nn.functional as F

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
def wn(x):
    return torch.nn.utils.weight_norm(x)
def act(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
def Conv1x1(in_channels, out_channels, bias=True):
    return Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias)
def DepthWiseConv(channel_in, channel_out, bias=False):
    return Conv2d(channel_in, channel_out, kernel_size=3, padding=3//2, groups=channel_in, bias=bias)
def Channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class Shuffle(nn.Module):
    def __init__(self, C_in, C_out):
        super(Shuffle, self).__init__()
        self.conv = wn(Conv1x1(C_in, C_out))

    def forward(self, x):
        x_shuff = Channel_shuffle(x, groups=2)
        y = self.conv(x_shuff)
        # print(x.shape, x_shuff.shape)
        return y
# upsampling

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, wn, act):
        m = []
        n_feats_inner = n_feats // 2
        m.append(wn(Conv1x1(n_feats, n_feats_inner)))
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(wn(Conv2d(n_feats_inner, 4 * n_feats_inner, 3, padding=1)))
                m.append(nn.PixelShuffle(2))
                if act: m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(wn(Conv2d(n_feats_inner, 9 * n_feats_inner, 3, padding=1)))
            m.append(nn.PixelShuffle(3))
            if act is not None: m.append(act)
        else:
            raise NotImplementedError
        m.append(wn(Conv1x1(n_feats_inner, n_feats)))

        super(Upsampler, self).__init__(*m)

'''
CALayerv1: from RCAN_2018
CALayerv2: from CBAM_2018
PALayer: from CANet_2020
SALayerv1: from CBAM_2018
SALayerv2: from BAM_2018
DALayer: from MIRNet_2020
CBAM, BAM
PAM, CAM from DANet_2019
SKFF: from MIRNET origin from SKNet
NLB: from RNAN
Laplacian Attention Layer: from DRLN
Contrast-Aware Channel Attention: from IMDN_ACM2019
'''

class CALayerv2(nn.Module):
    def __init__(self, channel, reduction=16, act=nn.ReLU(inplace=True), bias=True):
        super(CALayerv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv_du = nn.Sequential(
            Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            act,
            Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y1 = self.conv_du(y1)

        y2 = self.max_pool(x)
        y2 = self.conv_du(y2)

        y = self.sigmoid(y1+y2)
        return x*y

class SALayer(nn.Module):
    def __init__(self, kernel_size=5):
        super(SALayer, self).__init__()
        self.compress = ChannelPool()
        self.op = nn.Sequential(
            wn(default_conv(2, 1, kernel_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = self.compress(x)
        y = self.op(x_compress)
        return x*y
class SALayerv2(nn.Module):
    def __init__(self, channel, reduction=4, dilation=4, act=nn.ReLU(inplace=True), bias=True):
        super(SALayerv2, self).__init__()
        self.conv_du = nn.Sequential(
            Conv1x1(channel, channel//reduction, bias=True),
            Conv2d(channel//reduction, channel//reduction, kernel_size=3, padding=dilation, dilation=dilation,
                      bias=bias),
            act,
            Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=dilation, dilation=dilation,
                      bias=bias),
            act,
            Conv1x1(channel//reduction, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x*y
class DALayer(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, conv=default_conv, bias=False, act=nn.ReLU(True)):
        super(DALayer, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias), act, conv(n_feat, n_feat, kernel_size, bias)]
        self.body = nn.Sequential(*modules_body)

        self.SA = SALayer()
        self.CA = CALayerv2(channel=n_feat, reduction=reduction, act=act, bias=True)

        self.conv1x1 = Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa = self.SA(res)
        # print(sa.shape)
        ca = self.CA(res)
        res = torch.cat([sa, ca], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

class PAM(nn.Module):
    """Position attention"""
    def __init__(self, channel, reduction=8):
        super(PAM, self).__init__()
        self.query_conv = Conv2d(channel, channel//reduction, kernel_size=1)
        self.key_conv = Conv2d(channel, channel//reduction, kernel_size=1)
        self.value_conv = Conv2d(channel, channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # print('proj_query: ', proj_query.shape)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # print(proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)  # (HxW)x(HxW)
        # print(energy.shape)
        attention = self.softmax(energy)
        # print(attention.shape)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # print(proj_value.shape)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = torch.bmm(proj_value, attention)

        out = out.view(m_batchsize, C, height, width)
        # print(self.gamma)
        out = self.gamma * out + x
        return out
class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            out : attention value + input feature
            attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
# Selective Kernel Feature Fusion (SKFF)
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            Conv2d(in_channels, d, 1, padding=0, bias=bias),
            nn.PReLU()
        )

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        # inp_feats is a list of tensor which has same resolution size from three scale branches
        batch_size = inp_feats[0].shape[0]
        # print(batch_size)
        n_feats = inp_feats[0].shape[1]
        # print(n_feats)

        inp_feats = torch.cat(inp_feats, dim=1)  # channel 64*3
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)  # 3 64 1 1
        feats_Z = self.conv_du(feats_S)  # 64=>8

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)  # 3 192 1 1
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)  # 3 3 64 1 1
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V
# NonLocalBlock2D
class NLB(nn.Module):
    def __init__(self, channel, reduction):
        super(NLB, self).__init__()

        self.channel_inter = channel//reduction

        self.g = Conv2d(channel, self.channel_inter, kernel_size=1)
        self.phi = Conv2d(channel, self.channel_inter, kernel_size=1)
        self.theta = Conv2d(channel, self.channel_inter, kernel_size=1)

        self.W = nn.Conv2d(self.channel_inter, channel, kernel_size=1)

    def forward(self, x):
        batch_size, H, W = x.size(0), x.size(2), x.size(3)
        g_x = self.g(x).view(batch_size, self.channel_inter, -1).permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.channel_inter, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.channel_inter, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous().view(batch_size, self.channel_inter, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z
# LaplacianAttention
class LaplacianAttentionLayer(nn.Module):
    def __init__(self, channel, reduction):
        super(LaplacianAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 空洞卷积: 卷积核视野相当于参数 (dilation-1)+ksize
        # 例如 dilation=3, ksize=3, 相当于卷积核为 (3-1)*2=5 的卷积
        self.c1 = Conv2d(channel, channel//reduction, kernel_size=3, stride=1, padding=3, dilation=3)
        self.c2 = Conv2d(channel, channel//reduction, kernel_size=3, stride=1, padding=5, dilation=5)
        self.c3 = Conv2d(channel, channel//reduction, kernel_size=3, stride=1, padding=7, dilation=7)
        self.c4 = Conv2d((channel//reduction)*3, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y
# 统计标准差
class ContrastAwareCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ContrastAwareCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        # print(self.contrast(x).shape, self.avg_pool(x).shape)
        y = self.conv_du(y)
        return x * y

def main():
    from torchstat import stat
    B = 3
    C = 16
    H = 48
    W = 48
    # nlb = NLB(channel=16, reduction=4)
    # print(nlb)
    # out = nlb(x)
    # print(out.shape)
    # gloConv = GlobalDepthWiseConv(16, 16, kernel_size=(x.size(-2), x.size(-1)))
    # out = gloConv(x)
    # print(out.shape)
    # stat(gloConv, (16, 16, 16))
    # lap = LaplacianAttentionLayer(channel=C, reduction=4)
    # out = lap(x)
    # print(out.shape)

    # con = ContrastAwareCALayer(channel=C, reduction=4)
    # out = con(x)
    # print(out.shape)
    up = Upsampler(scale=2, n_feats=64, wn=wn, act=None)
    print(up)
if __name__ == '__main__':
    main()


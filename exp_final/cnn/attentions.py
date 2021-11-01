import torch
import torch.nn as nn
from cnn.basic_ops import Conv1x1, default_conv, ChannelPool, wn, stdv_channels, DepthWiseConv
from torch.nn import Conv2d
import torch.nn.functional as F

ATS = {
    'none': lambda C: Zero(),
    'skip_connect': lambda C: Identity(),
    'pixel_wise_attention': lambda C: PixelWiseAttention(C, r=8),  # from CANet_2020  C=64 541
    'channel_wise_attention': lambda C: ChannelWiseAttention(C, r=8),  # from RCAN C=64 1168
    'contrast_aware_channel_attention': lambda C: ContrastAwareChannelAttention(C, r=8, act=nn.ReLU(True)),  # IMDN_ACM2019  C=64 1096
    'spatial_attention': lambda C: SpatialAttention(),  # from CBAM_ECCV2018  C=64 51
    'spatial_attention_v2': lambda C: SpatialAttentionv2(C, r=8, act=nn.ReLU(True)),  # from BAM_BMVC2018 C=64 1697
    'esab': lambda C: ESAB(C, r=8),  # RFANet_CVPR2020  C=64 3504
    'cea': lambda C: CEA(C, r=8),  # MAFFSRN  C=64 1096

    'pa_sa_cascade': lambda C: PA_CA_cascade(C, r=16, act=nn.ReLU(True)),
    'ca_sa_cascade': lambda C: CA_SA_cascade(C, r=16, kernel_size=5, act=nn.ReLU(True)),
    'ca_sa_parallel': lambda C: CA_SA_parallel(C, r=16, dilation=4, act=nn.ReLU(True)),
    'sa_ca_dual': lambda C: SA_CA_dual(C, r=16, kernel_size=5, act=nn.ReLU(True)),
    'pa_ca_parallel': lambda C: PA_CA_parallel(C, r=16),  # 计算量太大
}
# from CANet_2020

class PixelWiseAttention(nn.Module):
    def __init__(self, C, r, act=nn.ReLU(True)):
        super(PixelWiseAttention, self).__init__()
        self.at = nn.Sequential(
            Conv1x1(C, C//r),
            act,
            Conv1x1(C//r, 2),
            act,
            Conv1x1(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.at(x) * x
class ChannelWiseAttention(nn.Module):
    def __init__(self, C, r, act=nn.ReLU(True)):
        super(ChannelWiseAttention, self).__init__()
        self.at = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(Conv1x1(C, C//r)),
            act,
            wn(Conv1x1(C//r, C)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.at(x) * x
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.at = nn.Sequential(
            ChannelPool(),  # C: 64=>2
            default_conv(2, 1, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.at(x) * x
class SpatialAttentionv2(nn.Module):
    def __init__(self, C, r, act=nn.ReLU(True)):
        super(SpatialAttentionv2, self).__init__()
        d = 4
        k = 3
        self.at = nn.Sequential(
            Conv1x1(C, C//r),
            Conv2d(C//r, C//r, kernel_size=k, padding=d*(k - 1)//2, dilation=d),
            act,
            Conv2d(C//r, C//r, kernel_size=k, padding=d*(k - 1)//2, dilation=d),
            act,
            Conv1x1(C//r, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.at(x) * x
class ESAB(nn.Module):
    def __init__(self, C, r=4):
        super(ESAB, self).__init__()
        f = C // r
        self.conv1 = Conv1x1(C, f)
        self.conv_f = Conv1x1(f, f)
        self.conv_max = default_conv(f, f, kernel_size=3)
        self.conv2 = Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = default_conv(f, f, kernel_size=3)
        self.conv3_ = default_conv(f, f, kernel_size=3)
        self.conv4 = Conv1x1(f, C)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)

        return x*m
class ContrastAwareChannelAttention(nn.Module):
    def __init__(self, C, r=8, act=nn.ReLU(True)):
        super(ContrastAwareChannelAttention, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            Conv1x1(C, C//r, 1),
            act,
            Conv1x1(C//r, C),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        # print(self.contrast(x).shape, self.avg_pool(x).shape)
        y = self.conv_du(y)
        return x * y
class CEA(nn.Module):
    def __init__(self, C, r=2):
        super(CEA, self).__init__()
        self.at = nn.Sequential(
            Conv1x1(C, C//r),
            DepthWiseConv(C//r, C),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.at(x) * x


class PA_CA_cascade(nn.Module):
    """
    Pixel-wise Attention and Channel-wise Attention
    Combination in a cascade way
    """
    def __init__(self, C, r, act=nn.ReLU(True)):
        super(PA_CA_cascade, self).__init__()
        self.pa = nn.Sequential(
            wn(Conv1x1(C, C//r)),
            act,
            wn(Conv1x1(C//r, 2)),
            act,
            wn(Conv1x1(2, 1)),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(Conv1x1(C, C//r)),
            act,
            wn(Conv1x1(C//r, C)),
            nn.Sigmoid()
        )

    def forward(self, x):
        PA = self.pa(x)
        x = PA * x
        CA = self.ca(x)
        y = CA * x
        return y

# from CBAM_ECCV2018
class CA_SA_cascade(nn.Module):
    """
    Channel Attention Module and Spatial Attention Module
    Combination in a cascade way
    """
    def __init__(self, C, r, kernel_size=5, act=nn.ReLU(True)):
        super(CA_SA_cascade, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(Conv1x1(C, C//r)),
            act,
            wn(Conv1x1(C//r, C)),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            ChannelPool(),  # C: 64=>2
            wn(default_conv(2, 1, kernel_size)),
            nn.Sigmoid()
        )

    def forward(self, x):
        CA = self.ca(x)
        x = CA * x
        SA = self.sa(x)
        y = SA * x
        return y

# from BAM_BMVC2018
class CA_SA_parallel(nn.Module):
    """
    Channel Attention Module and Spatial Attention Module
    Combination in a parallel way
    """
    def __init__(self, C, r, dilation=4, act=nn.ReLU(True)):
        super(CA_SA_parallel, self).__init__()
        kernel_size = 3
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(Conv1x1(C, C//r)),
            act,
            wn(Conv1x1(C//r, C))
        )
        self.sa = nn.Sequential(
            wn(Conv1x1(C, C//r)),
            wn(Conv2d(C//r, C//r, kernel_size=kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)),
            act,
            wn(Conv2d(C//r, C//r, kernel_size=kernel_size, padding=dilation * (kernel_size - 1)//2, dilation=dilation)),
            act,
            Conv1x1(C//r, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        CA = self.ca(x)
        SA = self.sa(x)
        y = 1 + self.sigmoid(CA + SA)
        return x * y

# from MIRNet_ECCV2020
class SA_CA_dual(nn.Module):
    """
    spatial and channel attention
    """
    def __init__(self, C, r, kernel_size=5, act=nn.ReLU(True)):
        super(SA_CA_dual, self).__init__()
        self.sa = nn.Sequential(
            ChannelPool(),  # C: 64=>2
            wn(default_conv(2, 1, kernel_size)),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            wn(Conv1x1(C, C // r)),
            act,
            wn(Conv1x1(C // r, C)),
            nn.Sigmoid()
        )
        self.conv1x1 = Conv1x1(C*2, C)

    def forward(self, x):
        res = x
        SA = self.sa(res) * res
        CA = self.ca(res) * res
        res = torch.cat([SA, CA], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

# from DANet_CVPR2019
class PA_CA_parallel(nn.Module):
    """
    position attention module and Channel attention module
    """
    def __init__(self, C, r, act=nn.ReLU(True)):
        kernel_size = 3
        super(PA_CA_parallel, self).__init__()
        self.process_conv1 = default_conv(C, C//r, kernel_size=kernel_size)
        self.process_conv2 = default_conv(C, C//r, kernel_size=kernel_size)

        self.query_conv = wn(Conv1x1(C//r, C//r))
        self.key_conv = wn(Conv1x1(C//r, C//r))
        self.value_conv = wn(Conv1x1(C//r, C//r))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.tail_conv1 = default_conv(C//r, C, kernel_size=kernel_size)
        self.tail_conv2 = default_conv(C//r, C, kernel_size=kernel_size)

    def forward(self, x):
        """
        position att: B X (HxW) X (HxW)
        channel att: B X C X C
        """
        f1 = self.process_conv1(x)  # channel: C=>C//r
        f2 = self.process_conv2(x)

        B, C, H, W = f1.size()
        f1_query = self.query_conv(f1).view(B, -1, H*W).permute(0, 2, 1)
        f1_key = self.key_conv(f1).view(B, -1, H*W)
        f1_energy = torch.bmm(f1_query, f1_key)
        position_att = self.softmax(f1_energy)
        f1_value = self.value_conv(f1).view(B, -1, H*W)
        position_out = torch.bmm(f1_value, position_att)
        position_out = position_out.view(B, C, H, W)
        position_out = self.gamma1 * position_out + f1
        position_out = self.tail_conv1(position_out)

        f2_query = f2.view(B, C, -1)
        f2_key = f2.view(B, C, -1).permute(0, 2, 1)
        f2_energy = torch.bmm(f2_query, f2_key)
        f2_energy_new = torch.max(f2_energy, -1, keepdim=True)[0].expand_as(f2_energy) - f2_energy
        channel_att = self.softmax(f2_energy_new)
        f2_value = f2.view(B, C, -1)
        channel_out = torch.bmm(channel_att, f2_value)
        channel_out = channel_out.view(B, C, H, W)
        channel_out = self.gamma2 * channel_out + f2
        channel_out = self.tail_conv2(channel_out)

        return position_out + channel_out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)

def main():
    from torchstat import stat
    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()

        return num_params
    B = 3
    C = 64
    H = 43
    W = 29
    x = torch.randn(B, C, H, W)
    att = ATS['cea'](C)
    print(att)
    print('param computed by print_nerwork:{}'.format(print_network(att)))

    out = att(x)
    print(out.shape)
    stat(att, (64, 48, 48))


if __name__ == '__main__':
    main()

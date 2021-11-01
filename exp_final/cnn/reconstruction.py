import torch
import torch.nn as nn
from cnn.attentions import ChannelWiseAttention
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, C):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(C, C, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)
        out = self.k3(x) * y
        return out
class BFM(nn.Module):
    '''
    LatticeNet_ECCV2020
    '''
    def __init__(self, C, layers):
        super(BFM, self).__init__()
        self.layers = layers
        self.parallel_conv = nn.ModuleList()
        self.flow_conv = nn.ModuleList()

        for i in range(layers):
            self.parallel_conv.append(nn.Sequential(nn.Conv2d(C, C//2, 1), nn.ReLU()))
        for i in range(layers-1):
            if i == layers-2:
                self.flow_conv.append(nn.Sequential(nn.Conv2d(C, C, 1), nn.ReLU()))
            else:
                self.flow_conv.append(nn.Sequential(nn.Conv2d(C, C//2, 1), nn.ReLU()))

    def forward(self, feats):
        f = []
        for i in range(self.layers):
            if i == 0:
                tmp = self.parallel_conv[i](feats[-1])
            else:
                tmp = self.parallel_conv[i](feats[-(i+1)])
                tmp = torch.cat([tmp, f[-1]], 1)
                tmp = self.flow_conv[i-1](tmp)
            f.append(tmp)
            # print(len(f), tmp.shape)
        return f[-1]
class PFFM(nn.Module):
    '''
    Progressive feature fusion
    '''
    def __init__(self, C, layers):
        super(PFFM, self).__init__()
        self.layers = layers
        self.process_conv = nn.Conv2d(C, C, 1)
        self.convs = nn.ModuleList()
        for i in range(layers-1):
            self.convs.append(nn.Conv2d(2*C, C, 1))
        self.fusion = nn.Conv2d(layers*C, C, 1)

    def forward(self, feats):
        # print(len(feats))
        f0 = self.process_conv(feats[-1])
        # print('f0:', f0.shape)
        if self.layers == 1:
            return f0
        if self.layers == 2:
            f1 = self.convs[0](torch.cat([f0, feats[-2]], 1))
            # print('f1:', f1.shape)
            return self.fusion(torch.cat([f0, f1], 1))
        if self.layers == 3:
            f1 = self.convs[0](torch.cat([f0, feats[-2]], 1))
            f2 = self.convs[1](torch.cat([f1, feats[-3]], 1))
            return self.fusion(torch.cat([f0, f1, f2], 1))
        if self.layers == 4:
            f1 = self.convs[0](torch.cat([f0, feats[-2]], 1))
            f2 = self.convs[1](torch.cat([f1, feats[-3]], 1))
            f3 = self.convs[1](torch.cat([f2, feats[-4]], 1))
            return self.fusion(torch.cat([f0, f1, f2, f3], 1))
class PFFM_cell(nn.Module):
    '''
    Progressive feature fusion
    '''
    def __init__(self, C, layers):
        super(PFFM_cell, self).__init__()
        self.layers = layers
        self.process_conv = nn.Conv2d(C, C, 1)
        self.convs = nn.ModuleList()
        for i in range(layers-1):
            self.convs.append(nn.Sequential(nn.Conv2d(2*C, C, 1), nn.ReLU(True)))
        self.fusion = nn.Conv2d(layers*C, layers*C, 1)

    def forward(self, feats):
        # print(len(feats))
        f0 = self.process_conv(feats[-1])
        # print('f0:', f0.shape)
        if self.layers == 1:
            return f0
        if self.layers == 2:
            f1 = self.convs[0](torch.cat([f0, feats[-2]], 1))
            # print('f1:', f1.shape)
            return self.fusion(torch.cat([f0, f1], 1))
        if self.layers == 3:
            f1 = self.convs[0](torch.cat([f0, feats[-2]], 1))
            f2 = self.convs[1](torch.cat([f1, feats[-3]], 1))
            return self.fusion(torch.cat([f0, f1, f2], 1))
        if self.layers == 4:
            f1 = self.convs[0](torch.cat([f0, feats[-2]], 1))
            f2 = self.convs[1](torch.cat([f1, feats[-3]], 1))
            f3 = self.convs[1](torch.cat([f2, feats[-4]], 1))
            return self.fusion(torch.cat([f0, f1, f2, f3], 1))
class MS3Conv(nn.Module):

    def __init__(self, n_f):
        super(MS3Conv, self).__init__()
        self.n_f = n_f
        self.share_conv = nn.Conv2d(n_f, n_f//2, kernel_size=3)
        self.H2L_conv = nn.Conv2d(n_f, n_f//2)
        self.L2H_conv = nn.Conv2d(n_f, n_f//2)
        self.pa_H = PA(n_f//2)
        self.pa_L = PA(n_f//2)

        self.H_conv = nn.Conv2d(n_f//2, n_f//2, kernel_size=3)
        self.L_conv = nn.Conv2d(n_f//2, n_f//2, kernel_size=3)



    def forward(self, H_feats, L_feats):
        H_feats_H = self.share_conv(H_feats)
        L_feats_L = self.share_conv(L_feats)

        H_feats = self.L2H_conv(H_feats) + H_feats_H
        L_feats = self.H2L_conv(H_feats) + L_feats_L

        H_feats = self.pa_H(H_feats)
        L_feats = self.pa_L(L_feats)

        H_feats = self.H_conv(H_feats)
        L_feats = self.L_conv(L_feats)

        return H_feats
class CSSA(nn.Module):
    def __init__(self, C):
        super(CSSA, self).__init__()
        inner_C = C // 2
        self.process = nn.Conv2d(C, inner_C, 1)
        self.pa_1 = PAConv(inner_C)
        self.pa_2 = PAConv(inner_C)

        self.ca_1_1 = ChannelWiseAttention(inner_C, r=16)
        self.ca_1_2 = ChannelWiseAttention(inner_C, r=16)
        self.ca_2_1 = ChannelWiseAttention(inner_C, r=16)
        self.ca_2_2 = ChannelWiseAttention(inner_C, r=16)

    def forward(self, x):
        x = self.process(x)
        H_x = self.pa_1(x)
        P_1_x = self.ca_1_2(H_x) + x  # 上面
        Q_1_x = self.ca_1_1(x) + H_x  # 下面

        G_x = self.pa_2(P_1_x)
        P_2_x = self.ca_2_1(Q_1_x) + G_x
        Q_2_x = self.ca_2_2(G_x) + Q_1_x

        out = P_2_x + Q_2_x
        return out
class UPM(nn.Sequential):
    def __init__(self, n_feats, scale):
        n_colors = 3
        m = []
        if scale == 2 or scale == 3:
            m.append(nn.Conv2d(n_feats, scale**2*n_feats, 3, padding=1))
            m.append(nn.PixelShuffle(scale))
        if scale == 4:
            m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1))
            m.append(nn.PixelShuffle(2))
        m.append(nn.Conv2d(n_feats, n_colors, 3, padding=1))
        super(UPM, self).__init__(*m)
def main():
    from utility import print_network
    from torchstat import stat
    C = 64
    layers = 4
    scale = 4
    # n_feats = 32
    x = torch.randn(3, C, 123, 59)
    # up = Upsampler(scale, n_feats, act=None)
    # print(up)
    # out = up(x)
    # print(out.shape)

    # bfm = BFM(C, layers=3)
    # print(bfm)
    # print(print_network(bfm))
    # feats = []
    # for i in range(layers):
    #     feats.append(torch.randn(3, 64, 48, 48))
    #
    # out = bfm(feats)
    # print('bfm out:', out.shape)

    # cpa = Cross_PA_module(C)
    # print(cpa)
    # print(print_network(cpa))
    # up_out = cpa(x)
    # print(up_out.shape)
    # feats = []
    # for i in range(3):
    #     feats.append(torch.randn(3, 64, 48, 48))
    # pffm = PFFM(C=64, layers=3)
    # print(pffm)  # 45K layers=4 33K layers=3
    # print(print_network(pffm))
    # out = pffm(feats)
    # print(out.shape)

    # x = torch.randn(4, 64, 32, 32)
    # cpam = CPAM(C=64)
    # print(cpam)
    # print('{}K'.format(print_network(cpam)))  # 42K
    # cpam_out = cpam(x)
    # print(cpam_out.shape)

    y = torch.randn(4, 32, 32, 32)
    renet = UPM(n_feats=32, scale=4)
    print(renet)
    print(print_network(renet))  # 37K 84K 74K
    print(renet(y).shape)
    stat(renet, (32, 32, 32))


if __name__ == '__main__':
    main()
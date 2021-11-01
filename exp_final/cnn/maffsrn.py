import torch
import torch.nn as nn
from basic_ops import Conv1x1, DepthWiseConv, Channel_shuffle, Upsampler
import torch.nn.functional as F
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
class MAB(nn.Module):
    def __init__(self, channel):
        super(MAB, self).__init__()
        reduction = 4
        channel_inter = channel // reduction
        kernel_size = 3
        stride = 2  # strided conv
        dilation = 2
        max_pool_size = 2
        self.scale_factor = stride * max_pool_size
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.Conv2d(channel, channel_inter, kernel_size=1)
        )
        self.sconv = nn.Conv2d(channel_inter, channel_inter, kernel_size=kernel_size, stride=stride)
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size)
        self.dconv_1 = nn.Conv2d(channel_inter, channel_inter, kernel_size=kernel_size, dilation=dilation, padding=dilation)
        self.dconv_2 = nn.Conv2d(channel_inter, channel_inter, kernel_size=kernel_size, dilation=dilation, padding=dilation)

        self.conv1x1 = Conv1x1(channel_inter, channel)
        self.sigmoid = nn.Sigmoid()

        self.cea = nn.Sequential(
            Conv1x1(channel, channel_inter),
            DepthWiseConv(channel_inter, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        cea_in = long_x = x
        H, W = x.size(-2), x.size(-1)
        # print('long_x', long_x.shape)
        conv_x = self.conv(x)
        # print('conv_x:', conv_x.shape)
        sconv_x = self.sconv(conv_x)  # spatial resolution: 48=>16
        # print('sconv_x:', sconv_x.shape)
        reduction_x = self.max_pool(sconv_x)  # spatial resolution: 16=>8
        # print('reduction_x:', reduction_x.shape)
        dconv_1_x = self.dconv_1(reduction_x)
        dconv_2_x = self.dconv_2(reduction_x)
        dconv_x = dconv_1_x + dconv_2_x
        # print('dconv_x:', dconv_x.shape)
        upsample_x = F.interpolate(dconv_x, size=(H, W))
        # print('upsample_x:', upsample_x.shape)

        conv1x1_x = self.conv1x1(upsample_x)
        # print('conv1x1_x:', conv1x1_x.shape)
        att_map = self.sigmoid(conv1x1_x)
        out_map = att_map * long_x

        cea_out = self.cea(cea_in)

        return cea_out*out_map
class FFG_concat(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(FFG_concat, self).__init__()
        self.conv = Conv1x1(channel_in, channel_out)
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x_shuff = Channel_shuffle(x, groups=2)
        y = self.conv(x_shuff)
        return y
class FFG(nn.Module):
    def __init__(self, n_feats, n_MABs):
        super(FFG, self).__init__()
        # n_feats = 32
        concat_n_feats = n_feats * 2
        self.n_MABs = n_MABs
        mabs_list = nn.ModuleList()

        for i in range(self.n_MABs):
            mabs_list.append(MAB(n_feats))

        ffg_list = nn.ModuleList()
        for i in range(self.n_MABs-1):
            ffg_list.append(FFG_concat(concat_n_feats, n_feats))

        self.mab_body = nn.Sequential(*mabs_list)
        self.ffg_body = nn.Sequential(*ffg_list)

        self.rescale = nn.ModuleList([Scale(1) for i in range(2)])

    def forward(self, x):
        long_x = x
        mab_out = []
        for i in range(self.n_MABs):
            x = self.mab_body[i](x)
            if i >= 2:
                x = self.mab_body[i](mab_out[-1] + mab_out[-2])
            mab_out.append(x)

        # print('mab_out[0]:', mab_out[0].shape)
        ffg_out = []
        f1 = self.ffg_body[0](mab_out[0], mab_out[1])
        ffg_out.append(f1)
        # print('f1:', f1.shape)

        for i in range(self.n_MABs-2):
            tmp = self.ffg_body[i](ffg_out[i], mab_out[i+2])
            ffg_out.append(tmp)
        # print(len(ffg_out))
        scale_1_x = self.rescale[0](ffg_out[-1])
        scale_2_x = self.rescale[1](long_x)
        out = scale_1_x + scale_2_x
        return out
class MAFFSRN(nn.Module):
    def __init__(self, scale, n_FFGs, n_MABs, n_branches=2):
        super(MAFFSRN, self).__init__()

        self.n_FFGs = n_FFGs
        self.n_MABs = n_MABs
        self.n_branches = n_branches
        n_colors = 3
        n_feats = 32
        kernel_size = 3
        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size//2)

        # feature propagation
        body = nn.ModuleList()
        for i in range(self.n_FFGs):
            body.append(FFG(n_feats, n_MABs))

        self.body = nn.Sequential(*body)

        self.branches = nn.ModuleList([])
        for i in range(self.n_branches):
            self.branches.append(nn.Sequential(
                nn.Conv2d(n_feats, n_colors * scale ** 2, kernel_size=kernel_size+i*2, padding=(kernel_size+i*2)//2),
                Scale(1),
                nn.PixelShuffle(scale)
            ))
        # print(self.branches)
        self.upsample = Upsampler(scale, n_colors, wn, act)

    def forward(self, x):
        long_x = x = self.sub_mean(x)
        x = self.head(x)

        for i in range(self.n_FFGs):
            x = self.body[i](x)

        state = []
        for i in range(self.n_branches):
            # tmp = self.branches[i](x)
            state.append(self.branches[i](x))
            # print('tmp:', tmp.shape)  # 2, 32, 96, 96
        batch_size = state[0].shape[0]
        n_feats = state[0].shape[1]

        state = torch.cat(state, dim=1)
        state = state.view(batch_size, self.n_branches, n_feats, state.shape[2], state.shape[3])
        state = torch.sum(state, dim=1)
        # print('state:', state.shape)
        long_x = self.upsample(long_x)
        # print('long_x:', long_x.shape)

        out = self.add_mean(state + long_x)
        return out

def main():
    from torchstat import stat
    C = 32
    n_colors = 3
    x = torch.randn(2, C, 293, 196)
    mab = MAB(channel=C)
    out = mab(x)
    # ffg = FFG(n_feats=32, n_MABs=5)
    # out = ffg(x)
    # net = MAFFSRN(scale=2, n_FFGs=4, n_MABs=4, n_branches=2)
    # print(net)
    # out = net(x)
    print(out.shape)
    # stat(net, (3, 48, 48))

if __name__ == '__main__':
    main()
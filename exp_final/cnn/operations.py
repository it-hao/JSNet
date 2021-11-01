from cnn.basic_ops import default_conv, Conv1x1, act
from torch.nn import Conv2d
import torch
import torch.nn as nn
from option import args

OPS = {
    'none': lambda C: Zero(),
    'skip_connect': lambda C: Identity(),
    'sep_conv_3x3': lambda C: SepConv(C, kernel_size=3),  # 4929 11.2MFlops
    # 'sep_conv_5x5': lambda C: SepConv(C, kernel_size=5),  # 4929 11.2MFlops
    'dil_conv_3x3': lambda C: DilConv(C, kernel_size=3),  # 5653 12.6M
    # 'dil_conv_5x5': lambda C: DilConv(C, kernel_size=5),  # 5653 12.6M
    'rb': lambda C: RB(C),  # C=16 12304 28M
    'lrb': lambda C: LightRB(C),  # C=16 6160 23.9M
    'srb': lambda C: ShallowRB(C),  # C=16 7696 18M
    'acb': lambda C: ACB(C)
}
# 空洞卷积
class DilConv(nn.Module):
    def __init__(self, C, kernel_size):  # dilation:kernel间距
        super(DilConv, self).__init__()
        dilation1 = 2
        dilation2 = 3

        bias = args.bias
        self.op1 = nn.Sequential(
            Conv2d(C, C, kernel_size=kernel_size, padding=dilation1*(kernel_size-1)//2, dilation=dilation1, groups=C, bias=bias),
            Conv2d(C, C, kernel_size=kernel_size, padding=dilation2*(kernel_size-1)//2, dilation=dilation2, groups=C, bias=bias),
            Conv1x1(C, C, bias),
            act(args.activation),
        )
        self.op2 = nn.Sequential(
            Conv2d(2*C, C, kernel_size=kernel_size, padding=dilation1 * (kernel_size - 1) // 2, dilation=dilation1,
                   groups=C, bias=bias),
            Conv2d(C, C, kernel_size=kernel_size, padding=dilation2 * (kernel_size - 1) // 2, dilation=dilation2,
                   groups=C, bias=bias),
            Conv1x1(C, C, bias),
            act(args.activation),
        )
        self.reduce = Conv1x1(C*3, C)

    def forward(self, x):
        res = x
        x1 = self.op1(x)
        x2 = self.op2(torch.cat([x, x1], 1))
        x = self.reduce(torch.cat([x, x1, x2], 1))
        return x + res
# 深度可分离卷积
class SepConv(nn.Module):
    def __init__(self, C, kernel_size):
        super(SepConv, self).__init__()
        bias = args.bias

        self.op1 = nn.Sequential(
            Conv2d(C, C, kernel_size=kernel_size, padding=kernel_size//2, groups=C, bias=bias),
            Conv1x1(C, C, bias),
            act(args.activation),
        )
        self.op2 = nn.Sequential(
            Conv2d(2*C, C, kernel_size=kernel_size, padding=kernel_size // 2, groups=C, bias=bias),
            Conv1x1(C, C, bias),
            act(args.activation),
        )
        self.reduce = Conv1x1(C*3, C)

    def forward(self, x):
        res = x
        x1 = self.op1(x)
        x2 = self.op2(torch.cat([x, x1], 1))
        x = self.reduce(torch.cat([x, x1, x2], 1))
        return x + res

# Residual Block
class RB(nn.Module):
    def __init__(self, C):
        super(RB, self).__init__()
        bias = args.bias
        self.op1 = nn.Sequential(
            default_conv(C, C, kernel_size=3, bias=bias),
            act(args.activation),
            default_conv(C, C, kernel_size=3, bias=bias)
        )
        self.op2 = nn.Sequential(
            default_conv(2*C, C, kernel_size=3, bias=bias),
            act(args.activation),
            default_conv(C, C, kernel_size=3, bias=bias)
        )
        self.reduce = Conv1x1(C*3, C)

    def forward(self, x):
        res = x
        x1 = self.op1(x)
        x2 = self.op2(torch.cat([x, x1], 1))
        x = self.reduce(torch.cat([x, x1, x2], 1))
        return x + res

# Light Residual Block


class LightRB(nn.Module):
    def __init__(self, C):
        super(LightRB, self).__init__()
        bias = args.bias
        self.op1 = nn.Sequential(
            # act(args.activation),
            Conv1x1(C, C, bias),
            act(args.activation),
            default_conv(C, C, kernel_size=3, bias=bias)
        )
        self.op2 = nn.Sequential(
            # act(args.activation),
            Conv1x1(2*C, C, bias),
            act(args.activation),
            default_conv(C, C, kernel_size=3, bias=bias)
        )
        self.reduce = Conv1x1(C*3, C)

    def forward(self, x):
        res = x
        x1 = self.op1(x)
        x2 = self.op2(torch.cat([x, x1], 1))
        x = self.reduce(torch.cat([x, x1, x2], 1))
        return x + res

class ShallowRB(nn.Module):
    def __init__(self, C):
        super(ShallowRB, self).__init__()
        bias = args.bias
        self.op1 = nn.Sequential(
            default_conv(C, C, kernel_size=3, bias=bias),
            act(args.activation)
        )
        self.op2 = nn.Sequential(
            default_conv(2*C, C, kernel_size=3, bias=bias),
            act(args.activation)
        )
        self.reduce = Conv1x1(C*3, C)

    def forward(self, x):
        res = x
        x1 = self.op1(x)
        x2 = self.op2(torch.cat([x, x1], 1))
        x = self.reduce(torch.cat([x, x1, x2], 1))
        return x + res

class ACB(nn.Module):
    def __init__(self, C):
        super(ACB, self).__init__()
        bias = args.bias
        self.op = nn.Sequential(
            Conv2d(C, C, (1, 3), stride=(1, 1), padding=(0, 1), bias=bias),
            Conv2d(C, C, (3, 1), stride=(1, 1), padding=(1, 0), bias=bias),
            act(args.activation)
        )

    def forward(self, x):
        res = self.op(x)
        res += x
        return res

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
    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()

        return num_params
    from torchstat import stat
    B = 3
    C = 16
    H = 48
    W = 48

    x = torch.randn(B, C, H, W)
    y = torch.randn(3, C, H, W)

    op = OPS['rb'](C)
    print(op)
    print('param computed by print_nerwork:{}'.format(print_network(op)))
    out = op(x)
    print('out:', out.shape)
    stat(op, (C, 48, 48))


if __name__ == '__main__':
    main()

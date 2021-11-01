import sys
sys.path.append('F:\lwr\exp_final')
sys.path.append('F:\lwr\exp_final\cnn')
from cnn.operations import OPS
from cnn.attentions import ATS
from cnn.basic_ops import default_conv, Conv1x1, MeanShift
import torch
import torch.nn as nn
from cnn.reconstruction import CSSA, UPM


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params

class ResidualBlock(nn.Module):
    def __init__(self, C):
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        bias = False
        self.convs = nn.Sequential(
            default_conv(C, C, kernel_size, bias),
            nn.ReLU(True),
            default_conv(C, C, kernel_size, bias)
        )
    def forward(self, x):
        return x + self.convs(x)

class NASCell(nn.Module):
    def __init__(self, genotype, C):
        super(NASCell, self).__init__()
        op_names, op_indices = zip(*genotype.normal)
        at_names, at_indices = zip(*genotype.attention)
        self._compile(C, op_names, op_indices, at_names, at_indices)

    def _compile(self, C, op_names, op_indices, at_names, at_indices):
        assert len(op_names) == len(op_indices) and len(at_names) == len(at_indices)
        # print(op_names, op_indices)
        # print(at_names, at_indices)

        self.op_nodes = int(((1 + 8 * len(op_names))**0.5 - 1) / 2)  # 3
        self.at_nodes = int(((1 + 8 * len(at_names))**0.5 - 1) / 2)  # 3
        C_concat = C * (self.op_nodes + 1)

        self.process = Conv1x1(C_concat, C)
        self.ops = nn.ModuleList()
        self.atts = nn.ModuleList()

        for name, index in zip(op_names, op_indices):
            op = OPS[name](C)
            self.ops.append(op)
        for name, index in zip(at_names, at_indices):
            at = ATS[name](C_concat)
            self.atts.append(at)
        # print(self.ops)
        # print(self.atts)

        self.op_nodes_fusion = Conv1x1(C_concat, C_concat)

    def forward(self, c_prev):
        x = self.process(c_prev)
        states = [x]
        offset = 0
        for i in range(self.op_nodes):
            s = sum(self.ops[offset+j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        f0 = self.op_nodes_fusion(torch.cat(states, dim=1))

        feats = [f0]
        offset = 0
        for i in range(self.at_nodes):
            f = sum(self.atts[offset + j](f) for j, f in enumerate(feats))
            offset += len(feats)
            feats.append(f)

        return c_prev + feats[-1]


class NASnetWork(nn.Module):
    def __init__(self, args, genotype):
        super(NASnetWork, self).__init__()
        kernel_size = 3  # for head and tail
        self.args = args
        self.C = args.C
        self.op_nodes = args.op_nodes
        self.at_nodes = args.at_nodes

        self.layers = args.layers
        cell_concat = self.C * (self.op_nodes + 1)  # 64
        layer_concat = self.layers * cell_concat  # 192

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        self.head = default_conv(3, cell_concat, kernel_size)

        self.cells = nn.ModuleList()
        for i in range(self.layers):
            cell = NASCell(genotype, self.C)
            self.cells.append(cell)

        self.fusion = Conv1x1(layer_concat, cell_concat)
        self.cpam = CSSA(C=cell_concat)
        self.up = UPM(n_feats=cell_concat // 2, scale=args.scale[0])

    def forward(self, x):
        x = self.sub_mean(x)
        s = res = self.head(x)

        cell_out = []
        for i, cell in enumerate(self.cells):
            s = cell(s)
            cell_out.append(s)

        x = self.fusion(torch.cat(cell_out, dim=1))

        x += res
    
        # print('after fusion: {}'.format(x.shape))
        x = self.cpam(x)

        x = self.up(x)
        x = self.add_mean(x)

        return x


def main():
    import cnn.genotypes as genotypes
    from option import args
    from torchstat import stat
    genotype = eval("genotypes.%s" % args.arch)
    print('{}: {}'.format(args.arch, genotype))
    C = 64
    x = torch.randn(3, 3, 48, 48)
    s0 = torch.randn(3, C, 24, 24)
    s1 = torch.randn(3, C, 24, 24)

    # PC_DARTS_SR_x2_end =
    # Genotype(
    # normal=[('lrb', 0), ('dil_conv_3x3', 1), ('rb', 0), ('lrb', 1), ('rb', 0), ('acb', 1), ('rb', 4), ('rb', 1)],
    # attention=[('esab', 0), ('cea', 0), ('cea', 1), ('esab', 0), ('cea', 1), ('cea', 2)])
    # cell = NASCell(genotype, args.C)
    # # print(cell)
    # print('cell:{}'.format(print_network(cell)))  # 78568
    # cell_out = cell(s0)
    # # stat(cell, (64, 24, 24))
    # print('out_cell.shape:', cell_out.shape)

    net = NASnetWork(args, genotype)
    # print(net)
    print('model param: {}'.format(print_network(net)))
    print('C:{}\tlayers:{}\tscale:{}\top_nodes:{}\tat_nodes:{}'.format(args.C, args.layers, args.scale, args.op_nodes, args.at_nodes))
    # stat(net, (3, 48, 48))

    out = net(x)
    print('out.shape:\t', out.shape)


if __name__ == '__main__':
    main()

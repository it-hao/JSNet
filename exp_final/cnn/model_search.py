from cnn.operations import OPS
from cnn.attentions import ATS
from cnn.basic_ops import default_conv, Conv1x1, MeanShift, Channel_shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn.genotypes import OPS_PRIMITIVES, ATS_PRIMITIVES
import numpy as np
from utility import print_network
from cnn.genotypes import Genotype
from cnn.reconstruction import PFFM, CSSA, UPM
class MixOp(nn.Module):
    def __init__(self, C):
        super(MixOp, self).__init__()
        self.ops = nn.ModuleList()
        self.k = 4
        for ops_primitive in OPS_PRIMITIVES:
            op = OPS[ops_primitive](C//self.k)
            self.ops.append(op)

    def forward(self, x, weights_op):
        dim_2 = x.shape[1]
        xtemp = x[:, :dim_2//self.k, :, :]
        xtemp2 = x[:, dim_2//self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights_op, self.ops))
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = Channel_shuffle(ans, self.k)
        return ans


class MixAt(nn.Module):
    def __init__(self, C):
        super(MixAt, self).__init__()
        self.ats = nn.ModuleList()
        for ats_primitive in ATS_PRIMITIVES:
            at = ATS[ats_primitive](C)
            self.ats.append(at)

    def forward(self, x, weights_at):
        return sum(w * at(x) for w, at in zip(weights_at, self.ats))


class NASCell(nn.Module):
    def __init__(self, C=32, op_nodes=3, at_nodes=3):
        super(NASCell, self).__init__()
        self.op_nodes = op_nodes
        self.at_nodes = at_nodes
        self.ops = nn.ModuleList()
        self.atts = nn.ModuleList()
        C_concat = C * (op_nodes + 1)

        self.process = Conv1x1(C_concat, C)

        for i in range(self.op_nodes):
            for j in range(i+1):
                op = MixOp(C)
                self.ops.append(op)
        for i in range(self.at_nodes):
            for j in range(i+1):
                at = MixAt(C_concat)
                self.atts.append(at)
        self.op_nodes_fusion = Conv1x1(C_concat, C_concat)

    def forward(self, c_prev, weights_op, weights_at, weights_eg):
        x = self.process(c_prev)
        states = [x]
        offset = 0
        for i in range(self.op_nodes):
            s = sum(weights_eg[offset+j]*self.ops[offset + j](h, weights_op[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        f0 = self.op_nodes_fusion(torch.cat(states, dim=1))
        offset = 0
        feats = [f0]
        for i in range(self.at_nodes):
            f = sum(self.atts[offset + j](f, weights_at[offset + j]) for j, f in enumerate(feats))
            offset += len(feats)
            feats.append(f)

        return c_prev + feats[-1]

class NASnetWork(nn.Module):
    def __init__(self, args):
        super(NASnetWork, self).__init__()
        kernel_size = 3  # for head and tail
        self.args = args
        self.C = args.C
        self.op_nodes = args.op_nodes
        self.at_nodes = args.at_nodes

        self.layers = args.layers
        cell_concat = self.C*(self.op_nodes+1)  # cell内部node concat
        layers_concat = cell_concat*self.layers
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        self.head = default_conv(3, cell_concat, kernel_size)

        self.cells = nn.ModuleList()
        for i in range(self.layers):
            cell = NASCell(self.C, self.op_nodes, self.at_nodes)
            self.cells.append(cell)
        # self.confusion = Conv1x1(layers_concat, cell_concat)
        self.pffm = PFFM(C=cell_concat, layers=self.layers)
        self.cpam = CSSA(C=cell_concat)
        self.res_reduction = Conv1x1(cell_concat, cell_concat//2)
        self.up = UPM(n_feats=cell_concat//2, scale=args.scale[0])

        self._initialize_alphas()

    def forward(self, x):
        x = self.sub_mean(x)
        s = res = self.head(x)

        cell_out = []
        for i, cell in enumerate(self.cells):
            weights_op = F.softmax(self.alphas_normal, dim=-1)
            weights_at = F.softmax(self.alphas_attention, dim=-1)

            n = 2
            start = 1
            weights2 = F.softmax(self.betas_normal[0:1], dim=-1)
            for i in range(self.op_nodes-1):
                end = start + n
                tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)
            s = cell(s, weights_op, weights_at, weights2)
            cell_out.append(s)

        # for i in range(len(cell_out)):
            # print('cell out[{}] shape:{}'.format(i, cell_out[i].shape))
        # x = self.confusion(torch.cat(cell_out, dim=1))
        x = self.pffm(cell_out)
        # print('after fusion:', x.shape)
        x = self.cpam(x)
        # print('after cpam:', x.shape)

        # print('res:', res.shape)
        res = self.res_reduction(res)
        # print('after res reduction:', res.shape)

        x += res
        x = self.up(x)
        # print(x.shape)
        x = self.add_mean(x)

        return x

    def new(self):
        model_new = NASnetWork(self.args).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_alphas(self):
        num_op_connection = sum(1 for i in range(self.op_nodes) for n in range(i+1))
        # print('num_op_connection:', num_op_connection)
        num_ops = len(OPS_PRIMITIVES)
        # print('num_ops:', num_ops)
        num_at_connection = sum(1 for i in range(self.at_nodes) for n in range(i+1))
        # print('num_at_connection:', num_at_connection)
        num_ats = len(ATS_PRIMITIVES)
        # print('num_ats:', num_ats)
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(num_op_connection, num_ops))
        self.alphas_attention = nn.Parameter(1e-3 * torch.randn(num_at_connection, num_ats))
        self.betas_normal = nn.Parameter(1e-3 * torch.randn(num_op_connection))

        self._arch_parameters = [self.alphas_normal, self.alphas_attention, self.betas_normal]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse_op(weights, weights2):
            gene = []
            n = 1
            start = 0
            for i in range(self.op_nodes):
                end = start + n
                # print('start: {} end: {} n: {}'.format(start, end, n))
                W = weights[start:end].copy()
                # print('W:', W)

                W2 = weights2[start:end].copy()
                # print('W2:', W2)

                for j in range(n):
                    W[j, :] = W[j, :]*W2[j]
                # print('W:', W)

                for j, w in enumerate(W):
                    index = np.argmax(w[1:])
                    # print('index:', index+1)
                    gene.append((OPS_PRIMITIVES[index + 1], j))
                start = end
                n += 1

            return gene

        def _parse_at(weights_at):
            gene = []
            n = 1
            start = 0
            for i in range(self.at_nodes):
                end = start + n
                W = weights_at[start:end].copy()

                for j, w in enumerate(W):
                    # print('前驱结点j: ', j)
                    index = np.argmax(w[1:])
                    # print('index:', index)
                    gene.append((ATS_PRIMITIVES[index + 1], j))  # 把(操作，前驱节点序号)放到list gene中，[('sep_conv_3x3', 1),...,]
                start = end
                n += 1

            return gene

        n = 2
        start = 1
        weights_eg = F.softmax(self.betas_normal[0:1], dim=-1)

        for i in range(self.op_nodes-1):
            end = start + n
            # print(start, end)
            tn = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weights_eg = torch.cat([weights_eg, tn], dim=0)
        # print(weights_eg, weights_eg.shape)

        gene_normal = _parse_op(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weights_eg.data.cpu().numpy())
        gene_attention = _parse_at(F.softmax(self.alphas_attention, dim=-1).data.cpu().numpy())

        genotype = Genotype(normal=gene_normal, attention=gene_attention)

        return genotype
def main():
    from option import args
    from torchstat import stat
    B = 4
    C = args.C
    H = 32
    W = 32
    img = torch.randn(B, args.n_colors, H, W)
    # x = torch.randn(B, C, H, W)
    # weights_op = torch.randn(6)
    # ops = MixOp(C)
    # print(ops)
    # print('ops: {}K'.format(print_network(ops)))
    # # stat(ops, (32, 48, 48))
    # out = ops(x, weights_op)
    # print('ops out shape: ', out.shape)

    # x = torch.randn(B, C, H, W)
    # weights_at = torch.randn(9)
    # ats = MixAt(C)
    # print(ats)
    # out = ats(x, weights_at)
    # print(out.shape)

    c_prev = torch.randn(B, C*4, H, W)
    num_op_connection = sum(1 for i in range(args.op_nodes) for n in range(i + 1))
    print('num_op_connection:', num_op_connection)
    num_ops = len(OPS_PRIMITIVES)
    print('num_ops:', num_ops)
    num_at_connection = sum(1 for i in range(args.at_nodes) for n in range(i + 1))
    print('num_at_connection:', num_at_connection)
    num_ats = len(ATS_PRIMITIVES)
    print('num_ats:', num_ats)
    weights_op = torch.randn(num_op_connection, num_ops)
    weights_at = torch.randn(num_at_connection, num_ats)
    weights_eg = torch.randn(num_op_connection)
    print()
    cell = NASCell(C=args.C, op_nodes=args.op_nodes, at_nodes=args.at_nodes)
    # print(cell)
    print('cell: {:.2f}K'.format(print_network(cell)))  # 70K

    out = cell(c_prev, weights_op, weights_at, weights_eg)
    print('cell out shape: ', out.shape)
    print()

    net = NASnetWork(args)
    print('C:{}\tlayers:{}\top_nodes:{}\tat_nodes:{}'.format(args.C, args.layers, args.op_nodes, args.at_nodes))
    # print(net)
    print('model size: {:.2f}K'.format(print_network(net)))
    out = net(img)
    # stat(net, (3, 32, 32))
    # print(net.genotype())
    print('net out shape: ', out.shape)


if __name__ == '__main__':
    main()

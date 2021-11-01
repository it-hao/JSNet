import os
import math
import time
import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if not args.create_results:
            if args.load == '.':
                if args.save == '.':
                    args.save = now
                self.dir = '../experiment/' + args.save

            else:
                self.dir = '../experiment/' + args.load
                if not os.path.exists(self.dir):
                    args.load = '.'
                else:
                    self.log = torch.load(self.dir + '/psnr_log.pt')
                #     print('Continue from epoch {}...'.format(len(self.log)))

            def _make_dir(path):
                if not os.path.exists(path):
                    os.makedirs(path)

            _make_dir(self.dir)
            _make_dir(self.dir + '/model')
            _make_dir(self.dir + '/results')

            open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
            # print(open_type)
            self.log_file = open(self.dir + '/log.txt', open_type)
            self.arch_log_file = open(self.dir + '/arch_log.txt', open_type)
            with open(self.dir + '/config.txt', open_type) as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            self.dir = args.test_dir

            def _make_dir(path):
                if not os.path.exists(path):
                    os.makedirs(path)
            _make_dir(args.test_dir)
            _make_dir(args.test_dir + '/' + args.testset)
            _make_dir(args.test_dir + '/' + args.testset + '/' + 'X' + str(args.scale[0]))
            open_type = 'a' if os.path.exists(self.dir + '/psnr_log.txt') else 'w'
            self.psnr_log_file = open(self.args.test_dir + '/' +self.args.testset + '/' + 'X' + str(self.args.scale[0]) + '/psnr.txt', open_type)

    def write_psnr(self, log, refresh=False):
        self.psnr_log_file.write(log + '\n')
        if refresh:
            self.psnr_log_file.close()
            self.psnr_log_file = open(self.args.test_dir + '/' + self.args.testset + '/' + 'X' + str(self.args.scale[0]) + '/psnr.txt', 'a')

    def save(self, trainer):
        # trainer.model.save(self.dir, epoch, is_best=is_best)
        # trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)
        #
        # self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])
        # print('add ckp log: {}'.format(self.log.size()))

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def write_arch_log(self, arch_log, refresh=False):
        self.arch_log_file.write(arch_log + '\n')
        if refresh:
            self.arch_log_file.close()
            self.arch_log_file = open(self.dir + '/arch_log.txt', 'a')

    def done(self):
        if not self.args.create_results:
            self.log_file.close()
            self.arch_log_file.close()
        else:
            self.psnr_log_file.close()

    def plot_psnr(self, step):
        axis = np.linspace(self.args.print_psnr, step, step//self.args.print_psnr)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            io.imsave('{}{}.png'.format(filename, p), ndarr)

    def create_results(self, filename, save_list, scale):
        filename = '{}/{}/X{}/{}_x{}_'.format(self.dir, self.args.testset, scale, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            io.imsave('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def second2standard(second):
    h = math.floor(second / 3600)
    m = math.floor(second / 60 - h * 60)
    s = math.floor(second - h * 3600 - m * 60)
    return h, m, s

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.mul_(mask)
    return x

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params

import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        self.args = args
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
           
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('Preparing loss function: {:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        # print('self.loss: ', self.loss)
        # [{'type': 'L1', 'weight': 0.999, 'function': L1Loss()},
        #  {'type': 'MSE', 'weight': 0.001, 'function': MSELoss()},
        #  {'type': 'Total', 'weight': 0, 'function': None}]
        # print('self.loss_module: ', self.loss_module)
        # ModuleList(
        #    (0): L1Loss()
        #    (1): MSELoss()
        # )
        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)

        if args.load != '.':
            self.load(ckp.dir, cpu=False)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                # print('effective_loss: ', effective_loss)
                losses.append(effective_loss)
                self.log[-1, i] = effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] = loss_sum.item()

        # print('self.log: {}'.format(self.log))
        # print('self.log[-1]: {}'.format(self.log[-1]))

        return loss_sum

    def add_log(self):
        # print('add log')
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))
        # print('start loss log: {}'.format(self.log.size()))

    def display_loss(self):
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c))
        return ''.join(log)

    def plot_loss(self, apath):
        iteration = self.log.size(0)
        # print(iteration)
        axis = np.linspace(1, iteration, iteration)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()

            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        print('Continue from [Step: {}]...'.format(self.log.size(0)))



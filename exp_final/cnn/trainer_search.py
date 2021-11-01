import utility
import torch
from cnn.architect import Architect
# from adamp import AdamP
import torch.nn as nn
import os
import torch.nn.functional as F
import time
import math

class Trainer_Search():
    def __init__(self, args, loader, model, loss, ckp):
        self.args = args
        self.scale = args.scale
        self.loader_train = loader.loader_train
        self.loader_eval = loader.loader_eval
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.ckp = ckp
        self.step = loss.log.size(0)

        arch_params = list(map(id, self.model.arch_parameters()))
        weight_params = filter(lambda p: id(p) not in arch_params,
                               self.model.parameters())
        # self.optimizer = AdamP(weight_params, args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        self.optimizer = torch.optim.Adam(weight_params, args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay, gamma=args.lr_decay_factor)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            self.model.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'model', 'model_latest.pt'))
            )
            for _ in range(self.step):
                self.scheduler.step()

        criterion = nn.L1Loss()
        criterion = criterion.cuda()

        self.architect = Architect(self.model, criterion, self.args)

    def train(self):
        args = self.args
        ckp = self.ckp
        while True:
            timer_loader = time.time()
            timer_data, timer_model = utility.timer(), utility.timer()
            for inputs in self.loader_train:
                self.scheduler.step()
                learning_rate = self.scheduler.get_lr()[0]

                lr, hr = self.prepare([inputs[0], inputs[1]])
                search = next(iter(self.loader_eval))
                lr_search, hr_search = self.prepare([search[0], search[1]])
                timer_data.hold()
                timer_model.tic()

                if self.step+1 > args.arch_step:
                    self.architect.step(lr, hr, lr_search, hr_search, learning_rate, self.optimizer, unrolled=True)

                self.optimizer.zero_grad()
                sr = self.model(lr)
                self.loss.add_log()  # add loss log
                loss = self.loss(sr, hr)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                self.optimizer.step()

                self.loss.save(ckp.dir)  # save loss.pt and loss_log.pt
                self.loss.plot_loss(ckp.dir)

                self.step += 1

                self.display_loss(args, ckp, learning_rate)

                self.display_psnr(args, ckp)

                self.save_models(args, ckp)

                self.print_genotype(args, ckp)

                if self.step >= args.max_steps: break  # 1000

                timer_data.tic()

            if self.step >= args.max_steps: break

            self.display_time(timer_loader, args, ckp)

    def test(self):
        args = self.args
        ckp = self.ckp
        self.model.eval()
        ckp.add_log(torch.zeros(1, len(self.scale)))
        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                for idx_img, inputs in enumerate(self.loader_test):
                    lr, hr = self.prepare([inputs[0], inputs[1]])
                    filename = inputs[2][0]
                    sr = self.model(lr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    if args.save_results and self.step > args.arch_step:
                        save_list = [sr]
                        save_list.extend([lr, hr])
                        self.ckp.save_results(filename, save_list, scale)
                axis = self.step / args.print_psnr
                if axis != ckp.log.size(0):
                    ckp.add_log(torch.zeros(1, len(self.args.scale)))
                    ckp.log[-2, idx_scale] = ckp.log[-3, idx_scale]

                ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                # print(ckp.log)
                best = ckp.log.max(0)
                best_step = (best[1][idx_scale]+1)*args.print_psnr
                self.ckp.write_log('[{} x{}]\t[Step: {}]\tPSNR: {:.3f} (Best: {:.3f} @Step {})  {:.2f}s'.format(
                        self.args.data_test, scale, self.step, self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best_step,
                        timer_test.toc()), refresh=True)
                if best_step == self.step:
                    torch.save(self.model.state_dict(), os.path.join(ckp.dir, 'model', 'model_best.pt'))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def decay_learning_rate(self):
        lr = self.args.lr * (0.5 ** (self.step // self.args.lr_decay))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def display_loss(self, args, ckp, lr):
        if self.step % args.print_loss == 0:
            ckp.write_log('lr: {:.2e}\t[Step: {}]\tloss: {}'.format(lr, self.step, self.loss.display_loss()))

    def display_psnr(self, args, ckp):
        if self.step % args.print_psnr == 0:
            self.test()
            ckp.save(self)  # save psnr_log.pt and optimizer.pt
            ckp.plot_psnr(self.step)

    def save_models(self, args, ckp):
        if self.step % args.save_models == 0:
            torch.save(self.model.state_dict(), os.path.join(ckp.dir, 'model', 'model_latest.pt'))

    def print_genotype(self, args, ckp):
        if self.step % args.print_genotype == 0 and self.step >= args.arch_step:  # 20
            genotype = self.model.genotype()
            ckp.write_arch_log('[Step: {}]\t{}'.format(self.step, genotype), refresh=True)
            # alphas_normal = F.softmax(self.model.alphas_normal, dim=-1)
            # ckp.write_arch_log('alphas_normal = {}'.format(alphas_normal), refresh=True)

    def display_time(self, timer_loader, args, ckp):
        rest_time = (time.time() - timer_loader) * (args.max_steps - self.step) / len(self.loader_train)
        # print('{:.2f}s'.format(rest_time))
        rest_hour = math.floor(rest_time / 3600)
        rest_min = math.floor(rest_time / 60 - rest_hour * 60)
        rest_second = math.floor(rest_time - rest_hour * 3600 - rest_min * 60)
        ckp.write_log('{}\t\tTo Finish: {:.0f}h {:.0f}min {:.0f}s'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            rest_hour,
            rest_min,
            rest_second))








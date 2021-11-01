import utility
import torch
import torch.nn as nn
# from adamp import AdamP
import os
import time

class Trainer():
    def __init__(self, args, loader, model, loss, ckp):
        self.args = args
        self.scale = args.scale
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.ckp = ckp
        self.step = loss.log.size(0)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          args.lr, betas=(args.beta1, args.beta2),
                                          eps=args.epsilon,
                                          weight_decay=args.weight_decay)
        # self.optimizer = AdamP(self.model.parameters(), args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay, gamma=args.lr_decay_factor)

        if self.args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            self.model.load_state_dict(torch.load(os.path.join(ckp.dir, 'model', 'model_latest.pt')) )
            for _ in range(self.step):
                self.scheduler.step()

    def train(self):
        args = self.args
        ckp = self.ckp
        while True:
            timer_loader = time.time()
            for inputs in self.loader_train:
                self.model.train()
                self.scheduler.step()
                learning_rate = self.scheduler.get_lr()[0]

                lr, hr = self.prepare([inputs[0], inputs[1]])

                self.optimizer.zero_grad()
                sr = self.model(lr)
                self.loss.add_log()  # add loss log
                loss = self.loss(sr, hr)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)

                self.optimizer.step()

                self.loss.save(ckp.dir)  # save loss.pt and loss_log.pt
                self.loss.plot_loss(ckp.dir)

                self.step += 1

                if self.step % args.print_loss == 0:
                    ckp.write_log('lr: {:.2e}\t[Step: {}]\tloss: {}'.format(learning_rate, self.step, self.loss.display_loss()))

                self.display_psnr(args, ckp)

                if self.step >= args.max_steps: break  # 1000

            if self.step >= args.max_steps: break

            torch.save(self.model.state_dict(), os.path.join(ckp.dir, 'model', 'model_latest.pt'))

            rest_time = (time.time() - timer_loader) * (args.max_steps - self.step) / len(self.loader_train)
            h, m, s = utility.second2standard(rest_time)
            ckp.write_log('{}\t\tTo Finish: {}h {}min {}s'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                h, m, s))

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
                    save_list = [sr]
                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    save_list.extend([lr, hr])
                    if args.save_results:
                        ckp.save_results(filename, save_list, scale)
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
        device = torch.device('cuda')

        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def display_psnr(self, args, ckp):
        if self.step % args.print_psnr == 0:
            self.test()
            ckp.save(self)  # save psnr_log.pt and optimizer.pt
            ckp.plot_psnr(self.step)


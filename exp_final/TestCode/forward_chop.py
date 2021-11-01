import torch
import sys
import os
from tqdm import tqdm
import time
sys.path.append('F:\lwr\exp_final')
sys.path.append('F:\lwr\exp_final\cnn')
import utility
from option import args
from cnn.model import NASnetWork
from myimage import MyImage
from torch.utils.data.dataloader import DataLoader
import cnn.genotypes as genotypes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def forward_chop(scale, model, x, shave=10, min_size=160000):
    n_GPUs = 2
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        # print('here')
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(scale=scale, model=model, x=patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def prepare(l, volatile=False):
    device = torch.device('cuda:0')
    def _prepare(tensor):
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

def test(model, test_loader, args, ckp):
    with torch.no_grad():
        eval_acc = 0
        tqdm_test = tqdm(test_loader, ncols=80)
        for idx_img, inputs in enumerate(tqdm_test):
            lr, hr = prepare([inputs[0], inputs[1]])
            filename = inputs[2][0]
            print('creating img file:{}'.format(filename))
            sr = forward_chop(args.scale[0], model, lr)

            sr = utility.quantize(sr, args.rgb_range)

            save_list = [sr]
            cur_acc = utility.calc_psnr(sr, hr, args.scale[0], args.rgb_range)
            eval_acc += cur_acc
            ckp.write_psnr('{}: {}'.format(filename, cur_acc), refresh=True)

            save_list.extend([lr, hr])
            if args.save_results:
                ckp.create_results(filename, save_list, args.scale[0])
        ckp.write_psnr('average: {}'.format(eval_acc/len(test_loader)), refresh=True)


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

myimage = MyImage(args)
test_loader = DataLoader(myimage)
print('loader length: ', len(test_loader))

genotype = eval("genotypes.%s" % args.arch)
normal = genotype.normal
print('{}:\nnormal: {}\nattention: {}'.format(args.arch, genotype.normal, genotype.attention))
print('layers: {}\t scale: X{}'.format(args.layers, args.scale[0]))
model = NASnetWork(args, genotype)
# print(model)

model = model.cuda()
model_path = './pre_train_model' + '/' + args.pre_train
model.load_state_dict(torch.load(model_path + '\\' + 'model_best.pt'), strict=False)
start = time.time()
test(model, test_loader, args, checkpoint)
total_time = time.time() - start
print('test time: {:.2f}s'.format(total_time))
checkpoint.done()
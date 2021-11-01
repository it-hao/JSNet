import torch
import utility
from option import args
from cnn.model import NASnetWork
from myimage import MyImage
from torch.utils.data.dataloader import DataLoader
import cnn.genotypes as genotypes


def prepare(l, volatile=False):
    device = torch.device('cpu' if args.cpu else 'cuda')

    def _prepare(tensor):
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

def test(model, test_loader, args, ckp):
    eval_acc = 0
    for idx_img, inputs in enumerate(test_loader):
        lr, hr = prepare([inputs[0], inputs[1]])
        filename = inputs[2][0]
        sr = model(lr)
        sr = utility.quantize(sr, args.rgb_range)

        save_list = [sr]
        # eval_acc += utility.calc_psnr(sr, hr, args.scale[0], args.rgb_range)
        save_list.extend([lr, hr])
        if args.save_results:
            ckp.create_results(filename, save_list, args.scale[0])


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

myimage = MyImage(args)
test_loader = DataLoader(myimage)
print('loader length: ', len(test_loader))

genotype = eval("genotypes.%s" % args.arch)
normal = genotype.normal
print('{}:\nnormal: {}\nattention: {}'.format(args.arch, genotype.normal, genotype.attention))
model = NASnetWork(args, genotype)
# print(model)

model = model.cuda()
model.load_state_dict(torch.load(args.pre_train), strict=False)

test(model, test_loader, args, checkpoint)


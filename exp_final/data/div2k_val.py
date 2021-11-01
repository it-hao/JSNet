import os
from data import srdata


class DIV2K_VAL(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K_VAL, self).__init__(args, train)
        self.repeat = args.repeat * 8

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        idx_begin = self.args.offset_val
        idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        # print(list_hr)

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):

        self.apath = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubic')

        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            'valid_bin_HR.npy'
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            'valid_bin_LR_X{}.npy'.format(scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx


if __name__ == '__main__':
    from option import args

    div2k_val = DIV2K_VAL(args)
    print(args)
    import matplotlib.pyplot as plt
    from skimage import io
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch

    device = torch.device('cuda')
    # epochs = 2
    results_dir = '../results/'

    unloader = transforms.ToPILImage()

    print(len(div2k_val))  # 32
    # batch_size = 4 n_train=16 repeat = 8/(16/4) = 2
    db = DataLoader(div2k_val, batch_size=20, shuffle=True)
    print(len(db))
    for inputs in db:
        print(inputs[0].shape, inputs[1].shape, inputs[2], inputs[3])


import os
import os.path
import sys
sys.path.append('D:\Codes\mycode_exp1')
sys.path.append('D:\Codes\mycode_exp1\cnn')
from skimage import io
import torch.utils.data as data
from data import common, srdata
class MyImage(srdata.SRData):
    def __init__(self, args, train=False):
        super(MyImage, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_test_hr):
            filename = os.path.splitext(entry.name)[0]
            # print(os.path.join(self.dir_test_hr, filename + self.ext))
            list_hr.append(os.path.join(self.dir_test_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                print(os.path.join(
                    self.dir_test_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
                list_lr[si].append(os.path.join(
                    self.dir_test_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()
        # print(list_hr)
        # print(list_lr)
        return list_hr, list_lr

    def _set_filesystem(self, dir_test_data):
        self.apath = os.path.join(dir_test_data, 'benchmarkbicubic', self.args.testset)
        self.dir_test_hr = os.path.join(self.apath, 'HR')
        self.dir_test_lr = os.path.join(self.apath, 'LR_bicubic')

        self.ext = '.png'

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
def main():
    from option import args
    myimage = MyImage(args)


if __name__ == '__main__':
    main()

from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate


class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = MSDataLoader(args, trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            if not args.train_only:
                module_val = import_module('data.' + args.data_val.lower())
                valset = getattr(module_val, args.data_val)(args)
                self.loader_eval = MSDataLoader(args, valset, batch_size=args.batch_size, shuffle=True, **kwargs)

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100', 'Test']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)
        else:
            module_test = import_module('data.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )

if __name__ == '__main__':
    from option import args
    data = Data(args)
    for idx, (lr, hr, filename, idx_scale) in enumerate(data.loader_test):
        print(filename[0], lr.shape, hr.shape, idx_scale)

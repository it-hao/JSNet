import torch
import sys
import os
import time
import utility
import data
from cnn import model_search
import loss
from option import args
from cnn.genotypes import OPS_PRIMITIVES, ATS_PRIMITIVES
from cnn.trainer_search import Trainer_Search

# sys.exit(0)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    checkpoint.write_log('pid: {}'.format(os.getpid()))
    checkpoint.write_log('OPS_PRIMITIVES:\t{}'.format(OPS_PRIMITIVES))
    checkpoint.write_log('ATS_PRIMITIVES:\t{}'.format(ATS_PRIMITIVES))

    loader = data.Data(args)
    print('loader length: [train:{}]\t[valid:{}]\t[test:{}]'.format(
        len(loader.loader_train),
        len(loader.loader_eval),
        len(loader.loader_test)))

    model = model_search.NASnetWork(args)
    model = model.cuda()

    checkpoint.write_log(
        'Total Param: {:.2f}K'.format(utility.print_network(model))
    )
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    start = time.time()
    t = Trainer_Search(args, loader, model, loss, checkpoint)
    checkpoint.write_log('Start training: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    t.train()
    total_time = time.time() - start
    checkpoint.write_log('Total training time: {} h {} min'.format(
        int(total_time / 3600), int(total_time / 60)))
    checkpoint.done()
    print('Well done bro~')


import torch
import sys
sys.path.append('E:\Codes\mycode7.0')
sys.path.append('E:\Codes\mycode7.0\cnn')
import os
import time
import utility
import data
from cnn import model
import loss
from option import args
from cnn.trainer import Trainer
import genotypes


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    checkpoint.write_log('This is one exp of exp_final!!!')
    checkpoint.write_log('pid: {}'.format(os.getpid()))

    genotype = eval("genotypes.%s" % args.arch)
    normal = genotype.normal
    checkpoint.write_log('{}:\nnormal: {}\nattention: {}'.format(args.arch, genotype.normal, genotype.attention))
    checkpoint.write_log('layers: {}\t scale: X{}'.format(args.layers, args.scale[0]))
    loader = data.Data(args)
    print('loader length: [train:{}]\t[test:{}]'.format(
        len(loader.loader_train),
        len(loader.loader_test)))

    model = model.NASnetWork(args, genotype)
    model = model.cuda()

    checkpoint.write_log(
        'Total Param: {}'.format(utility.print_network(model))
    )

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    start = time.time()
    t = Trainer(args, loader, model, loss, checkpoint)
    checkpoint.write_log('Start training: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    t.train()
    total_time = time.time() - start
    hour, minute, second = utility.second2standard(total_time)
    checkpoint.write_log('Finish training: {}\t\tTotal time: {}h {}min {}s'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        hour, minute, second))
    checkpoint.done()
    print('Well done bro~')

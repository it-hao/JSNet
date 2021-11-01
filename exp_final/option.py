import argparse

parser = argparse.ArgumentParser(description='NAS-SR')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--n_GPUs', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--id_GPU', type=str, default='0')

# Data specifications
parser.add_argument('--dir_data', type=str, default='F:/lwr/srdata/',
                    choices=['F:/lwr/srdata/', '../../../../srdata_ex', '/home/zhao/liaowenrui/srdata'])
parser.add_argument('--data_train', type=str, default='DIV2K')
parser.add_argument('--data_val', type=str, default='DIV2K_VAL',)
parser.add_argument('--data_test', type=str, default='Set14')
parser.add_argument('--n_train', type=int, default=800)
parser.add_argument('--n_val', type=int, default=100)
parser.add_argument('--offset_val', type=int, default=800)
parser.add_argument('--ext', type=str, default='sep', choices=['sep', 'sep_reset', 'bin', 'bin_reset'])
parser.add_argument('--scale', default='4')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--rgb_range', type=int, default=255)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--repeat', type=int, default=20)

# Model specifications
parser.add_argument('--C', type=int, default=16)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--op_nodes', type=int, default=3)
parser.add_argument('--at_nodes', type=int, default=3)
parser.add_argument('--bias', default=False)
parser.add_argument('--activation', type=str, default='relu', help='activation function')


# Architect specifications
parser.add_argument('--arch', type=str, default='SR_x2_final', help='which architecture to use')
parser.add_argument('--arch_step', type=int, default=60000, help='start architect step')
parser.add_argument('--arch_lr', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--unrolled', action='store_true', default=True)

# Training specifications
parser.add_argument('--max_steps', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--train_only', action='store_true', default=True)


# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_decay', type=int, default=200000)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9, choices=[0.5, 0.9])
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=0)

# Loss specifications
parser.add_argument('--loss', type=str, default='1.0*L1')

# Log specifications
parser.add_argument('--save', type=str, default='.', help='file name to save')
parser.add_argument('--load', type=str, default='.', help='file name to load')
parser.add_argument('--save_models', type=int, default=1000)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--print_loss', type=int, default=500)
parser.add_argument('--print_psnr', type=int, default=1000)
parser.add_argument('--print_genotype', type=int, default=1000)

# test
parser.add_argument('--create_results', action='store_true', default=True)
parser.add_argument('--pre_train', type=str, default='baseline')
parser.add_argument('--test_dir', type=str, default='./baseline')
parser.add_argument('--testset', type=str, default='Set5', choices=('Set5','Set14','B100','Urban100','Manga109'), help='dataset name for testing')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

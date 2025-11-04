import sys
import argparse
import os

from ms_common import Logger
from ms_ee_attack import MSEEAttack
from ms_defense import MSDefense
import ms_common as comm


def run_ee(args):

    if not args.l_only:
        scenario = "p_only"
    else:
        scenario = "l_only"

    msd = MSDefense(args)

    if args.dataset == 'FashionMNIST':
        msd.load(netv_path='saved_model/pretrained_net/vgg16_fashion_mnist.pth')
    else:
        return

    msa = MSEEAttack(args, defense_obj=msd)
    msa.load()

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    msa.ee_train_netS('res/ee_netS_%s_%s' % (args.dataset, scenario),
                   'res/ee_netG_%s_%s' % (args.dataset, scenario), 'res/ee_netGE_%s_%s' % (args.dataset, scenario))

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

def get_args(dataset, cuda, expt="attack", l_only=False):
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=cuda, action='store_true', help='using cuda')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_exploit', type=int, default=1, help='fix to be one')

    if not l_only:
        args.add_argument('--epoch_itrs', type=int, default=50)
        args.add_argument('--epoch_dg_s', type=int, default=5, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_g', type=int, default=1, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_ge', type=int, default=1, help='for training dynamic net G and net S')
    else:
        args.add_argument('--epoch_itrs', type=int, default=50)
        args.add_argument('--epoch_dg_s', type=int, default=5, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_g', type=int, default=1, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_ge', type=int, default=1, help='for training dynamic net G and net S')

    args.add_argument('--z_dim', type=int, default=128, help='the dimension of noise')
    args.add_argument('--batch_size_g', type=int, default=256, help='for training net G and net S')

    args.add_argument('--attack_exp', default=(expt == "attack"), action='store_true', help='running attack experiments')
    args.add_argument('--test_exp', default=(expt == "test"), action='store_true', help='running attack experiments')

    args.add_argument('--l_only', default=l_only, action='store_true', help='label only')

    if not l_only:
        args.add_argument('--epoch_dg', type=int, default=100, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.001)
        args.add_argument('--lr_tune_g', type=float, default=0.00001)
        args.add_argument('--lr_tune_ge', type=float, default=0.000001)
        args.add_argument('--steps', nargs='+', default=[0.5, 0.8], type=float)
        args.add_argument('--scale', type=float, default=3e-1)
    else:
        args.add_argument('--epoch_dg', type=int, default=50, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.001)
        args.add_argument('--lr_tune_g', type=float, default=0.00001)
        args.add_argument('--lr_tune_ge', type=float, default=0.000001)
        args.add_argument('--steps', nargs='+', default=[0.5, 0.8], type=float)
        args.add_argument('--scale', type=float, default=3e-1)
    args.add_argument('--dataset', type=str, default='FashionMNIST')
    args.add_argument('--res_filename', type=str, default='fashion_mnist_ee')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    if not (os.path.exists('res') and os.path.isdir('res')):
        os.mkdir('res')
    sys.stdout = Logger('res/ee_normal.log', sys.stdout)

    args = get_args(dataset="fashion_mnist", cuda=True, expt="attack", l_only=False)
    print(args)

    run_ee(args)

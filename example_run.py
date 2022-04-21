import os
import torch.nn.parallel
import torch.utils.data
import pandas as pd

from makedata import EPLData
from utils import *
from dataloaders import *
from EPL_train import EPL_train
from EPL_val import EPL_val

import argparse

parser = argparse.ArgumentParser(description='End-to-end Part Learning (EPL)')

# General training options
parser.add_argument('--dataset_name', default='IO.G.3000', type=str, help='Name of the dataset')
parser.add_argument('--stage', default='train', type=str, help='train,val,test')
parser.add_argument('--cv', default=1, type=int, help='Which split of cross validation')
parser.add_argument('--level', default=0, type=int, help='level used by openslide to read different resolution of svs files')
parser.add_argument('-j', '--workers', default=10, type=int)
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('-sl', '--scheduler_length', default=400, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('-wd', '--w_decay', default=1e-4, type=float)
parser.add_argument('--waist', default=64, type=int, help='Feature length')
parser.add_argument('-nc', '--num_classes', default=2, type=int)
parser.add_argument('-ps', '--patch_size', default=224, type=int)
parser.add_argument('--dev', action='store_true', help='if true, choosing a very small subset for development')
parser.add_argument('-pt', '--pretrained', action='store_true', help='if true, load pretrained weights')

# EPL specific training options
parser.add_argument('-n', '--n_cluster', default=12, type=int, help='Number of parts')
parser.add_argument('-p', '--nearest_p', default=1, type=int, help='Pool of approximation tile sampling')
parser.add_argument('-ssp', '--subsample_slide_percentage', default=1.0, type=float,
                    help='Percentage of slides to choose from training set')
parser.add_argument('-stp', '--subsample_tiles_per_slide', default=100, type=int,
                    help='Number of tiles to sample from each slide')
parser.add_argument('-ncps', '--n_cs_per_ss', default=2, type=float,
                    help='Number of constraint samples per supervised samples in training')
parser.add_argument('-ssps', '--supervised_subsample_per_slide', default=10, type=int,
                    help='Number of supervised samples to sample from each slide')
parser.add_argument('-cew', '--center_loss_weight', default=1.0, type=float)
parser.add_argument('-pw', '--part_weight', action='store_true',
                    help='If true, model considers proportion of tiles belonging to different parts of a slide')
parser.add_argument('-at', '--attribution', default=0, type=int, help='Attribution type; 0: no, 1: x*dy/dx, 2: gradient')
parser.add_argument('-ki', '--kmeans-iter', default=1, type=int)

# test stage options
parser.add_argument('-tsn', '--test_subsample_num', default=500, type=int)


def main():
    args = parser.parse_args()
    '''
    Make the dataset as a pandas dataframe with columns: [name, split, target, x, y]. 
    '''

    class MyEPLData(EPLData):
        def get_df(self):
            IO_info = args.dataset_name.split('.')[2:]
            path = '/lila/data/fuchs/projects/lung/IO_geo_library/library_k%s_cv%s.csv' % (IO_info[0], args.cv)
            dataset = pd.read_csv(path, header=None)
            self.df = dataset

    root_output = '/lila/data/fuchs/xiec/results/EPL/'
    subout = '%s_cv%s_n%s_at%s_p%s_pw%s' % (args.dataset_name, args.cv, args.n_cluster, args.attribution,
                                            args.nearest_p, args.part_weight)
    out_dir = os.path.join(root_output, subout)
    slide_path = '/scratch/lung_impacted/'

    EPL_data = MyEPLData(out_dir, slide_path, args)
    EPL_data.get_df()
    try:
        EPL_data.mean = np.load(os.path.join(EPL_data.root, 'mean.npy')).tolist()
        EPL_data.std = np.load(os.path.join(EPL_data.root, 'std.npy')).tolist()
        print('mean: ', EPL_data.mean)
        print('std: ', EPL_data.std)
    except:
        EPL_data.get_mean_std()

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.stage == 'train':
        EPL_train(EPL_data, args, gpu)

    elif args.stage == 'val':
        EPL_val(EPL_data, args, gpu, sel_epoch=0)

    elif args.stage == 'test':
        # update the output directory
        EPL_data.root = os.path.join(EPL_data.root, 'test')
        if not os.path.exists(EPL_data.root):
            os.mkdir(EPL_data.root)
        EPL_data.introspection_path = os.path.join(EPL_data.root, 'introspection')
        if not os.path.exists(EPL_data.introspection_path):
            os.mkdir(EPL_data.introspection_path)

        # load the test dataset to EPLData.df
        IO_info = args.dataset_name.split('.')[2:]
        test_file = '/lila/data/fuchs/projects/lung/IO_geo_library/library_k%s_test.csv' % IO_info[0]
        EPLData.df = pd.read_csv(test_file, header=None)
        EPL_val(EPL_data, args, gpu, sel_epoch=0)


if __name__ == '__main__':
    main()

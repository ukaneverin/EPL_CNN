import numpy as np
import os
import argparse
import torch
from dataloaders import DatasetFromLibrary
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from pdb import set_trace


class EPLData(ABC):
    def __init__(self, root: str, slide_path: str, args: argparse.Namespace):
        """
        A class to handle and process EPL data.
        :param root: root directory where all train-validation files will be saved under
        :param slide_path: directory where slides are stored
        :param args: the parsed parameters from running scripts
        """
        self.root = root
        self.level = args.level
        self.slide_path = slide_path
        self.patch_size = args.patch_size
        self.num_classes = args.num_classes
        self.dataset_name = args.dataset_name

        self.df = 0
        self.mean = 0
        self.std = 0

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.convergence_file = os.path.join(self.root, 'convergence.csv')
        # make sub-directories
        self.introspection_path = os.path.join(self.root, 'introspection')
        if not os.path.exists(self.introspection_path):
            os.mkdir(self.introspection_path)

        self.model_path = os.path.join(self.root, 'trained_models')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.centroid_path = os.path.join(self.root, 'centroids')
        if not os.path.exists(self.centroid_path):
            os.mkdir(self.centroid_path)

        self.attribution_path = os.path.join(self.root, 'attribution')
        if not os.path.exists(self.attribution_path):
            os.mkdir(self.attribution_path)

        self.sample_tile_path = os.path.join(self.root, 'sample_tiles')
        if not os.path.exists(self.sample_tile_path):
            os.mkdir(self.sample_tile_path)

        self.filtered_tile_path = os.path.join(self.root, 'filtered_tiles')
        if not os.path.exists(self.filtered_tile_path):
            os.mkdir(self.filtered_tile_path)

    @abstractmethod
    def get_df(self):
        """
        Customized method to get the pandas dataframe for training of the following columns:
        {slide_id, split, targets, x, y},
        where 'targets' can be a number or a vector (several numbers) for various tasks
        dataframe should be returned and also assigned to self.df
        """
        ...

    def get_mean_std(self):
        """
        Calculate the normalization parameters for R, G, B channels
        :return: normalization parameters: [mean, std]
        """
        print('==> Computing mean and std..')
        trainset_ms = self.df.loc[self.df.iloc[:, 1] == 'train'].to_numpy()
        np.random.shuffle(trainset_ms)
        trainset_ms = trainset_ms[:10000]
        mean_std_trans = {'img': transforms.ToTensor()}
        train_dset_ms = DatasetFromLibrary(trainset_ms, self.slide_path, mean_std_trans, self.patch_size, self.level)

        dataloader = torch.utils.data.DataLoader(train_dset_ms, batch_size=1024, shuffle=True, num_workers=10)
        mean = torch.zeros(3).cuda()
        std = torch.zeros(3).cuda()
        image_dimension_x = 0.0
        image_dimension_y = 0.0
        for img, _, _, _, _, _ in dataloader:
            image_dimension_x = img.shape[2]
            image_dimension_y = img.shape[3]
            for i in range(3):
                mean[i] += img[:, i, :, :].mean(1).mean(1).sum()
                non_reduced_mean = img[:, i, :, :].mean(1, keepdim=True).mean(2, keepdim=True).expand_as(
                    img[:, i, :, :])
                std[i] += (img[:, i, :, :] - non_reduced_mean).pow(2).sum()
        std.div_(image_dimension_x * image_dimension_y * len(train_dset_ms) - 1).pow_(0.5)
        mean.div_(len(train_dset_ms))

        print('mean: ', mean.cpu())
        print('std: ', std.cpu())
        np.save(os.path.join(self.root, 'mean.npy'), mean.cpu())
        np.save(os.path.join(self.root, 'std.npy'), std.cpu())
        self.mean = mean.cpu()
        self.std = std.cpu()

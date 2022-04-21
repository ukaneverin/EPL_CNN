import numpy as np
import torch
from torch.utils.data import Sampler
import os
from openslide import OpenSlide
from utils import slide_dist
import random


class DatasetFromLibrary(torch.utils.data.Dataset):
    def __init__(self, library, data_path, transform, patch_size, level):
        unique_ids = np.unique(library[:, 0])  # the input is downsampled and shuffled already
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name))))
        self.data = np.delete(library, 1, axis=1)
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.size = len(library)
        self.patch_size = patch_size
        self.level = level

    def __getitem__(self, index):
        im_id = self.data[index][0]
        target = self.data[index][1:-2].astype(float)
        x, y = self.data[index][-2:].astype(int)
        center_x = int(x) + int(self.patch_size / 2)
        center_y = int(y) + int(self.patch_size / 2)
        if self.level == 0:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size / 2),
                                                                          center_y - int(self.patch_size / 2)],
                                                                         self.level, [self.patch_size, self.patch_size])
        elif self.level == 1:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size),
                                                                          center_y - int(self.patch_size)],
                                                                         self.level - 1,
                                                                         [2 * self.patch_size, 2 * self.patch_size])
        elif self.level == 2:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size / 2),
                                                                          center_y - int(self.patch_size / 2)],
                                                                         self.level - 1, [self.patch_size, self.patch_size])
        img = img.resize((self.patch_size, self.patch_size)).convert('RGB')
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        img = self.transform['img'](img)

        return img, im_id, target, x, y, index

    def __len__(self):
        return self.size


def prepare_data(dataset, num_classes):
    trainset = dataset.loc[dataset.iloc[:, 1] == 'train']
    N_train = len(np.unique(trainset.iloc[:, 0]))
    unique_train_ids = []  # list of set: [(class1_ids), (class2_ids)]
    train_sets = []  # list of train_class matrix
    train_id_index_dict = dict()  # these indices correspond to each train_class matrix
    for c in range(num_classes):  # c is the number of classes; here in IO, it's 0, 1
        trainset_class = trainset.loc[trainset.iloc[:, 2] == c]  # here in IO, it's 0, 1
        train_sets.append(trainset_class)
        unique_train_ids_class = set(np.unique(trainset_class.iloc[:, 0]))
        unique_train_ids.append(unique_train_ids_class)
        for id in unique_train_ids_class:
            id_index = (trainset_class.iloc[:, 0].to_numpy() == id).nonzero()[0]
            train_id_index_dict[id] = id_index

    valset = dataset.loc[dataset.iloc[:, 1] == 'validation']
    unique_val_ids = set(np.unique(valset.iloc[:, 0]))
    val_id_index_dict = dict()
    for id in unique_val_ids:
        sub_index_id_all = (valset.iloc[:, 0].to_numpy() == id).nonzero()[0]
        val_id_index_dict[id] = sub_index_id_all

    processed_data_train = {
        'N_train': N_train,  # number of slides in train set
        'unique_train_ids': unique_train_ids,  # unique slide ids in train set
        'train_sets': train_sets,  # the dataframe of train set
        'train_id_index_dict': train_id_index_dict,  # a dictionary of id: index in whole dataframe
        'unique_val_ids': unique_val_ids,  # unique slide ids in validation set
        'valset': valset,  # the dataframe of validation set
        'val_id_index_dict': val_id_index_dict  # a dictionary of id: index in whole dataframe
    }
    return processed_data_train


def prepare_test_data(dataset, val=False):
    if not val:
        valset = dataset.loc[dataset.iloc[:, 1] == 'test']
    else:
        valset = dataset.loc[dataset.iloc[:, 1] == 'validation']
    unique_val_ids = set(np.unique(valset.iloc[:, 0]))
    val_id_index_dict = dict()
    for id in unique_val_ids:
        sub_index_id_all = (valset.iloc[:, 0].to_numpy() == id).nonzero()[0]
        val_id_index_dict[id] = sub_index_id_all

    return unique_val_ids, valset, val_id_index_dict


def train_sampling(processed_data_train, slide_path, trans, ids_sampled, args):
    ids_previously_sampled = [set()] * len(ids_sampled)
    sub_trainset = []
    sub_index_all = [[], []]
    for c in range(len(ids_sampled)):  # c is the number of classes; here in IO, it's 0, 1
        trainset_class = processed_data_train['train_sets'][c]
        ids_sampled_class = ids_sampled[
            c]  # ids_sampled is list of sets: [{sampled ids in class 0}, {sampled ids in class 1}, ...]
        unique_train_ids_class = processed_data_train['unique_train_ids'][c]

        id_sample_size = min(int(processed_data_train['N_train'] * args.ssp / len(ids_sampled)), len(unique_train_ids_class))
        if len(unique_train_ids_class - ids_sampled_class) > id_sample_size:  # if sampling size < len(rest samples)
            sub_ids = np.random.choice(list(unique_train_ids_class - ids_sampled_class), id_sample_size, replace=False)
            ids_previously_sampled[c] = set(sub_ids) | ids_sampled_class
        else:
            # choose (sampling_size - rest_sample_size) samples from a refreshed whole list
            sub_ids = np.random.choice(list(unique_train_ids_class),
                                       id_sample_size - len(unique_train_ids_class - ids_sampled_class), replace=False)
            ids_previously_sampled[c] = set(sub_ids)  # the sampled ids of the refreshed list
            sub_ids = np.hstack((sub_ids, np.asarray(list(unique_train_ids_class - ids_sampled_class))))

        for id in sub_ids:
            sub_index_id_all = processed_data_train['train_id_index_dict'][id]
            sub_index = np.random.choice(sub_index_id_all, min(args.stp, len(sub_index_id_all)), replace=False)
            sub_index_all[c].extend(sub_index)
            trainset_this_slide = trainset_class.iloc[sub_index].to_numpy().tolist()
            # print(trainset_class.iloc[sub_index])
            sub_trainset.extend(trainset_this_slide)

    sub_trainset = np.asarray(sub_trainset)
    np.random.shuffle(sub_trainset)
    train_dset = DatasetFromLibrary(sub_trainset, slide_path, trans, args.patch_size, args.level)
    return train_dset, sub_index_all, ids_previously_sampled


def val_sampling(slide_path, trans, processed_data, args):
    # Creating validation dataset
    unique_val_ids = np.asarray(list(processed_data['unique_val_ids']))
    np.random.shuffle(unique_val_ids)
    sub_valset = []
    for id in unique_val_ids:
        sub_index_id_all = processed_data['val_id_index_dict'][id]
        sub_index = np.random.choice(sub_index_id_all, min(args.stp, len(sub_index_id_all)), replace=False)
        valset_this_slide = processed_data['valset'].iloc[sub_index].to_numpy().tolist()
        sub_valset.extend(valset_this_slide)
        if len(sub_valset) > args.stp * 500:  # only subsample up to 500 slides to validate
            break

    sub_valset = np.asarray(sub_valset)
    np.random.shuffle(sub_valset)
    val_dset = DatasetFromLibrary(sub_valset, slide_path, trans, args.patch_size, args.level)
    return val_dset


def test_sampling(slide_path, trans, unique_val_ids, valset, val_id_index_dict, args):
    # Creating test dataset
    unique_val_ids = np.asarray(list(unique_val_ids))
    unique_val_ids.sort()
    # np.random.shuffle(unique_val_ids)
    sub_valset = []
    i = 0
    sampled_ids = set()
    for id in unique_val_ids:
        sampled_ids.add(id)
        sub_index_id_all = val_id_index_dict[id]
        sub_index = sub_index_id_all
        valset_this_slide = valset.iloc[sub_index].to_numpy().tolist()
        sub_valset.extend(valset_this_slide)
        i += 1
        if i > args.test_subsample_num:
            break
    sub_valset = np.asarray(sub_valset)
    val_dset = DatasetFromLibrary(sub_valset, slide_path, trans, args.patch_size, args.level)
    return val_dset, set(unique_val_ids) - sampled_ids


class SupervisedSampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, dset, supervised_sample, unique_ids, transform, args):
        # dset if the train_dataset.data
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name))))
        self.data = np.asarray(dset)
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.xindex_list = supervised_sample['xindex_list']
        self.size = len(supervised_sample['xindex_list'])
        self.target = supervised_sample['y_list']
        self.c_mask = supervised_sample['c_mask']
        self.pw_list = supervised_sample['pw_list']
        self.n_cluster = args.n_cluster
        self.patch_size = args.patch_size
        self.level = args.level

    def __getitem__(self, index):
        xindex = np.asarray(self.xindex_list[index])
        img_id = self.data[xindex, 0][0]
        target = self.target[index]
        c_mask = self.c_mask[index]
        pw = self.pw_list[index]
        data_list = self.data[xindex]  # [im_id, target, (tile_metrics), x, y], ...
        img_list = []
        tile_metrics_list = []
        i = 0
        for c in range(self.n_cluster):
            if c_mask[c] == 1:
                data = data_list[i]
                im_id = data[0]
                x, y = data[-2:].astype(int)
                center_x = int(x) + int(self.patch_size / 2)
                center_y = int(y) + int(self.patch_size / 2)
                if self.level == 0:
                    img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size / 2),
                                                                                  center_y - int(self.patch_size / 2)],
                                                                                 self.level,
                                                                                 [self.patch_size, self.patch_size])
                elif self.level == 1:
                    img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size),
                                                                                  center_y - int(self.patch_size)],
                                                                                 self.level - 1, [2 * self.patch_size,
                                                                                                  2 * self.patch_size])
                elif self.level == 2:
                    img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size / 2),
                                                                                  center_y - int(self.patch_size / 2)],
                                                                                 self.level - 1,
                                                                                 [self.patch_size, self.patch_size])
                img = img.resize((self.patch_size, self.patch_size)).convert('RGB')
                seed = np.random.randint(2147483647)  # make a seed with numpy generator
                random.seed(seed)
                img = self.transform['img'](img)
                img_list.append(img)

                i += 1
            else:
                # padding with zero tensor
                img_list.append(torch.zeros((3, 224, 224)))
        sm = 0
        return img_id, img_list, tile_metrics_list, target, c_mask, pw, sm

    def __len__(self):
        return self.size


class ConstraintSampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, dset, unique_ids, patch_size, level, transform):
        # dset if the train_dataset.data
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name))))
        self.data = np.asarray(dset)
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.size = len(dset)
        self.patch_size = patch_size
        self.level = level

    def __getitem__(self, index):
        im_id = self.data[index][0]
        x, y = self.data[index][-2:]
        center_x = int(x) + int(self.patch_size / 2)
        center_y = int(y) + int(self.patch_size / 2)
        tile_metrics = torch.FloatTensor(self.data[index][2:-2].astype(float))
        if self.level == 0:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size / 2),
                                                                          center_y - int(self.patch_size / 2)],
                                                                         self.level,
                                                                         [self.patch_size, self.patch_size])
        elif self.level == 1:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size),
                                                                          center_y - int(self.patch_size)],
                                                                         self.level - 1, [2 * self.patch_size,
                                                                                          2 * self.patch_size])
        elif self.level == 2:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([center_x - int(self.patch_size / 2),
                                                                          center_y - int(self.patch_size / 2)],
                                                                         self.level - 1,
                                                                         [self.patch_size, self.patch_size])

        img = img.resize((self.patch_size, self.patch_size)).convert('RGB')
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        img = self.transform['img'](img)

        return img, tile_metrics, im_id, index

    def __len__(self):
        return self.size


def CreatePartSample(unique_ids, total_id_old, target_old, full_code_old, assigned_clusters_old, attribution, waist, args, p,
                     ssps):
    supervised_index_list = []
    supervised_label_list = []  # these two lists should be of the same length
    supervised_clusters_mask = []
    supervised_clusters_weight = []
    slide_cluster_centers_dict = dict()  # {slide_id: slide_cluster_centers}
    for id in unique_ids:
        sid_index_id = (total_id_old == id).nonzero()[0]  # index of the tiles in the whole dset
        slide_label = target_old[sid_index_id[0]]
        slide_code = full_code_old[sid_index_id]
        assigned_clusters_old_sid = assigned_clusters_old[sid_index_id]
        dist_sid, slide_cluster_centers = slide_dist(slide_code, assigned_clusters_old_sid, attribution,
                                                     args.n_cluster, waist, args.at)  # dist: shape (n,m)
        slide_cluster_centers_dict[id] = slide_cluster_centers
        # sample the nearest p tiles for each cluster;
        unique_clusters_sid = np.unique(assigned_clusters_old_sid)  # only clusters in sid

        assignment_sorted = assigned_clusters_old_sid[dist_sid.argsort(dim=0)]
        tile_index_sorted = sid_index_id[dist_sid.argsort(dim=0)]
        aug_sample_index = []
        for n in unique_clusters_sid:
            part_index = tile_index_sorted[np.where(assignment_sorted[:, n] == n), n].reshape(-1)[:p].tolist()
            aug_sample_index.append(part_index)
        supervised_index_list_sid = []
        for i in range(ssps):
            supervised_index_list_sid.append([np.random.choice(a) for a in aug_sample_index])

        supervised_clusters_weight_sid = np.asarray([(assigned_clusters_old_sid == x).sum().item()
                                                     for x in range(args.n_cluster)])
        supervised_clusters_weight_sid = supervised_clusters_weight_sid / supervised_clusters_weight_sid.sum()
        supervised_clusters_weight.extend([supervised_clusters_weight_sid.tolist()] * len(supervised_index_list_sid))
        supervised_clusters_mask_sid = np.zeros(args.n_cluster, dtype=int)
        supervised_clusters_mask_sid[unique_clusters_sid] = 1  # only clusters in sid are 1, others are 0
        supervised_clusters_mask.extend([supervised_clusters_mask_sid.tolist()] * len(supervised_index_list_sid))
        supervised_index_list.extend(supervised_index_list_sid)
        supervised_label_list.extend([slide_label] * len(supervised_index_list_sid))

    supervised_index_list = np.asarray(supervised_index_list)
    supervised_clusters_mask = np.asarray(supervised_clusters_mask)

    supervised_sample = {
        'supervised_index_list': supervised_index_list,
        'supervised_clusters_weight': supervised_clusters_weight,
        'supervised_label_list': supervised_label_list,
        'supervised_clusters_mask': supervised_clusters_mask,
        'slide_cluster_centers_dict': slide_cluster_centers_dict
    }
    return supervised_sample

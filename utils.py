import numpy as np
import torch
from torch.utils.data import Sampler
import os
from openslide import OpenSlide
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

'''
Meters keep track of losses, accuracy, attribution, etc over training
'''


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = (self.sum + 1e-12) / (self.count + 1e-12)


class AccuracyMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, nc):
        self.nc = nc
        self.reset()

    def reset(self):
        self.correct = [0] * self.nc
        self.count = [0] * self.nc
        self.acc = [0] * self.nc

    def update(self, correct, c, total_n):
        (self.correct)[c] += correct
        (self.count)[c] += total_n
        (self.acc)[c] = ((self.correct)[c] + 1e-12) / ((self.count)[c] + 1e-12)


class AttributionMeter(object):
    def __init__(self, waist):
        self.val = torch.zeros(waist)
        self.sum = torch.zeros(waist)
        self.count = torch.zeros(waist)
        self.avg = torch.ones(waist)  # set for epoch 1: all ones
        self.waist = waist

    def reset(self):
        self.val = torch.zeros(self.waist)
        self.avg = torch.zeros(self.waist)
        self.sum = torch.zeros(self.waist)
        self.count = torch.zeros(self.waist)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = (self.sum + 1e-12) / (self.count + 1e-12)


'''
Saving and displaying functions
'''


def save_checkpoint(state, path, filename):
    fullpath = os.path.join(path, filename)
    torch.save(state, fullpath)


def save_error(trainAcc, valAcc, supervised_loss, center_loss, epoch, slname):
    if epoch == 0:
        f = open(slname, 'w+')
        f.write('epoch,trainAcc,valAcc,supervisedLoss,centerLoss\n')
        f.write('{},{:.4f},{:.4f},{:.4f},{:.4f}\n'
                .format(epoch + 1, trainAcc, valAcc, supervised_loss, center_loss))
        f.close()
    else:
        f = open(slname, 'a')
        f.write('{},{:.4f},{:.4f},{:.4f},{:.4f}\n'
                .format(epoch + 1, trainAcc, valAcc, supervised_loss, center_loss))
        f.close()


def display(data_path, cluster_centers, attribution, at, full_code, assigned_clusters, x_patch,
            y_patch, patch_size, level, total_id, epoch, out_dir):
    k = 0
    cols = 20
    n_cluster = cluster_centers.shape[0]
    grid_batch = torch.randn(n_cluster * cols, 3, patch_size, patch_size, requires_grad=False)
    for i in range(0, len(cluster_centers)):
        cluster_members = full_code[(assigned_clusters == i).nonzero().squeeze()]
        if cluster_members.size(0) < cols or len(cluster_members.size()) == 1:
            for n in range(0, cols):
                grid_batch[k] = torch.zeros(1, 3, patch_size, patch_size)
                k = k + 1
        else:
            cluster_x = x_patch[(assigned_clusters == i).nonzero().squeeze()]
            cluster_y = y_patch[(assigned_clusters == i).nonzero().squeeze()]
            cluster_id = total_id[(assigned_clusters == i).nonzero().squeeze()]
            # distances = torch.nn.PairwiseDistance()(cluster_members,
            #                                         cluster_centers[i].repeat(cluster_members.size(0), 1))

            if at != 0:
                distances = torch.pow(torch.pow((cluster_members - cluster_centers[i].repeat(cluster_members.size(0), 1))
                                                * (attribution), 2).sum(1), .5)
            else:
                distances = torch.pow(
                    torch.pow(cluster_members - cluster_centers[i].repeat(cluster_members.size(0), 1), 2).sum(1), .5)

            sel_index = distances.topk(cols, largest=False)[1]

            for n in range(0, len(sel_index)):
                index = sel_index[n]
                svs = OpenSlide(os.path.join(data_path, str(cluster_id[index])))
                center_x = int(cluster_x[index]) + int(patch_size / 2)
                center_y = int(cluster_y[index]) + int(patch_size / 2)
                if level == 0:
                    patch = svs.read_region([center_x - int(patch_size / 2), center_y - int(patch_size / 2)],
                                            level, [patch_size, patch_size])
                elif level == 1:
                    patch = svs.read_region([center_x - int(patch_size), center_y - int(patch_size)],
                                            level - 1, [2 * patch_size, 2 * patch_size])
                elif level == 2:
                    patch = svs.read_region([center_x - int(patch_size / 2), center_y - int(patch_size / 2)],
                                            level - 1, [patch_size, patch_size])
                patch = patch.resize((224, 224)).convert('RGB')
                patch = transforms.ToTensor()(patch)
                grid_batch[k] = patch
                k = k + 1
    save_image(grid_batch, os.path.join(out_dir, 'image_{}.png'.format(epoch + 1)), nrow=cols)


'''
Loss functions
'''


class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = F._Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class MSECenter(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSECenter, self).__init__(size_average, reduce, reduction)

    def forward(self, embedding, centers, at, attribution):
        # center_loss = F.mse_loss(embedding, centers, reduction=self.reduction)
        n = embedding.size(0)
        m = centers.size(0)
        d = embedding.size(1)
        x = embedding.unsqueeze(1).expand(n, m, d)
        y = centers.unsqueeze(0).expand(n, m, d)
        if at != 0:
            attribution = (attribution - attribution.min()) / (attribution.mean() - attribution.min())
            center_loss = torch.pow((x - y) * attribution, 2).mean()
        else:
            center_loss = torch.pow(x - y, 2).mean()
        return center_loss


class CESupervised(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CESupervised, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        supervised_loss = F.cross_entropy(input, target, reduction=self.reduction)
        return supervised_loss


'''
Others
'''


def slide_dist(slide_code, slide_assignment, attribution, n_cluster, waist, at=0):
    slide_cluster_centers = torch.randn(n_cluster, waist, requires_grad=False)
    for j in range(0, n_cluster):
        cluster_j = slide_code[(slide_assignment == j).nonzero().squeeze()]
        if len(cluster_j) != 0:
            slide_cluster_centers[j] = cluster_j.mean(0)
    n = slide_code.size(0)
    m = slide_cluster_centers.size(0)
    d = slide_code.size(1)
    x = slide_code.unsqueeze(1).expand(n, m, d)
    y = slide_cluster_centers.unsqueeze(0).expand(n, m, d)
    if at != 0:
        dist = torch.pow(torch.pow((x - y) * attribution, 2).sum(2), .5)
    else:
        dist = torch.pow(torch.pow(x - y, 2).sum(2), .5)
    return dist, slide_cluster_centers


def k_means(code, cluster_centers, assignments, n_cluster, at, attribution, n_iters=1):
    for iters in range(n_iters):
        # calculate assignments of new code of new batch to new centers
        for j in range(0, n_cluster):
            cluster_j = code[(assignments == j).nonzero().squeeze()]
            if len(cluster_j) != 0:
                cluster_centers[j] = cluster_j.mean(0)
        n = code.size(0)  # N,64
        m = cluster_centers.size(0)  # M, 64
        d = code.size(1)  # d=64
        x = code.unsqueeze(1).expand(n, m, d)
        y = cluster_centers.unsqueeze(0).expand(n, m, d)
        if at != 0:
            dist = torch.pow(torch.pow((x - y) * (attribution), 2).sum(2), .5)
        else:
            dist = torch.pow(torch.pow((x - y), 2).sum(2), .5)
        assignments = dist.argmin(dim=1)
    return cluster_centers

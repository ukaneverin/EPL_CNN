import os
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models.resnet import ConcatResNet, resblock34
from pdb import set_trace
import matplotlib
from utils import *
from dataloaders import SupervisedSampleDataset, test_sampling, prepare_test_data, CreatePartSample

matplotlib.use('agg')
import matplotlib.pyplot as plt


def EPL_val(EPL_data, args, gpu, sel_epoch):
    print('Initializing end-to-end part learning (EPL) for whole slide assessment validation stage')
    print('1...Loading the best model weights and cluster centers.')
    image_channels = 3
    feature_length = args.waist

    model_tile = resblock34(args.waist, image_channels).to(gpu)
    model_slide = ConcatResNet(model_tile, args.waist, num_classes=2, num_clusters=args.n_cluster,
                               if_pw=args.part_weight).to(gpu)

    # load weights and centers for the best validation epoch
    with open(EPL_data.convergence_file, 'r') as f:
        convergence_epochs = np.asarray([line.strip().split(',') for line in f]).astype(float)
    epoch_num = convergence_epochs[:, 0].astype(int)
    val_acc = convergence_epochs[:, 2]

    best_epoch = epoch_num[np.argmax(val_acc)]
    if sel_epoch != 0:
        best_epoch = sel_epoch
    print('epoch best: ', best_epoch)

    # load centroids of best epoch
    best_centers_file = os.path.join(EPL_data.centroid_path, '%s_centers.csv' % best_epoch)
    with open(best_centers_file, 'r') as f:
        cluster_centers = torch.Tensor(np.asarray([line.strip().split(',') for line in f][1:]).astype(float))
    # load the best tile and slide models weights
    best_model_tile = torch.load(os.path.join(EPL_data.model_path, 'checkpoint_{}_tile.pth'.format(best_epoch)))
    model_tile.load_state_dict(best_model_tile['state_dict'])
    best_model_slide = torch.load(os.path.join(EPL_data.model_path, 'checkpoint_{}_slide.pth'.format(best_epoch)))
    model_slide.load_state_dict(best_model_slide['state_dict'])

    # load feature attribution if any
    if args.attribution != 0:
        best_attribution_file = os.path.join(EPL_data.attribution_path, '%s_attribution.csv' % best_epoch)
        with open(best_attribution_file, 'r') as f:
            attribution = torch.Tensor(np.asarray([line.strip().split(',') for line in f][1:]).astype(float)).reshape(-1)
            print('Attribution: %s' % attribution)
    else:
        attribution = torch.ones(feature_length)

    trans = {'img': transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=EPL_data.mean, std=EPL_data.std)])
             }

    model_slide.eval()

    unique_val_ids, valset, val_id_index_dict = prepare_test_data(EPL_data.df, val=True)

    AccMeter_val = AccuracyMeter(args.num_classes)

    prediction_scores = []
    save_hx = []
    rest_val_ids = unique_val_ids
    while len(rest_val_ids) > 0:
        val_dset, rest_val_ids = test_sampling(EPL_data.slide_path, trans, rest_val_ids, valset, val_id_index_dict, args)
        with torch.no_grad():
            val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1024, shuffle=False, num_workers=args.workers)
            val_code = torch.zeros(len(val_loader.dataset), feature_length, requires_grad=False)
            target_val = torch.LongTensor(np.asarray(val_dset.data)[:, 1].astype(int))
            total_id_val = np.asarray(val_dset.data)[:, 0]  # slide id
            for i, (input, _, _, _, _, index) in enumerate(val_loader):
                input = input.to(gpu)
                code = model_tile(input)
                val_code[index] = code.detach().cpu()

            #######
            n = val_code.size(0)
            m = cluster_centers.size(0)
            d = val_code.size(1)
            x = val_code.unsqueeze(1).expand(n, m, d)
            y = cluster_centers.unsqueeze(0).expand(n, m, d)
            if args.attribution != 0:
                dist = torch.pow(torch.pow((x - y) * (attribution), 2).sum(2), .5)
            else:
                dist = torch.pow(torch.pow(x - y, 2).sum(2), .5)
            assigned_clusters_val = dist.argmin(dim=1)

            unique_ids = np.unique(total_id_val)
            supervised_test_sample = CreatePartSample(unique_ids, total_id_val, target_val, val_code, assigned_clusters_val,
                                                      attribution, feature_length, args, 1, 1)

            supervised_dset = SupervisedSampleDataset(EPL_data.slide_path, val_dset.data, supervised_test_sample, unique_ids,
                                                      trans, args)
            supervised_loader = torch.utils.data.DataLoader(supervised_dset, batch_size=20, shuffle=False,
                                                            num_workers=args.workers)
        # here we want the grads for attribution analysis
        for img_id, img_list, tile_metrics_list, target, c_mask, pw, sm in supervised_loader:
            img_list = [x.to(gpu) for x in img_list]
            sm = sm.to(gpu)
            c_mask = c_mask.numpy()
            output, hx = model_slide(img_list, c_mask, pw, sm)

            target = target.to(gpu)
            supervised_loss = CESupervised().to(gpu)
            supervised_loss = supervised_loss(output, target.squeeze())

            loss = supervised_loss
            loss.backward()
            hx_grad = hx.grad.detach().cpu()
            display_eval(img_id, img_list, target, c_mask, hx_grad, EPL_data.introspection_path, EPL_data.mean, EPL_data.std,
                         args)
            break

        # no grad for result scores
        supervised_loader = torch.utils.data.DataLoader(supervised_dset, batch_size=128, shuffle=False,
                                                        num_workers=args.workers)
        with torch.no_grad():
            for img_id, img_list, tile_metrics_list, target, c_mask, pw, sm in supervised_loader:
                img_list = [x.to(gpu) for x in img_list]
                sm = sm.to(gpu)
                c_mask = c_mask.numpy()
                output, hx = model_slide(img_list, c_mask, pw, sm)
                target = target.to(gpu)

                # combine results to write
                write_ids = np.asarray(img_id).reshape(-1, 1)
                preds = torch.max(F.softmax(output.detach(), dim=1), dim=1)[1].detach()
                write_scores = F.softmax(output.detach(), dim=1)[:, 1].cpu().detach().numpy().reshape(-1, 1)
                write_targets = target.cpu().detach().numpy().reshape(-1, 1)
                write_preds = preds.cpu().numpy().reshape(-1, 1)

                write_hx = np.hstack((write_ids, hx.cpu())).tolist()
                save_hx.extend(write_hx)
                write_all = np.hstack((write_ids, write_targets, write_scores, write_preds)).tolist()
                prediction_scores.extend(write_all)

                for c in range(args.num_classes):  # need modification for multi-class
                    batch_class_size = torch.sum(target.squeeze().data == c).item()
                    batch_corrects = torch.sum(preds[target.squeeze().data == c] == c).item()
                    AccMeter_val.update(batch_corrects, c, batch_class_size)

            print('Validation Accuracy: {Acc:.4f}\t'.format(Acc=sum(AccMeter_val.acc) / args.num_classes))

            del input
            del img_list
            torch.cuda.empty_cache()
            print('Number of remaining slides: ', len(rest_val_ids))

    prediction_scores = np.asarray(prediction_scores)
    np.savetxt(os.path.join(EPL_data.root, 'test_score.csv'), prediction_scores, fmt='%5s', delimiter=",")
    save_hx = np.asarray(save_hx)
    np.savetxt(os.path.join(EPL_data.root, 'hx.csv'), save_hx, fmt='%5s', delimiter=",")

    # plot ROC curve
    from sklearn.metrics import roc_curve, roc_auc_score
    roc_path = os.path.join(EPL_data.root, 'roc')
    if not os.path.exists(roc_path):
        os.mkdir(roc_path)

    y = prediction_scores[:, 1].reshape(-1).astype(int)
    scores = prediction_scores[:, 2].reshape(-1).astype(float)
    fpr, tpr, thresholds = roc_curve(y, scores)
    roc_auc = roc_auc_score(y, scores)
    roc_file = open(os.path.join(roc_path, '%s.%s' % (best_epoch, roc_auc)), 'w+')
    roc_file.close()
    print('AUROC: ', roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='slide_classification (area = %0.3f)' % roc_auc)  # the tile level curve
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: %s test' % args.dataset_name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(EPL_data.root, 'ROC.png'))
    plt.close()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def display_eval(img_id, img_list, target, c_mask, hx_grad, out_dir, mean, std, args):
    # save the grid_batch first
    k = 0
    cols = 20
    grid_batch = torch.randn(args.n_cluster * cols, 3, args.patch_size, args.patch_size, requires_grad=False)
    for i in range(args.n_cluster):
        for j in range(cols):
            if c_mask[j, i] == 0:  # if there is no tile in this cluster for a img then save blank
                grid_batch[k] = torch.zeros(1, 3, args.patch_size, args.patch_size)
            else:
                unnormalize = UnNormalize(mean, std)
                grid_batch[k] = unnormalize(img_list[i][j][:3])
            k = k + 1
    save_image(grid_batch, os.path.join(out_dir, 'inference_tiles.png'), nrow=cols)

    # now save the cluster importance
    if len(hx_grad.shape) == 2:
        attribution_eval = torch.abs(hx_grad.reshape(hx_grad.shape[0], args.n_cluster, -1).mean(-1))
    else:
        attribution_eval = torch.abs(hx_grad.reshape(args.num_classes, hx_grad.shape[0], args.n_cluster, -1).mean(-1))
    min_val = attribution_eval.min(-1)[0].unsqueeze(-1).expand_as(attribution_eval)
    max_val = attribution_eval.max(-1)[0].unsqueeze(-1).expand_as(attribution_eval)
    attribution_eval = ((attribution_eval - min_val) / (max_val - min_val))
    attribution_eval = attribution_eval.mean(-2)
    if len(attribution_eval.shape) == 2:
        plt.imshow(attribution_eval.transpose(1, 0), cmap='hot', interpolation='nearest')
    else:
        plt.imshow(attribution_eval.unsqueeze(-1), cmap='hot', interpolation='nearest')
    plt.savefig(os.path.join(out_dir, 'heatmap.png'))

    # save the img_id and targets
    info_file = open(os.path.join(out_dir, 'info.csv'), 'w+')
    info_file.write(','.join(img_id) + '\n')
    target = target.cpu().numpy().astype(str)
    if len(hx_grad.shape) != 2:
        target = ['_'.join(x) for x in target]
        info_file.write(','.join(target) + '\n')
    else:
        info_file.write(','.join(target.squeeze()) + '\n')
    info_file.close()

import os
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
from models.resnet import ConcatResNet, resblock34
from pdb import set_trace
from utils import *
from dataloaders import *


def EPL_train(EPL_data, args, gpu):
    if args.dev:  # make a very small subset of samples for debugging etc..
        EPL_data.df = EPL_data.df.sample(1000)
    '''
    Load tile and slide model, create optimizer, scheduler, transforms
    '''
    print('Initializing end-to-end part learning (EPL) for whole slide assessment')
    image_channels = 3
    feature_length = args.waist
    model_tile = resblock34(args.waist, image_channels).to(gpu)

    if args.pretrained:
        # use pretrained weights for tile encoder
        pretrained_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        model_dict = model_tile.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k.split('.')[0] != 'layer4' and k.split('.')[0] != 'fc'}
        model_dict.update(pretrained_dict)
        model_tile.load_state_dict(model_dict)

    model_slide = ConcatResNet(model_tile, args.waist, num_classes=2, num_clusters=args.n_cluster,
                               if_pw=args.part_weight).to(gpu)
    optimizer = torch.optim.SGD(model_slide.parameters(), lr=args.lr, weight_decay=args.w_decay)
    sl = args.scheduler_length
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[sl, 2 * sl, 3 * sl, 4 * sl], gamma=args.gamma)

    trans_train = {'img': transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=EPL_data.mean, std=EPL_data.std)])
                   }
    trans_val = {'img': transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=EPL_data.mean, std=EPL_data.std)])
                 }
    '''
    Initialize training set, global centers, assignments.
    '''
    processed_data_train = prepare_data(EPL_data.df, args.num_classes)
    ids_sampled = [set()] * args.num_classes
    train_dset_old, tile_index_old, ids_sampled = train_sampling(processed_data_train, EPL_data.slide_path,
                                                                 trans_train,
                                                                 ids_sampled,
                                                                 args)

    cluster_centers = torch.randn(args.n_cluster, feature_length, requires_grad=False)
    assigned_clusters_old = torch.LongTensor(np.random.randint(args.n_cluster, size=(1, train_dset_old.size))).squeeze()
    attribution = AttributionMeter(feature_length)
    part_attr = AttributionMeter(args.n_cluster)
    for epoch in range(0, args.epochs):
        print('current learning rate: ', optimizer.param_groups[0]['lr'])

        train_loader_old = torch.utils.data.DataLoader(train_dset_old, batch_size=1024, shuffle=False,
                                                       num_workers=args.workers)
        # Each data loaded is: [img, im_id, targets, x, y, index]

        '''
        Prepare part samples and MSE samples
        '''
        model_slide.eval()
        with torch.no_grad():
            full_code_old = torch.zeros(len(train_loader_old.dataset), feature_length,
                                        requires_grad=False)  # (n_samples, n_features)
            target_old = torch.LongTensor(np.asarray(train_dset_old.data)[:, 1].astype(int))
            total_id_old = np.asarray(train_dset_old.data)[:, 0]
            print('In total {} tiles of the current subsampled dataset'.format(len(train_dset_old)))
            for i, (input, _, _, _, _, index) in enumerate(train_loader_old):
                input = input.to(gpu)
                code = model_tile(input)
                full_code_old[index] = code.detach().cpu()
        del input

        unique_ids = np.unique(total_id_old)
        # load tiles for MSE loss
        constraint_dset = ConstraintSampleDataset(EPL_data.slide_path, train_dset_old.data, unique_ids, args.patch_size,
                                                  args.level, trans_train)  # getting the whole training set
        constraint_loader = torch.utils.data.DataLoader(constraint_dset,
                                                        batch_size=int(args.n_cs_per_ss * args.batch_size),
                                                        shuffle=False,
                                                        num_workers=args.workers)
        # centroid approximation for slide loss
        supervised_train_sample = CreatePartSample(unique_ids, total_id_old, target_old, full_code_old,
                                                   assigned_clusters_old,
                                                   attribution.avg, feature_length, args, args.p,
                                                   args.supervised_subsample_per_slide)
        supervised_dset = SupervisedSampleDataset(EPL_data.slide_path, train_dset_old.data, supervised_train_sample,
                                                  unique_ids,
                                                  trans_train, args)
        supervised_loader = torch.utils.data.DataLoader(supervised_dset, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.workers)
        # calculate the ramp up weights for MSE loss
        if epoch == 0:
            ramp_weight = 0
        else:
            ramp_weight = 1.0
        print('Current weight of constraint loss: {:.4f}'.format(ramp_weight))

        iter = 0
        lossMeter_supervised = AverageMeter()
        lossMeter_center = AverageMeter()
        AccMeter = AccuracyMeter(args.num_classes)
        print('4...Starting the training...')
        print('Epoch|Iteration\t'
              'SupervisedLoss\t'
              'CenterLoss\t')
        model_slide.train()
        attribution_old = attribution.avg
        attribution.reset()
        part_attr.reset()
        '''
        Training iterations
        '''
        for (_, img_list, tile_metrics_list, target, c_mask, pw, sm), (
                constraint_img, constraint_tile_metrics, batch_img_id, index) in zip(supervised_loader, constraint_loader):
            img_list = [x.to(gpu) for x in img_list]
            sm = sm.to(gpu)
            c_mask = c_mask.numpy()
            output, hx = model_slide(img_list, c_mask, pw, sm)
            batch_img_id = np.asarray(batch_img_id)
            batch_assignments = assigned_clusters_old[index]

            batch_centers = []
            for img_id, assignment in zip(batch_img_id, batch_assignments):
                batch_centers.append(slide_cluster_centers_dict[img_id][assignment].numpy().tolist())
            batch_centers = torch.FloatTensor(batch_centers)
            constraint_output = model_tile(constraint_img.to(gpu))

            target = target.to(gpu)
            supervised_loss = CESupervised().to(gpu)
            supervised_loss = supervised_loss(output, target.squeeze())

            center_loss = MSECenter().to(gpu)
            center_loss = center_loss(constraint_output, batch_centers.to(gpu), args.attribution, attribution_old.to(gpu))
            # the more embedding is modified, the worse the chosen tiles can represent the centroids

            loss = supervised_loss + args.center_loss_weight * ramp_weight * center_loss

            optimizer.zero_grad()
            loss.backward()
            if args.attribution != 0:
                if args.attribution == 1:
                    # using x * x.grad as the feature attribution estimation
                    batch_attribution = torch.abs(hx * hx.grad).detach().cpu()
                elif args.attribution == 2:
                    # using x.grad as the feature attribution estimation
                    batch_attribution = torch.abs(hx.grad).detach().cpu()
                n_batch = batch_attribution.shape[0]
                batch_attribution = batch_attribution.reshape(batch_attribution.shape[0], args.n_cluster, -1)[:, :,
                                    :feature_length]
                batch_part_attribution = batch_attribution.mean((0, 2))
                batch_attribution = batch_attribution.mean((0, 1))
                attribution.update(batch_attribution, n=n_batch)
                part_attr.update(batch_part_attribution, n=n_batch)

            if epoch != 0:
                optimizer.step()  # update to new model by the loss with old sample
            preds = torch.max(F.softmax(output.detach(), dim=1), dim=1)[1].detach()
            for c in range(args.num_classes):  # need modification for multi-class
                batch_class_size = torch.sum(target.squeeze().data == c).item()
                batch_corrects = torch.sum(preds[target.squeeze().data == c] == c).item()
                AccMeter.update(batch_corrects, c, batch_class_size)

            lossMeter_supervised.update(supervised_loss.item(), target.size(0))
            lossMeter_center.update(center_loss.item(), target.size(0))
            print('[{0}][{1}/{2}]\t'
                  '{lossMeter_supervised.val:.4f} ({lossMeter_supervised.avg:.4f})\t'
                  '{lossMeter_center.val:.4f} ({lossMeter_center.avg:.4f})\t'
                  .format(epoch + 1, iter + 1, len(supervised_loader),
                          lossMeter_supervised=lossMeter_supervised,
                          lossMeter_center=lossMeter_center))
            iter += 1

        del img_list
        print('Training Accuracy: {Acc:.4f}\t'.format(Acc=sum(AccMeter.acc) / args.num_classes))
        print('Cluster size: ', torch.stack([(assigned_clusters_old == c).sum() for c in range(0, len(cluster_centers))]))
        # calculate the new code of the old sample
        full_code_old = torch.zeros(len(train_loader_old.dataset), feature_length,
                                    requires_grad=False)  # (n_samples, n_features)
        model_slide.eval()
        with torch.no_grad():
            for i, (input, _, _, _, _, index) in enumerate(train_loader_old):
                input = input.to(gpu)
                code = model_tile(input)
                full_code_old[index] = code.detach().cpu()
        del input

        # calcualte new centers based on old sample new code and old assignment
        cluster_centers = k_means(full_code_old, cluster_centers, assigned_clusters_old,
                                  args.n_cluster, args.attribution,
                                  attribution=attribution.avg,
                                  n_iters=args.kmeans_iter)

        '''
        Calculate the validation classfification accuracy
        '''
        torch.cuda.empty_cache()
        if epoch % 5 == 1:
            model_slide.eval()
            with torch.no_grad():
                val_dset = val_sampling(EPL_data.slide_path, trans_val, processed_data_train, args)
                val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1024, shuffle=False, num_workers=args.workers)
                val_code = torch.zeros(len(val_loader.dataset), feature_length, requires_grad=False)
                target_val = torch.LongTensor(np.asarray(val_dset.data)[:, 1].astype(int))
                total_id_val = np.asarray(val_dset.data)[:, 0]  # slide id
                x_patch_val = np.asarray(val_dset.data)[:, -2].astype(int)
                y_patch_val = np.asarray(val_dset.data)[:, -1].astype(int)
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
                    dist = torch.pow(torch.pow((x - y) * (attribution.avg), 2).sum(2), .5)
                else:
                    dist = torch.pow(torch.pow(x - y, 2).sum(2), .5)
                assigned_clusters_val = dist.argmin(dim=1)

                unique_ids = np.unique(total_id_val)
                supervised_val_sample = CreatePartSample(unique_ids, total_id_val, target_val, val_code,
                                                         assigned_clusters_val, attribution.avg, feature_length, args, 1, 1)
                supervised_dset = SupervisedSampleDataset(EPL_data.slide_path, val_dset.data, supervised_val_sample,
                                                          unique_ids, trans_val, args)
                supervised_loader = torch.utils.data.DataLoader(supervised_dset, batch_size=128,
                                                                shuffle=False,
                                                                num_workers=args.workers)
                AccMeter_val = AccuracyMeter(args.num_classes)

                img_id_val = []
                for img_id, img_list, tile_metrics_list, target, c_mask, pw, sm in supervised_loader:
                    img_id_val.extend(list(img_id))
                    img_list = [x.to(gpu) for x in img_list]
                    sm = sm.to(gpu)
                    c_mask = c_mask.numpy()
                    output, hx = model_slide(img_list, c_mask, pw, sm)
                    target = target.to(gpu)
                    preds = torch.max(F.softmax(output.detach(), dim=1), dim=1)[1].detach()
                    for c in range(args.num_classes):  # need modification for multi-class
                        batch_class_size = torch.sum(target.squeeze().data == c).item()
                        batch_corrects = torch.sum(preds[target.squeeze().data == c] == c).item()
                        AccMeter_val.update(batch_corrects, c, batch_class_size)

                print('Validation Accuracy: {Acc:.4f}\t'.format(Acc=sum(AccMeter_val.acc) / args.num_classes))
                del input
                del img_list
                torch.cuda.empty_cache()

            '''
            Write results and save checkpoints models
            '''
            save_error(sum(AccMeter.acc) / args.num_classes, sum(AccMeter_val.acc) / args.num_classes,
                       lossMeter_supervised.avg, lossMeter_center.avg, epoch, EPL_data.convergence_file)
            display(EPL_data.slide_path, cluster_centers, attribution.avg, args.attribution, val_code, assigned_clusters_val,
                    x_patch_val, y_patch_val, args.patch_size, args.level, total_id_val, epoch,
                    EPL_data.sample_tile_path)
            """"""""""""""""""""""""""""""""""""""""""""
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_tile.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, EPL_data.model_path, 'checkpoint_{}_tile.pth'.format(epoch + 1))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_slide.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, EPL_data.model_path, 'checkpoint_{}_slide.pth'.format(epoch + 1))
            centersout = pd.DataFrame(cluster_centers.numpy())
            centersout.to_csv(os.path.join(EPL_data.centroid_path, str(epoch + 1) + '_centers.csv'), index=False)
            attribution_out = pd.DataFrame((attribution.avg).numpy())
            attribution_out.to_csv(os.path.join(EPL_data.attribution_path, str(epoch + 1) + '_attribution.csv'),
                                   index=False)

        # calculate the new code of new sample
        train_dset_new, tile_index_new, ids_sampled = train_sampling(processed_data_train, EPL_data.slide_path, trans_train,
                                                                     ids_sampled, args)
        train_loader_new = torch.utils.data.DataLoader(train_dset_new, batch_size=1024, shuffle=False,
                                                       num_workers=args.workers)
        full_code_new = torch.zeros(len(train_loader_new.dataset), feature_length,
                                    requires_grad=False)  # (n_samples, n_features)
        with torch.no_grad():
            for i, (input, _, _, _, _, index) in enumerate(train_loader_new):
                input = input.to(gpu)
                code = model_tile(input)
                full_code_new[index] = code.detach().cpu()

        del input
        # calculate assignments of new sample to new centers
        n = full_code_new.size(0)
        m = cluster_centers.size(0)
        d = full_code_new.size(1)
        x = full_code_new.unsqueeze(1).expand(n, m, d)
        y = cluster_centers.unsqueeze(0).expand(n, m, d)
        if args.attribution != 0:
            dist = torch.pow(torch.pow((x - y) * (attribution.avg), 2).sum(2), .5)
        else:
            dist = torch.pow(torch.pow(x - y, 2).sum(2), .5)
        assigned_clusters_new = dist.argmin(dim=1)
        # new assignments becomes old assignments (different training sets)
        assigned_clusters_old = assigned_clusters_new
        train_dset_old = train_dset_new
        tile_index_old = tile_index_new

        scheduler.step()

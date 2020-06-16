from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import PoseDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
from pointnet.fusion_model import PoseNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2 
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import sklearn.metrics

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchsize', type=int, default=64, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_annotation', type=str, required=True, help="dataset coco file path")
    parser.add_argument('--test_dataset_annotation', type=str, required=False, help="test dataset coco file path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--render_poses', action='store_true', help="use pose rendering for viz")
    parser.add_argument('--test_only', action='store_true', help="use pose rendering for viz")
    parser.add_argument('--model', type=str, default='', help='model path')

    opt = parser.parse_args()
    print(opt)
    return opt


class ModelInterface():
    def __init__(self, opt):
        self.opt = opt
        # if opt.dataset_type == 'shapenet':
        #     self.dataset = ShapeNetDataset(
        #                         root=opt.dataset,
        #                         classification=True,
        #                         npoints=opt.num_points)
        #     self.test_dataset = ShapeNetDataset(
        #                             root=opt.dataset,
        #                             classification=True,
        #                             split='test',
        #                             npoints=opt.num_points,
        #                             data_augmentation=False)

        # elif opt.dataset_type == 'modelnet40':
        #     self.dataset = ModelNetDataset(
        #                         root=opt.dataset,
        #                         npoints=opt.num_points,
        #                         split='trainval')

        #     self.test_dataset = ModelNetDataset(
        #                             root=opt.dataset,
        #                             split='test',
        #                             npoints=opt.num_points,
        #                             data_augmentation=False)  

        if opt.dataset_type == "ycb":
            self.opt.num_points = 2000 #number of points on the input pointcloud
            self.opt.outf = 'trained_models/ycb' #folder to save trained models
            self.opt.log_dir = 'experiments/logs/ycb' #folder to save logs
            self.opt.noise_trans = 0.00
            self.opt.feature_transform = False
            self.opt.num_objects = 1
            self.dataset = PoseDataset(
                                'train', 
                                self.opt.num_points, 
                                True, 
                                self.opt.dataset, 
                                self.opt.dataset_annotation,
                                self.opt.noise_trans, 
                                False)
            if 'test_dataset_annotation' in self.opt:
                self.test_dataset = PoseDataset(
                                        'test', 
                                        opt.num_points, 
                                        False, 
                                        self.opt.dataset, 
                                        self.opt.test_dataset_annotation,
                                        self.opt.noise_trans, 
                                        False)
            else:
                print("Not loading test dataset!")

            print('num pose samples', self.dataset.num_pose_samples)
        else:
            exit('wrong dataset type')

        self.tboard = SummaryWriter(self.opt.log_dir)

        self.dataloader = torch.utils.data.DataLoader(
                            self.dataset,
                            batch_size=self.opt.batchsize,
                            shuffle=True,
                            num_workers=int(self.opt.workers))

        self.testdataloader = torch.utils.data.DataLoader(
                                self.test_dataset,
                                batch_size=self.opt.batchsize,
                                shuffle=False,
                                num_workers=int(self.opt.workers))

        print("Train dataset size :{}, Test dataset size : {}".format(len(self.dataset), len(self.test_dataset)))
        num_classes = len(self.dataset.classes)
        print('classes', self.dataset.classes)

        # self.classifier = PointNetCls(k=self.dataset.num_pose_samples, feature_transform=self.opt.feature_transform)
        self.estimator = PoseNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects, num_classes = self.dataset.num_pose_samples)
        print(self.estimator)
        self.estimator.cuda()
        
        if self.opt.model != '':
            print("Preloading saved model : {}".format(self.opt.model))
            # self.classifier.load_state_dict(torch.load(self.opt.model))
            # print(torch.load(self.opt.model))
            self.estimator.load_state_dict(torch.load(self.opt.model))


        try:
            os.makedirs(self.opt.outf)
        except OSError:
            pass

    def compute_multilabel_ap(self, gt, pred, average="macro"):
        """
        Compute the multi-label classification accuracy.
        Args:
            gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
                image.
            pred (np.ndarray): Shape Nx20, probability of that object in the image
                (output probablitiy).
            valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
                image. Some objects are labeled as ambiguous.
        Returns:
            AP (list): average precision for all classes
        """
        nclasses = gt.shape[1]
        AP = []
        # pred_cls = pred_cls[pred_cls > 0]
        # print(gt.shape)
        gt[gt > 0] = 1
        for cid in range(nclasses):
            gt_cls = gt[:, cid].astype('float32')
            pred_cls = pred[:, cid].astype('float32')
            # As per PhilK. code:
            # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
            pred_cls -= 1e-5 * gt_cls
            # print(pred_cls)
            # print(gt_cls)
            # if (np.count_nonzero(gt_cls) > 0):

            ap = sklearn.metrics.average_precision_score(
                gt_cls, pred_cls, average=average)
            AP.append(ap)
        # print(AP)
        # return AP
        return np.nanmean(AP), AP



    def plot_comparisons(self, mode, batch_target_scores, batch_pred_scores, batch_input_imgs, batch_masked_input_img, batch_input_clouds, epoch, i):
        
        fig, (axs) = plt.subplots(2, 4)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        tboard_tag = "{}_images_epoch_{}/iteration_{}".format(mode, epoch, i)
        topk = 3

        input_img = batch_input_imgs[0, :, :]
        masked_input_img = batch_masked_input_img[0, :, :]
        
        # Plot point clouds
        input_cloud = batch_input_clouds[0, :, :]
        input_cloud = np.expand_dims(input_cloud, axis=0)
        input_cloud_color = np.zeros((input_cloud.shape[0], 3), dtype=int)
        input_cloud_color = np.expand_dims(input_cloud_color, axis=0)
        # self.tboard.add_mesh(
        #     "images_epoch_{}_iteration_{}_{}/input_cloud".format(epoch, i, 0), 
        #     vertices=input_cloud, 
        #     colors=input_cloud_color
        # )
        
        # Point image RGB inputs
        input_img = np.transpose(input_img, (1, 2, 0))
        # print(input_img.shape)
        masked_input_img = np.transpose(masked_input_img, (1, 2, 0))
        axs[0,0].imshow(input_img)
        axs[0,0].axis('off')
        axs[1,0].imshow(masked_input_img)
        axs[1,0].axis('off')

        # Render topk predictions and target
        target_rgb_dls = self.render_scores(
                            "train",
                            batch_target_scores, 
                            topk
                        )
        pred_rgb_dls = self.render_scores(
                            "train",
                            batch_pred_scores, 
                            topk
                        )
        for i in range(topk):
            axs[0, 1+i].imshow(target_rgb_dls[i])
            axs[0, 1+i].axis('off')
            axs[1, 1+i].imshow(pred_rgb_dls[i])
            axs[1, 1+i].axis('off')
        
        # Plot scores
        # axs[0, 1].plot(batch_target_scores[0, :])
        # axs[1, 1].plot(batch_pred_scores[0, :])

        self.tboard.add_figure(tboard_tag, fig, 0)
        plt.close(fig)


    def render_scores(self, mode, batch_pose_scores, topk):
        pose_scores = batch_pose_scores[0, :]
        if mode == "train":
            topk_rgbs, topk_depths = self.dataset.render_poses(pose_scores, topk)
        elif mode == "test":
            topk_rgbs, topk_depths = self.test_dataset.render_poses(pose_scores, topk)

        # rgb_dls = np.zeros((topk, self.dataset.img_height, self.dataset.img_width, 3))
        rgb_dls = []
        for p_i in range(len(topk_rgbs)):
            rgb_dl = topk_rgbs[p_i]
            rgb_dl = cv2.cvtColor(rgb_dl, cv2.COLOR_BGR2RGB)
            # rgb_dls[p_i, :, :, :] = rgb_dl
            rgb_dls.append(rgb_dl)

        return rgb_dls        
        
    def plot_poses(self, mode, batch_pose_scores, batch_input_imgs, batch_masked_input_img, batch_input_clouds, epoch, i, prefix):
        # print(batch_input_imgs.shape)
        # print(batch_input_clouds.shape)
        tboard_tag = "{}_images_epoch_{}/iteration_{}".format(mode, epoch, i)
        topk = 3
        pose_scores = batch_pose_scores[0, :]
        input_img = batch_input_imgs[0, :, :]
        masked_input_img = batch_masked_input_img[0, :, :]
        # print(batch_input_clouds.shape)
        input_cloud = batch_input_clouds[0, :, :]
        input_cloud = np.expand_dims(input_cloud, axis=0)
        # print(input_cloud.shape)
        input_cloud_color = np.zeros((input_cloud.shape[0], 3), dtype=int)
        input_cloud_color = np.expand_dims(input_cloud_color, axis=0)
        # print(input_img.shape)
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # self.tboard.add_mesh(
        #     "images_epoch_{}_iteration_{}_{}/input_cloud".format(epoch, i, 0), 
        #     vertices=input_cloud, 
        #     colors=input_cloud_color
        # )

        # input_img = np.transpose(input_img, (2, 0, 1))
        # print(input_img.shape)
        self.tboard.add_image("{}_input_img".format(tboard_tag), input_img)
        # self.tboard.add_image("{}_input_img".format(tboard_tag), masked_input_img)
        if mode == "train":
            topk_rgbs, topk_depths = self.dataset.render_poses(pose_scores, topk)
        elif mode == "test":
            topk_rgbs, topk_depths = self.test_dataset.render_poses(pose_scores, topk)

        rgb_dls = np.zeros((topk, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
        # print(rgb_dls.shape)
        for p_i in range(len(topk_rgbs)):
            rgb_dl = topk_rgbs[p_i]
            rgb_dl = cv2.cvtColor(rgb_dl, cv2.COLOR_BGR2RGB)
            rgb_dl = np.transpose(rgb_dl, (2, 0, 1))
            # print(rgb_dl.shape)
            rgb_dls[p_i, :, :, :] = rgb_dl
            # print(rgb_dl.shape)
        
        self.tboard.add_images("{}_{}_topk_pose".format(tboard_tag, prefix), rgb_dls)
        
        fig = plt.figure()
        plt.plot(pose_scores)
        self.tboard.add_figure("{}_{}_pose_scores".format(tboard_tag, prefix), fig, 0)
        plt.close()
    
    def test(self, epoch):
        # self.classifier.eval()

        self.estimator.eval()
        # criterion = nn.KLDivLoss(reduction="batchmean")
        criterion = nn.BCEWithLogitsLoss()
        targets_nonzero_all = []
        preds_nonzero_all = []
        num_batch = len(self.testdataloader)
        test_avg_loss = 0.0

        targets_all = []
        preds_all = []
        for i, data in enumerate(self.testdataloader, 0):
            # if i > 10:
            #     break
            # if i % 60 != 0:
            #     continue
            # imgs, points_orig, img_masked, choose, target = data
            # points = points_orig.transpose(2, 1)
            # imgs, points, img_masked, choose, target = \
            #     imgs.cuda(), points.cuda(), img_masked.cuda(), choose.cuda(), target.cuda()
            
            imgs, points_orig, img_masked_orig, img_masked, choose, target = data
            points = points_orig
            imgs, points, img_masked_orig, img_masked, choose, target = \
                imgs.cuda(), points.cuda(), img_masked_orig.cuda(), img_masked.cuda(), choose.cuda(), target.cuda()

            # pred, trans, trans_feat = self.classifier(points)
            pred = self.estimator(img_masked, points, choose)

            loss = criterion(pred, target)

            targets_all.append(target.detach().cpu().numpy().flatten().tolist())

            # pred_prob = torch.exp(pred)
            pred_prob = torch.sigmoid(pred)
            preds_all.append(pred_prob.detach().cpu().numpy().flatten().tolist())

            # print(pred_prob)
            # print(target)
            # target_nonzero = target[target > 0].detach()
            # pred_nonzero = pred[target > 0].detach()

            # target_nonzero = target[target > 0].detach()
            # pred_nonzero = pred[target > 0].detach()

            # pred_prob_nonzero = torch.exp(pred_nonzero)
            # l2_pred = torch.norm(pred_prob_nonzero - target_nonzero, 2, -1)

            test_avg_loss += loss.item()

            # print('[%d: %d/%d] test loss: %f non-zero l2 error: %f' % (epoch, i, num_batch, loss.item(), l2_pred))
            print('[%d: %d/%d] test loss: %f ' % (epoch, i, num_batch, loss.item()))
            counter = epoch * len(self.testdataloader) + i
            self.tboard.add_scalar('test/loss', loss.item(), counter)

            if self.opt.render_poses and i % 6 == 0:
                self.plot_comparisons(
                    "test",
                    target.detach().cpu().numpy(), 
                    pred.detach().cpu().numpy(), 
                    imgs.detach().cpu().numpy(),
                    img_masked_orig.detach().cpu().numpy(),
                    points_orig.detach().cpu().numpy(),
                    epoch,
                    i,
                )

        test_avg_loss /= len(self.testdataloader)
        self.tboard.add_scalar('test/average_loss', test_avg_loss, epoch)

        targets_all = np.array(targets_all)
        preds_all = np.array(preds_all)
        # print(targets_all)
        # print(preds_all)
        mean_ap, _ = self.compute_multilabel_ap(
                        targets_all, 
                        preds_all
                    )
        print("test mean ap : {}".format(mean_ap))
        self.tboard.add_scalar('test/mean_ap', mean_ap, counter)  
        # preds_nonzero_all = torch.FloatTensor(preds_nonzero_all)
        # preds_nonzero_probs_all = torch.exp(preds_nonzero_all)
        # targets_nonzero_all = torch.FloatTensor(targets_nonzero_all)

        # loss = criterion(preds_nonzero_all, targets_nonzero_all)
        # l2_pred = torch.norm(preds_nonzero_probs_all - targets_nonzero_all, 2, -1)
        # print('[%d] test loss: %f non-zero l2 error: %f' % (epoch, loss.item(), l2_pred))

        # self.classifier.train()

    def train(self):

        # self.estimator.cuda()
        self.estimator.train()

        # optimizer = optim.Adam(self.classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
        optimizer = optim.Adam(self.estimator.parameters(), lr=0.0001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        # criterion = nn.KLDivLoss(reduction="batchmean")
        criterion = nn.BCEWithLogitsLoss()
        # self.classifier.cuda()

        num_batch = len(self.dataset) / opt.batchsize

        for epoch in range(self.opt.nepoch):
            # Test after every epoch
            # self.test(epoch)

            self.estimator.train()
            optimizer.zero_grad()
    
            for i, data in enumerate(self.dataloader, 0):

                imgs, points_orig, img_masked_orig, img_masked, choose, target = data

                # points = points_orig.transpose(2, 1)
                points = points_orig
                imgs, points, img_masked_orig, img_masked, choose, target = \
                    imgs.cuda(), points.cuda(), img_masked_orig.cuda(), img_masked.cuda(), choose.cuda(), target.cuda()
                # optimizer.zero_grad()
                
                # self.classifier = self.classifier.train()
                # pred, trans, trans_feat = self.classifier(points)
                # target[target > 0] = 1
                pred = self.estimator(img_masked, points, choose)
                loss = criterion(pred, target)
                loss.backward()

                
                if i % 8 == 0:
                    # Accumulate gradients to some batchsize before taking gradient
                    optimizer.step()
                    optimizer.zero_grad()

                # optimizer.step()
                # pred_prob = torch.exp(pred)
                pred_prob = torch.sigmoid(pred)

                print(pred_prob)
                print(target)

                # target_nonzero = target[target > 0].detach()
                # pred_nonzero = pred[target > 0].detach()

                # pred_prob_nonzero = torch.exp(pred_nonzero)
                # l2_pred = torch.norm(pred_prob_nonzero - target_nonzero, 2, -1)

                counter = epoch * len(self.dataloader) + i
                self.tboard.add_scalar('train/loss', loss.item(), counter)
                
                # print('[%d: %d/%d] train loss: %f, non-zero l2 error: %f' % (epoch, i, num_batch, loss.item(), l2_pred))
                print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

                if i % 35 == 0:

                    if self.opt.render_poses:
                        self.plot_comparisons(
                            "train",
                            target.detach().cpu().numpy(), 
                            pred_prob.detach().cpu().numpy(), 
                            imgs.detach().cpu().numpy(),
                            img_masked_orig.detach().cpu().numpy(),
                            points_orig.detach().cpu().numpy(),
                            epoch,
                            i,
                        )
                            
            # Save model
            if epoch % 10 == 0:
                # torch.save(self.classifier.state_dict(), '%s/cls_model_%d.pth' % (self.opt.outf, epoch))
                torch.save(self.estimator.state_dict(), '%s/cls_model_%d.pth' % (self.opt.outf, epoch))
                # torch.save({'state_dict': self.fcn_model.state_dict()}, model_path)
        
if __name__ == "__main__":

    opt = parse_arguments()
    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    model_interface = ModelInterface(opt)

    if opt.test_only:
        model_interface.test(0)        
    else:
        model_interface.train()

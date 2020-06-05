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
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_annotation', type=str, required=True, help="dataset coco file path")
    parser.add_argument('--test_dataset_annotation', type=str, required=False, help="test dataset coco file path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--render_poses', action='store_true', help="use pose rendering for viz")

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

        try:
            os.makedirs(self.opt.outf)
        except OSError:
            pass

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
        self.classifier.eval()

        self.estimator.eval()
        criterion = nn.KLDivLoss(reduction="batchmean")
        targets_nonzero_all = []
        preds_nonzero_all = []
        num_batch = len(self.testdataloader)
        test_avg_loss = 0.0
        for i, data in enumerate(self.testdataloader, 0):
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
            pred_prob = torch.exp(pred)

            target_nonzero = target[target > 0].detach()
            pred_nonzero = pred[target > 0].detach()

            target_nonzero = target[target > 0].detach()
            pred_nonzero = pred[target > 0].detach()

            pred_prob_nonzero = torch.exp(pred_nonzero)
            l2_pred = torch.norm(pred_prob_nonzero - target_nonzero, 2, -1)
            test_avg_loss += loss.item()

            print('[%d: %d/%d] test loss: %f non-zero l2 error: %f' % (epoch, i, num_batch, loss.item(), l2_pred))
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
                # self.plot_poses(
                #     "test",
                #     target.detach().cpu().numpy(), 
                #     imgs.detach().cpu().numpy(),
                #     points_orig.detach().cpu().numpy(),
                #     epoch,
                #     i,
                #     "target"
                # )
                # self.plot_poses(
                #     "test",
                #     pred_prob.detach().cpu().numpy(), 
                #     imgs.detach().cpu().numpy(),
                #     points_orig.detach().cpu().numpy(),
                #     epoch,
                #     i,
                #     "pred_prob"
                # )
            # targets_nonzero_all += target_nonzero.tolist()
            # preds_nonzero_all += pred_nonzero.tolist()
        test_avg_loss /= len(self.testdataloader)
        self.tboard.add_scalar('test/average_loss', test_avg_loss, epoch)
        # preds_nonzero_all = torch.FloatTensor(preds_nonzero_all)
        # preds_nonzero_probs_all = torch.exp(preds_nonzero_all)
        # targets_nonzero_all = torch.FloatTensor(targets_nonzero_all)

        # loss = criterion(preds_nonzero_all, targets_nonzero_all)
        # l2_pred = torch.norm(preds_nonzero_probs_all - targets_nonzero_all, 2, -1)
        # print('[%d] test loss: %f non-zero l2 error: %f' % (epoch, loss.item(), l2_pred))

        self.classifier.train()

    def train(self):
        self.classifier = PointNetCls(k=self.dataset.num_pose_samples, feature_transform=self.opt.feature_transform)
        self.estimator = PoseNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects, num_classes = self.dataset.num_pose_samples)
        print(self.estimator)
        self.estimator.cuda()
        self.estimator.train()

        if self.opt.model != '':
            self.classifier.load_state_dict(torch.load(self.opt.model))

        # optimizer = optim.Adam(self.classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
        optimizer = optim.Adam(self.estimator.parameters(), lr=0.0001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = nn.KLDivLoss(reduction="none")
        self.classifier.cuda()

        num_batch = len(self.dataset) / opt.batchsize

        for epoch in range(self.opt.nepoch):
            # Test after every epoch
            self.test(epoch)

            optimizer.zero_grad()
            self.estimator.train()
    
            for i, data in enumerate(self.dataloader, 0):

                imgs, points_orig, img_masked_orig, img_masked, choose, target = data
                # print(target[0, :])
                # target = target[:, 0]
                # points = points_orig.transpose(2, 1)
                points = points_orig
                # print(points.shape)
                imgs, points, img_masked_orig, img_masked, choose, target = \
                    imgs.cuda(), points.cuda(), img_masked_orig.cuda(), img_masked.cuda(), choose.cuda(), target.cuda()
                optimizer.zero_grad()
                
                # self.classifier = self.classifier.train()
                # pred, trans, trans_feat = self.classifier(points)

                pred = self.estimator(img_masked, points, choose)
                # loss = F.nll_loss(pred, target)
                # print(target.shape)
                # print(pred.shape)
                loss = criterion(pred, target)
                # if opt.feature_transform:
                #     loss += feature_transform_regularizer(trans_feat) * 0.001
                loss.backward()
                optimizer.step()

                pred_prob = torch.exp(pred)
                # print(l2)
                # print(pred[0, :])
                # print(target[0, :])
                # print(pred_prob[0, :])
                # target_nonzero_ind = torch.nonzero(target)
                # print(target_nonzero_ind.shape)
                # target_nonzero = target[target_nonzero_ind]
                target_nonzero = target[target > 0].detach()
                pred_nonzero = pred[target > 0].detach()

                pred_prob_nonzero = torch.exp(pred_nonzero)
                # print(pred_prob)
                # print(target_nonzero)
                l2_pred = torch.norm(pred_prob_nonzero - target_nonzero, 2, -1)
                # l2_pred = torch.sum(l2)

                # print(target_nonzero[0, :])
                # print(pred_prob[0, :])
                # pred_choice = pred.data.max(1)[1]
                # correct = pred_choice.eq(target.data).cpu().sum()
                # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), l2_pred / float(opt.batchsize)))
                print('[%d: %d/%d] train loss: %f non-zero l2 error: %f' % (epoch, i, num_batch, loss.item(), l2_pred))
                counter = epoch * len(self.dataloader) + i
                self.tboard.add_scalar('train/loss', loss.item(), counter)
                # print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

                if i % 35 == 0:
                    if self.opt.render_poses:
                        self.plot_comparisons(
                            "train",
                            target.detach().cpu().numpy(), 
                            pred.detach().cpu().numpy(), 
                            imgs.detach().cpu().numpy(),
                            img_masked_orig.detach().cpu().numpy(),
                            points_orig.detach().cpu().numpy(),
                            epoch,
                            i,
                        )
                        # self.plot_poses(
                        #     "train",
                        #     target.detach().cpu().numpy(), 
                        #     imgs.detach().cpu().numpy(),
                        #     img_masked.detach().cpu().numpy(),
                        #     points_orig.detach().cpu().numpy(),
                        #     epoch,
                        #     i,
                        #     "target"
                        # )
                        # self.plot_poses(
                        #     "train",
                        #     pred_prob.detach().cpu().numpy(), 
                        #     imgs.detach().cpu().numpy(),
                        #     img_masked.detach().cpu().numpy(),
                        #     points_orig.detach().cpu().numpy(),
                        #     epoch,
                        #     i,
                        #     "pred_prob"
                        # )
                            


            # Save model
            if epoch % 25 == 0:
                torch.save(self.classifier.state_dict(), '%s/cls_model_%d.pth' % (self.opt.outf, epoch))
        
        

# total_correct = 0
# total_testset = 0
# for i,data in tqdm(enumerate(testdataloader, 0)):
#     points, target = data
#     target = target[:, 0]
#     points = points.transpose(2, 1)
#     points, target = points.cuda(), target.cuda()
#     classifier = classifier.eval()
#     pred, _, _ = classifier(points)
#     pred_choice = pred.data.max(1)[1]
#     correct = pred_choice.eq(target.data).cpu().sum()
#     total_correct += correct.item()
#     total_testset += points.size()[0]

# print("final accuracy {}".format(total_correct / float(total_testset)))

if __name__ == "__main__":

    opt = parse_arguments()
    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    model_interface = ModelInterface(opt)
    model_interface.train()

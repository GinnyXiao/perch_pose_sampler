from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, PoseDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import cv2 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchsize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_annotation', type=str, required=True, help="dataset coco file path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)




if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)   
elif opt.dataset_type == "ycb":
    opt.num_points = 2000 #number of points on the input pointcloud
    opt.outf = 'trained_models/ycb' #folder to save trained models
    opt.log_dir = 'experiments/logs/ycb' #folder to save logs
    opt.noise_trans = 0.03
    opt.feature_transform = True
    dataset = PoseDataset(
                'train', 
                opt.num_points, 
                True, 
                opt.dataset, 
                opt.dataset_annotation,
                opt.noise_trans, 
                False)
    test_dataset = PoseDataset(
                    'test', 
                    opt.num_points, 
                    False, 
                    opt.dataset, 
                    opt.dataset_annotation,
                    0.0, 
                    False)
    print('num pose samples', dataset.num_pose_samples)

else:
    exit('wrong dataset type')

tboard = SummaryWriter(opt.log_dir)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', dataset.classes)

def plot_poses(batch_pose_scores, batch_input_imgs, batch_input_clouds, epoch, i, prefix):
    topk = 3
    pose_scores = batch_pose_scores[0, :]
    input_img = batch_input_imgs[0, :, :]
    # print(batch_input_clouds.shape)
    input_cloud = batch_input_clouds[0, :, :]
    input_cloud = np.expand_dims(input_cloud, axis=0)
    # print(input_cloud.shape)
    input_cloud_color = np.zeros((input_cloud.shape[0], 3), dtype=int)
    input_cloud_color = np.expand_dims(input_cloud_color, axis=0)
    # print(input_img.shape)
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # tboard.add_mesh(
    #     "images_epoch_{}_iteration_{}_{}/input_cloud".format(epoch, i, 0), 
    #     vertices=input_cloud, 
    #     colors=input_cloud_color
    # )

    input_img = np.transpose(input_img, (2, 0, 1))
    # print(input_img.shape)
    tboard.add_image("images_epoch_{}_iteration_{}_{}/input_img".format(epoch, i, 0), input_img)
    topk_rgbs, topk_depths = dataset.render_poses(pose_scores, topk)
    rgb_dls = np.zeros((topk, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
    # print(rgb_dls.shape)
    for p_i in range(len(topk_rgbs)):
        rgb_dl = topk_rgbs[p_i]
        rgb_dl = cv2.cvtColor(rgb_dl, cv2.COLOR_BGR2RGB)
        rgb_dl = np.transpose(rgb_dl, (2, 0, 1))
        # print(rgb_dl.shape)
        rgb_dls[p_i, :, :, :] = rgb_dl
        # print(rgb_dl.shape)
    
    tboard.add_images("images_epoch_{}_iteration_{}_{}/{}_{}_topk_pose".format(epoch, i, 0, prefix, p_i), rgb_dls)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=dataset.num_pose_samples, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.KLDivLoss(reduction="batchmean")
classifier.cuda()

num_batch = len(dataset) / opt.batchsize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        imgs, points_orig, target = data
        # print(target[0, :])
        # target = target[:, 0]
        points = points_orig.transpose(2, 1)
        # print(points.shape)
        imgs, points, target = imgs.cuda(), points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        # loss = F.nll_loss(pred, target)
        # print(target.shape)
        # print(pred.shape)
        loss = criterion(pred, target)
        # if opt.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        # print(l2)
        # print(pred[0, :])
        # print(target[0, :])
        # print(pred_prob[0, :])
        # target_nonzero_ind = torch.nonzero(target)
        # print(target_nonzero_ind.shape)
        # target_nonzero = target[target_nonzero_ind]
        target_nonzero = target[target > 0].detach()
        pred_nonzero = pred[target > 0].detach()

        pred_prob = torch.exp(pred_nonzero)
        # print(pred_prob)
        # print(target_nonzero)
        l2_pred = torch.norm(pred_prob - target_nonzero, 2, -1)
        # l2_pred = torch.sum(l2)

        # print(target_nonzero[0, :])
        # print(pred_prob[0, :])
        # pred_choice = pred.data.max(1)[1]
        # correct = pred_choice.eq(target.data).cpu().sum()
        # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), l2_pred / float(opt.batchsize)))
        print('[%d: %d/%d] train loss: %f non-zero l2 error: %f' % (epoch, i, num_batch, loss.item(), l2_pred))
        counter = epoch * len(dataloader) + i
        tboard.add_scalar('train/loss', loss.item(), counter)
        # print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        if i % 35 == 0:
            plot_poses(
                target.detach().cpu().numpy(), 
                imgs.detach().cpu().numpy(),
                points_orig.detach().cpu().numpy(),
                epoch,
                i,
                "target"
            )
            plot_poses(
                pred.detach().cpu().numpy(), 
                imgs.detach().cpu().numpy(),
                points_orig.detach().cpu().numpy(),
                epoch,
                i,
                "pred"
            )
            # batch_pose_scores = target.detach().cpu().numpy()
            # input_img = imgs[0, :, :].detach().cpu().numpy()
            # # print(input_img.shape)
            # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            # input_img = np.transpose(input_img, (2, 0, 1))
            # # input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
            # # input_img = cv2.flip(input_img, 1)
            # print(input_img.shape)
            # tboard.add_image("images_epoch_{}_iteration_{}_{}/input_img".format(epoch, i, 0), input_img)
            # topk_rgbs, topk_depths = dataset.render_poses(batch_pose_scores[0, :], 3)
            # for p_i in range(len(topk_rgbs)):
            #     rgb_dl = topk_rgbs[p_i]
            #     rgb_dl = cv2.cvtColor(rgb_dl, cv2.COLOR_BGR2RGB)
            #     rgb_dl = np.transpose(rgb_dl, (2, 0, 1))
            #     # rgb_dl = cv2.rotate(rgb_dl, cv2.ROTATE_90_CLOCKWISE)
            #     # rgb_dl = cv2.flip(rgb_dl, 1)
            #     print(rgb_dl.shape)
            #     tboard.add_image("images_epoch_{}_iteration_{}_{}/{}_topk_pose".format(epoch, i, 0, p_i), rgb_dl)

        #     j, data = next(enumerate(testdataloader, 0))
        #     points, target = data
        #     target = target[:, 0]
        #     points = points.transpose(2, 1)
        #     points, target = points.cuda(), target.cuda()
        #     classifier = classifier.eval()
        #     pred, _, _ = classifier(points)
        #     loss = criterion(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

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

print("final accuracy {}".format(total_correct / float(total_testset)))
from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import torch.nn.functional as F
from convert_fat_coco import *
from fat_pose_image import FATImage
import transformations 

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, coco_annotation_file, noise_trans, refine):
        if mode == 'train':
            # self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
            self.path = os.path.join(root, "image_sets/train.txt")
        elif mode == 'test':
            # self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
            self.path = os.path.join(root, "image_sets/keyframe.txt")
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        
        self.coco = COCO(coco_annotation_file)
        self.coco_category_ids = self.coco.getCatIds(catNms=['square'])
        self.coco_image_ids = self.coco.getImgIds(catIds=self.coco_category_ids)
        self.length = len(self.coco_image_ids)
        self.viewpoints_xyz = np.array(self.coco.dataset['viewpoints'])
        self.inplane_rotations = self.coco.dataset['inplane_rotations']

        coco_categories = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [category['name'] for category in coco_categories]
        # self.num_pose_samples = len(self.coco.dataset['viewpoints']) * len(self.coco.dataset['inplane_rotations'])
        self.num_pose_samples = len(self.coco.dataset['viewpoints'])

        self.depth_factor = 10000
        self.fat_image = FATImage(
            coco_annotation_file=coco_annotation_file,
            coco_image_directory=self.root,
            depth_factor=self.depth_factor,
            model_dir=os.path.join(self.root, "models"),
            model_mesh_in_mm=False,
            model_mesh_scaling_factor=1,
            models_flipped=False,
            img_width=640,
            img_height=480,
            distance_scale=1
        )
        # self.fat_image.render_machines = {}
        # self.fat_image.render_machines["004_sugar_box"] = fat_image.get_renderer("004_sugar_box")

        # self.list = []
        # self.real = []
        # self.syn = []
        # input_file = open(self.path)
        # while 1:
        #     input_line = input_file.readline()
        #     if not input_line:
        #         break
        #     if input_line[-1:] == '\n':
        #         input_line = input_line[:-1]
        #     if input_line[:5] == 'data/':
        #         self.real.append(input_line)
        #     else:
        #         self.syn.append(input_line)
        #     self.list.append(input_line)
        # input_file.close()

        # self.length = len(self.list)
        # self.len_real = len(self.real)
        # self.len_syn = len(self.syn)

        class_file = open(os.path.join(root, "image_sets/classes.txt"))

        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()
            
            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189
        self.img_width = 640
        self.img_height = 480

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        # self.refine = refine
        self.front_num = 2
        self.fixed_translation = [0, 0, 0.75]

        print(self.length)

    def get_depth_img_path(self, color_img_path):
        return color_img_path.replace('color', 'depth')

    def __getitem__(self, index):
        coco_image_data = self.coco.loadImgs(self.coco_image_ids[index])[0]
        color_img_path = '{0}/{1}'.format(self.root, coco_image_data['file_name'])
        img = Image.open(color_img_path)
        depth = np.array(Image.open(self.get_depth_img_path(color_img_path)))

        annotation_ids = self.coco.getAnnIds(imgIds=coco_image_data['id'], catIds=self.coco.getCatIds(), iscrowd=None)
        
        # Pick 0th object's annotation
        coco_annotation = self.coco.loadAnns(annotation_ids)[0]
        pose_scores = np.array(coco_annotation['pose_scores'])
        # print(pose_scores.shape)
        # pose_score_probs = pose_scores/np.sum(pose_scores)
        pose_score_probs = pose_scores
        xmin, ymin, width, height = coco_annotation['bbox']
        # rmin, rmax, cmin, cmax = ymin, ymin + height, xmin, xmin + width
        # print(coco_annotation) 

        mask = self.coco.annToMask(coco_annotation)

        # Need visible bounding box
        rmin, rmax, cmin, cmax = get_bbox(mask)

        # print(coco_image_data['file_name'])
        scene_id = int(coco_image_data['file_name'].split("/")[1])
        # print(scene_id)
        if scene_id >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        # mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        # add_front = False
        # if self.add_noise:
        #     for k in range(5):
        #         seed = random.choice(self.syn)
        #         front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
        #         front = np.transpose(front, (2, 0, 1))
        #         f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
        #         front_label = np.unique(f_label).tolist()[1:]
        #         if len(front_label) < self.front_num:
        #            continue
        #         front_label = random.sample(front_label, self.front_num)
        #         for f_i in front_label:
        #             mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
        #             if f_i == front_label[0]:
        #                 mask_front = mk
        #             else:
        #                 mask_front = mask_front * mk
        #         t_label = label * mask_front
        #         if len(t_label.nonzero()[0]) > 1000:
        #             label = t_label
        #             add_front = True
        #             break

        # obj = meta['cls_indexes'].flatten().astype(np.int32)

        # while 1:
        #     idx = np.random.randint(0, len(obj))
        #     mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        #     mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
        #     mask = mask_label * mask_depth
        #     if len(mask.nonzero()[0]) > self.minimum_num_pt:
        #         break

        # if self.add_noise:
        #     img = self.trancolor(img)

        # rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img_masked = np.copy(np.array(img))
        img_masked = np.transpose(img_masked[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))
        # if self.list[index][:8] == 'data_syn':
        #     seed = random.choice(self.real)
        #     back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
        #     back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        #     img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        # else:
        #     img_masked = img

        # if self.add_noise and add_front:
        #     img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        # if self.list[index][:8] == 'data_syn':
        #     img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        # p_img = np.transpose(img_masked, (1, 2, 0))
        # scipy.misc.imsave('temp/{0}_input.png'.format(index), p_img)
        # scipy.misc.imsave('temp/{0}_label.png'.format(index), mask[rmin:rmax, cmin:cmax].astype(np.int32))

        # target_r = meta['poses'][:, :, idx][:, 0:3]
        # target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        # add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])
        # print(choose.shape)
        # print(img_masked.shape)
        # cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / self.depth_factor
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        cloud = transform_cloud(cloud, trans=np.array(self.fixed_translation), quat=np.array([1, 0, 0, 0]))
        return torch.from_numpy(img), \
               torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(img_masked), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.LongTensor(choose.astype(np.int32)), \
               torch.from_numpy(pose_score_probs.astype(np.float32))
        #    F.softmax(torch.FloatTensor(pose_scores))
        # if self.add_noise:
        #     cloud = np.add(cloud, add_t)

        # fw = open('temp/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        # if self.refine:
        #     dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        # else:
        # dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        # model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        # fw = open('temp/{0}_model_points.xyz'.format(index), 'w')
        # for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # target = np.dot(model_points, target_r.T)
        # if self.add_noise:
        #     target = np.add(target, target_t + add_t)
        # else:
        #     target = np.add(target, target_t)
        
        # fw = open('temp/{0}_tar.xyz'.format(index), 'w')
        # for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()
        
        # return torch.from_numpy(cloud.astype(np.float32)), \
        #        torch.LongTensor(choose.astype(np.int32)), \
        #        self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
        #        torch.from_numpy(target.astype(np.float32)), \
        #        torch.from_numpy(model_points.astype(np.float32)), \
        #        torch.LongTensor([int(obj[idx]) - 1])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def render_poses(self, pose_scores, k):
        # pose_scores = pose_scores[0, :]
        # pose_scores = pose_scores.reshape((80, 5))
        topk_ii = np.unravel_index(np.argsort(pose_scores.ravel())[-k:], pose_scores.shape)
        topk_pose_scores = pose_scores[topk_ii]
        # print(topk_ii)
        topk_rgbs = []
        topk_depths = []
        for i in range(k):
            viewpoint_id = topk_ii[0][i]
            # inplane_rotation_id = topk_ii[1][i]
            theta, phi = \
                get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
            inplane_rotation_angle = 0
            # inplane_rotation_angle = \
            #     get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
            xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
            # print("Recovered rotation : {}".format(xyz_rotation_angles))
            rgb_gl, depth_gl = self.fat_image.render_pose(
                "004_sugar_box", xyz_rotation_angles, self.fixed_translation
            )
            topk_rgbs.append(rgb_gl)
            topk_depths.append(depth_gl)

        return topk_rgbs, topk_depths

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    # print(np.count_nonzero(label))
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def transform_cloud(cloud_in, trans=None, quat=None, mat=None):
    '''
        Tranform point cloud np array
    '''
    if trans is not None and quat is not None:
        R = transformations.quaternion_matrix(quat)
        T = transformations.translation_matrix(trans)
        total_transform = transformations.concatenate_matrices(T, R)
    elif mat is not None:
        total_transform = mat
    cloud_in = np.hstack((cloud_in, np.ones((cloud_in.shape[0], 1))))
    cloud_out = np.matmul(total_transform, np.transpose(cloud_in))
    cloud_out = np.transpose(cloud_out)[:,:3]
    cloud_out = np.array(cloud_out, dtype=np.float32)
    return cloud_out

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])


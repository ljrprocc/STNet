import torch
from torch.utils.data import Dataset
from utils.image_utils import *
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import cv2
import multiprocessing

def get_list(dir, gt=False):
    ori_list = list(os.listdir(dir))
    res = []
    mode = '.txt' if gt else '.JPG'
    for file in ori_list:
        if os.path.splitext(file)[-1] == mode:
            res.append(file)
    res.sort()
    # if gt:
    #     print(res)
    return res


def get_imsize_bboxes(img_size, gt_path):
    h, w = img_size
        # print(h, w)
    lines = open(gt_path).read().splitlines()
    bboxes = []
    tags = []
    # print(gt_path)
    for line in lines:
        line = line.strip('\xef\xbb\xbf\ufeff\n')
        gt = line.split(',')
            # print(gt)
        if len(gt) ==8 or gt[-1][0] == '#':
            tags.append(False)
        else:
            tags.append(True)
        box = [eval(gt[i]) for i in range(8)]
        box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(box)
    return np.array(bboxes), tags

class IC15Loader(Dataset):
    def __init__(self, data_root, train=True, patch_size=None):
        super(IC15Loader, self).__init__()
        self.root = self.init_root(images_root=data_root, train=train)
        self.train = train
        self.syn_img_paths, self.ori_img_paths, self.syn_gt_paths = self.gen_files()
        self.patch_size = patch_size

    def init_root(self, images_root, train):
        if train:
            sub = 'train'
        else:
            sub = 'val'
        if type(images_root) is not list:
            images_root = [images_root]
        return ['%s/%s' % (root, sub) for root in images_root]

    def gen_files(self):
        syn_roots = []
        ori_roots = []
        syn_gts = []
        for root in self.root:
            orilist = get_list(root)
            synlist = get_list(root + '_syn')
            syn_gtlist = get_list(root + '_syn_gt', gt=True)
            ori = [os.path.join(root, k) for k in orilist]
            syns = [os.path.join(root + '_syn', k) for k in synlist]
            syn_gt = [os.path.join(root + '_syn_gt', k) for k in syn_gtlist]
            syn_roots += syns
            ori_roots += ori
            syn_gts += syn_gt
        return syn_roots, ori_roots, syn_gts

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        ori_img_path = self.ori_img_paths[idx]
        syn_img_path = self.syn_img_paths[idx]
        syn_gt_path = self.syn_gt_paths[idx]
        # print(ori_img_path, syn_img_path, syn_gt_path)
        # exit(-1)
        ori_img = cv2.imread(ori_img_path)
        syn_img = cv2.imread(syn_img_path)
        h, w = syn_img.shape[:2]
        # print(h, w)
        c = syn_img.shape[2]
        bboxes, _ = get_imsize_bboxes((h, w), syn_gt_path)
        # if not self.train and (h >= 800 or w >= 800):
        #     # print(h, w)
        #     ori_img = cv2.resize(ori_img, None, fx=0.6, fy=0.6)
        #     syn_img = cv2.resize(syn_img, None, fx=0.6, fy=0.6)
        #     w, h = int(w * 0.6), int(h * 0.6)
            # print(h, w)
            # print(ori_img.shape)
        # print(syn_img.shape)
        mask = np.zeros((h, w), dtype=np.uint8)
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([w, h] * 4), (bboxes.shape[0], bboxes.shape[1] // 2, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(mask, [bboxes[i]], -1, 1, -1)
                # cv2.drawContours(syn_img, [bboxes[i]], -1, 0, -1)
        # debug
        # print(ori_img_path)
        # print(syn_img_path)
        # print(syn_gt_path)
        # print(bboxes)
        if self.train and (h < self.patch_size or w < self.patch_size):
            # print(ori_img_path)
            height_padding = max(self.patch_size + 1 - h, 0)
            width_padding = max(self.patch_size + 1 - w, 0)
            ori_img = cv2.copyMakeBorder(ori_img, 0, height_padding, 0, width_padding, cv2.BORDER_CONSTANT)
            syn_img = cv2.copyMakeBorder(syn_img, 0, height_padding, 0, width_padding, cv2.BORDER_CONSTANT)
            mask = cv2.copyMakeBorder(mask, 0, height_padding, 0, width_padding, cv2.BORDER_CONSTANT)
        area = np.sum(mask)
        if not self.train and (h % 16 != 0 or w % 16 != 0):
            print(ori_img_path)
            print(ori_img.shape)
        # print(area)
        # mask = np.stack([mask] * c, axis=2)
        # print(mask.shape)
        if not self.train:
            syn_img, ori_img, mask = self.add_padding(syn_img), self.add_padding(ori_img), self.add_padding(mask)
        syn_img, ori_img, mask = self.transforms(syn_img), self.transforms(ori_img), torch.from_numpy(mask)
        # print(syn_img.shape, ori_img.shape, mask.shape)
        if self.patch_size:
            syn_img, ori_img, mask =  self.crop_images(syn_img, ori_img, mask)
        # print(mask.sum())
        
        # if self.train and h < self.patch_size:
        #     print(syn_img.shape, ori_img.shape, mask.shape)
        if area == 0:
            area = 1
        return syn_img, ori_img, mask.unsqueeze(0).float(), float(area), float(h * w)

    def transforms(self, img):
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    def add_padding(self, img):
        import math
        h, w = img.shape[:2]
        if h % 16 != 0 or w % 16 != 0:
            up = math.ceil(h / 16) * 16
            right = math.ceil(w / 16) * 16
            img = cv2.copyMakeBorder(img, 0, up-h, 0, right - w, cv2.BORDER_CONSTANT)
            # hh, ww = img.shape[:2]
            # if hh % 16 != 0:
            #     raise('The height of img = ' + str(img.shape[0]) + ' % 16 = ' + str(img.shape[0] % 16))
        return img

    def crop_images(self, *images):
        w, h  = images[0].shape[1:]
        # if w <= self.patch_size + 1 or h <= self.patch_size + 1:
        #     print(w, h)
        left_most = random.randint(0, w - 1 - self.patch_size) if w > self.patch_size else 0
        top_most = random.randint(0, h - 1 - self.patch_size) if h > self.patch_size else 0
        # print(left_most, top_most)
        cropped = []
        for image in images:
            if len(image.shape) == 3:
                cropped.append(image[:, left_most: left_most + self.patch_size, top_most: top_most + self.patch_size])
            else:
                cropped.append(image[left_most: left_most + self.patch_size, top_most: top_most + self.patch_size])
        return cropped

    def __len__(self):
        return len(self.syn_img_paths)

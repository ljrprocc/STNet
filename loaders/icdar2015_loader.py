import torch
from torch.utils.data import Dataset
from utils.image_utils import *
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import cv2
import multiprocessing


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
            orilist = list(os.listdir(root))
            orilist.sort()
            synlist = list(os.listdir(root + '_syn'))
            synlist.sort()
            syn_gtlist = list(os.listdir(root + '_syn_gt'))
            syn_gtlist.sort()
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
        ori_img = cv2.imread(ori_img_path)
        syn_img = cv2.imread(syn_img_path)
        h, w = ori_img.shape[:2]
        # print(h, w)
        c = ori_img.shape[2]
        # syn_img = cv2.resize(syn_img, (w, h))
        bboxes, _ = get_imsize_bboxes((h, w), syn_gt_path)
        mask = np.zeros((h, w), dtype=np.uint8)
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([syn_img.shape[1], syn_img.shape[0]] * 4), (bboxes.shape[0], bboxes.shape[1] // 2, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(mask, [bboxes[i]], -1, 1, -1)
        # debug
        # print(ori_img_path)
        # print(syn_img_path)
        # print(syn_gt_path)
        # print(bboxes)
        area = np.sum(mask)
        # print(area)
        # mask = np.stack([mask] * c, axis=2)
        # print(mask.shape)
        syn_img, ori_img, mask = self.transforms(syn_img), self.transforms(ori_img), torch.from_numpy(mask)
        
        # print(syn_img.shape, ori_img.shape)
        if self.patch_size:
            syn_img, ori_img, mask =  self.crop_images(syn_img, ori_img, mask)
        # print(mask.sum())
        if area == 0:
            area = 1
        return syn_img, ori_img, mask.unsqueeze(0).float(), float(area)

    def transforms(self, img, mask=False):
        img = Image.fromarray(img)

        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    def crop_images(self, *images):
        size = images[0].shape[1]
        left_most = random.randint(0, size - 1 - self.patch_size)
        top_most = random.randint(0, size - 1 - self.patch_size)
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

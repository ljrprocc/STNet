import numpy as np
import os
import cv2
import sys
import math
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import multiprocessing
sys.path.append('/home/jingru.ljr/Motif-Removal')
from utils.image_utils import save_image
from utils.train_utils import *
from utils.text_utils import run_boxes
from networks.gan_model import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tqdm

device = torch.device('cuda:8')

root_path = '..'
# train_tag = 'demo_coco'
# train_tag = 'demo_msra_1'
train_tag = 'icdar_total2x_per'
0
nets_path = '%s/checkpoints/%s' % (root_path, train_tag)

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = False
use_rgb = True
dilation_depth= 0
dis_channels=64
gen_channels=48
batch_size=16
image_encoder=True

criterion = nn.MSELoss()

# datasets paths
cache_root = ['/data/jingru.ljr/icdar2015/syn_ds_root_1280_2xa/']
# cache_root = ['/data/jingru.ljr/COCO/syn_output/']
# cache_root = ['/data/jingru.ljr/MSRA-TD500/syn_ds_root/']

def visualize_feature(features, jpg_name):
    if not os.path.exists(os.path.join(feature_path, jpg_name)):
        os.mkdir(os.path.join(feature_path, jpg_name))
    for j, feature in enumerate(features):
        feature = feature[0]
        c, h, w = feature.shape
        fea = feature.cpu().numpy()
        fea_trans = np.transpose(fea, (1,2,0))
        # print('**')
        if c > 24:
            fea_trans = PCA(n_components=24).fit_transform(fea_trans.reshape(-1, c))
        if c >= 3:
            if c > 3:
                fea_trans = TSNE(n_components=3, n_iter=1000, verbose=1).fit_transform(fea_trans)
            fea_trans = 1.0 / (1 + np.exp(-fea_trans))
            fea_trans = fea_trans.reshape(h, w, 3)
            fea_trans = np.round(fea_trans * 255)
            img_name = os.path.join(feature_path, jpg_name, 'level{}.jpg'.format(j))
            cv2.imwrite(img_name, fea_trans)
        # for i in range(c):
        #     fea = feature[i].cpu().numpy()
        #     fea = 1.0 / (1 + np.exp(-fea))
        #     fea = np.round(fea * 255)
        #     img_name = os.path.join(feature_path, jpg_name, 'level{}_channel{}.jpg'.format(j, i))
        #     # print(img_name)
        #     cv2.imwrite(img_name, fea)
            

def test(test_loader, model, debug=False, baseline=False):
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_loader)):
            img, ori = batch[0].to(device), batch[1].to(device)
            # print(batch[-1].data, i)
            # exit(-1)
            model.eval()
            a = time.time()
            model = model.to(device)
            features = model.get_features(img)
            jpg_name = test_loader.dataset.syn_img_paths[i].split('/')[-1]
            visualize_feature(features, jpg_name)
            
def test_single(test_loader, model, idx=0):
    with torch.no_grad():
        # print(type(test_loader.dataset[idx]))
        img, ori = test_loader.dataset[idx][0].to(device), test_loader.dataset[idx][1].to(device)
        # print(batch[-1].data, i)
        # exit(-1)
        model.eval()
        a = time.time()
        model = model.to(device)
        features = model.get_features(img.unsqueeze(0))
        jpg_name = test_loader.dataset.syn_img_paths[idx].split('/')[-1]
        visualize_feature(features, jpg_name)

def run():
    opt = load_globals(nets_path, globals(), override=True)

    base_net = init_nets(opt, nets_path, device, tag='13003x', open_image=image_encoder)
    # base_net = InpaintModel(opt, nets_path, device, tag='1500', gate=False).to(device)
    # base_net.load(1000)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)

    # test(test_loader, base_net, debug=True)
    test_single(test_loader, base_net, 11)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    write_dir = '/data/jingru.ljr/AAAI2021/result/%s'%(train_tag)
    # vis_path = os.path.join(write_dir, 'vis/')
    # res_path = os.path.join(write_dir, 'res/')
    feature_path = os.path.join(write_dir, 'write_features/')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    # if not os.path.exists(vis_path):
    #     os.mkdir(vis_path)
    # if not os.path.exists(res_path):
    #     os.mkdir(res_path)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    run()

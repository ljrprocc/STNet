import numpy as np
import os
import cv2
import sys
import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import multiprocessing
sys.path.append('/home/jingru.ljr/Motif-Removal')
from utils.image_utils import save_image
from utils.train_utils import *

device = torch.device('cuda:0')

root_path = '..'
train_tag = 'demo_coco_1'

nets_path = '%s/checkpoints/%s' % (root_path, train_tag)

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = False
use_rgb = True

criterion = nn.MSELoss()

# datasets paths
cache_root = ['/data/jingru.ljr/icdar2015/syn_ds_root_1280/']

def cal_psnr(reconstructed_images, ori):
    mse = criterion(reconstructed_images, ori)
    psnr = 10 * math.log10(1 / mse.item())
    return psnr

def test(test_loader, model, debug=False):
    avg_psnr = 0
    avg_ssim = 0
    with torch.no_grad():
        for batch in test_loader:
            img, ori = batch[0].to(device), batch[1].to(device)
            model.eval()
            model = model.to(device)
            output = model(img)
            
            guess_images, guess_mask = output[0], output[1]
            expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_guess_mask
            reconstructed_images = img * (1 - expanded_guess_mask) + reconstructed_pixels
            transformed_guess_mask = expanded_guess_mask * 2 - 1
            ssim_val = ssim(ori, reconstructed_images, data_range=255, size_average=False)
            if debug:
                images_un = torch.cat((ori, reconstructed_images), 0)
                images_un = torch.clamp(images_un.data, min=-1, max=1)
                images_un = make_grid(images_un, nrow=img.shape[0], padding=5, pad_value=1)
                save_image(images_un, '../checkpoints/test.jpg')
                print(ssim_val)
                exit(-1)
            
            psnr = cal_psnr(reconstructed_images, ori)
            avg_ssim += ssim_val.item()
            avg_psnr += psnr

    print('=====> Avg. PSNR: {:.4f} dB'.format(avg_psnr / len(test_loader)))
    print('=====> Avg. SSIM: {:.4f} '.format(avg_ssim / len(test_loader)))


def run():
    opt = load_globals(nets_path, globals(), override=True)

    base_net = init_nets(opt, nets_path, device)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)
    test(test_loader, base_net, debug=False)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run()

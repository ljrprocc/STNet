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
from utils.text_utils import run_boxes
from networks.gan_model import *
import tqdm

device = torch.device('cuda:1')

root_path = '..'
# train_tag = 'demo_coco'
# train_tag = 'demo_msra_1'
train_tag = 'icdar_total3x_a'

nets_path = '%s/checkpoints/%s' % (root_path, train_tag)

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = False
use_rgb = True
dilation_depth= 0
dis_channels=64
gen_channels=48
batch_size=16

criterion = nn.MSELoss()

# datasets paths
cache_root = ['/data/jingru.ljr/icdar2015/syn_ds_root_1280_3x/']
# cache_root = ['/data/jingru.ljr/COCO/syn_output/']
# cache_root = ['/data/jingru.ljr/MSRA-TD500/syn_ds_root/']

def cal_psnr(reconstructed_images, ori):
    mse = criterion(reconstructed_images, ori)
    psnr = 10 * math.log10(1 / (mse.item() + 1e-8))
    return psnr

def test(test_loader, model, debug=False, baseline=False):
    avg_psnr = [0., 0.]
    avg_ssim = [0., 0.]
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_loader)):
            img, ori = batch[0].to(device), batch[1].to(device)
            # print(batch[-1].data, i)
            # exit(-1)
            model.eval()
            model = model.to(device)
            output = model(img)
            jpg_name = test_loader.dataset.syn_img_paths[i].split('/')[-1]
            write_path = '%s/%s'%(vis_path, jpg_name)
            write_res_name = '%s/%s.txt'%(res_path, jpg_name[:-4])
            write_inpaint_path = '%s/%s'%(inpaint_path, jpg_name)
            
            guess_images, guess_mask = output[0], output[3]
            real_img = img.data.squeeze().cpu().numpy().copy()
            real_img = np.transpose(real_img, (1,2,0)) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            real_img = np.around(real_img * 255).astype("uint8")
            # real_img = torch.clamp(img.data, min=-1, max=1)
            # real_img = real_img.squeeze().cpu().numpy().copy() / 2 + 0.5
            # print(real_img.shape)
            # exit(-1)
            # real_img = np.transpose(real_img, (1,2,0)) * 255
            # real_img = real_img[:, :, ::-1]
            # print(real_img)
            # run_boxes(real_img, guess_mask.squeeze().cpu().numpy(), write_path, write_res_name)
            expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_guess_mask
            reconstructed_images = img * (1 - expanded_guess_mask) + reconstructed_pixels
            transformed_guess_mask = expanded_guess_mask * 2 - 1
            ssim_val = ssim(ori, reconstructed_images, data_range=255, size_average=False)
            ssim_baseline = ssim(ori, img, data_range=255, size_average=False)
            if debug:
                images_un = torch.cat((ori, img, reconstructed_images, transformed_guess_mask), 0)
                images_un = torch.clamp(images_un.data, min=-1, max=1)
                images_un = make_grid(images_un, nrow=img.shape[0], padding=5, pad_value=1)
                save_image(images_un,write_inpaint_path)
                # print(ssim_val)
                # exit(-1)
            
            psnr = cal_psnr(reconstructed_images, ori)
            psnr_baseline = cal_psnr(img, ori)
            avg_ssim[0] += ssim_val.item()
            avg_psnr[0] += psnr
            avg_ssim[1] += ssim_baseline.item()
            avg_psnr[1] += psnr_baseline
            # exit(-1)

    print('=====> Avg. PSNR: {:.4f} dB, baseline: {:.4f} dB'.format(avg_psnr[0] / len(test_loader), avg_psnr[1] / len(test_loader)))
    print('=====> Avg. SSIM: {:.6f}, baseline: {:.6f}'.format(avg_ssim[0] / len(test_loader), avg_ssim[1] / len(test_loader)))


def run():
    opt = load_globals(nets_path, globals(), override=True)

    # base_net = init_nets(opt, nets_path, device, tag='3000')
    base_net = InpaintModel(opt, nets_path, device, tag='1000', gate=False).to(device)
    base_net.load(1000)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)

    test(test_loader, base_net)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    write_dir = '/data/jingru.ljr/AAAI2021/result/%s'%(train_tag)
    vis_path = os.path.join(write_dir, 'vis/')
    res_path = os.path.join(write_dir, 'res/')
    inpaint_path = os.path.join(write_dir, 'inpaint_fine/')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    if not os.path.exists(inpaint_path):
        os.mkdir(inpaint_path)
    run()

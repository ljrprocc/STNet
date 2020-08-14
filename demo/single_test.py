import numpy as np
import os
import cv2
import sys
import math
from PIL import Image
from torchvision import transforms
sys.path.append('/home/jingru.ljr/Motif-Removal')
from utils.image_utils import save_image
from utils.train_utils import *
from utils.text_utils import *

device = torch.device('cuda:0')

root_path = '..'
train_tag = 'demo_coco'

nets_path = '%s/checkpoints/%s' % (root_path, train_tag)

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = False
use_rgb = True

def single_test(img_dir, model, image_name):
    img = cv2.imread(img_dir)
    # img = cv2.resize(img, None, fx=0.6, fy=0.6)
    h, w = img.shape[:2]
    
    if h % 16 != 0 or w % 16 != 0:
        up = math.ceil(h / 16) * 16
        right = math.ceil(w / 16) * 16
        img = cv2.copyMakeBorder(img, 0, up-h, 0, right - w, cv2.BORDER_CONSTANT)
    img = Image.fromarray(img)
    img = img.convert("RGB")
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    model.eval()
    output = model(img)
    
    guess_images, guess_mask = output[0], output[-1]
    expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
    reconstructed_pixels = guess_images * expanded_guess_mask
    reconstructed_images = img * (1 - expanded_guess_mask) + reconstructed_pixels
    transformed_guess_mask = expanded_guess_mask * 2 - 1

    images_un = torch.cat((img, reconstructed_images, transformed_guess_mask), 0)
    images_un = torch.clamp(images_un.data, min=-1, max=1)
    images_un = make_grid(images_un, nrow=img.shape[0], padding=5, pad_value=1)
    save_image(images_un, image_name)


def run():
    opt = load_globals(nets_path, globals(), override=True)

    base_net = init_nets(opt, nets_path, device, tag='30')
    single_test(img_dir, base_net, os.path.join(nets_path, img_dir.split('/')[-1]))


if __name__ == '__main__':
    # img_dir = '/data/jingru.ljr/COCO/syn_output/val_syn/000000001503.jpg'
    img_dir = '/home/jingru.ljr/checkpoints/demo_coco/test1503.jpg'
    run()



import torch
import torch.nn as nn
import os
import pickle
from loaders.cache_loader import CacheLoader
from loaders.icdar2015_loader import IC15Loader
from torch.utils.data import DataLoader
from train.train_options import TrainOptions as Opt
from networks.baselines import *
from utils.image_utils import save_image
from torchvision.utils import make_grid
from torchvision import models
import numpy as np

def resize_like(x, target, mode='bilinear'):
    return nn.functional.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)

class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)

        for para in vgg16.parameters():
            para.requires_grad = False

        self.vgg16_pool_1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_pool_2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_pool_3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self, x):
        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)

        return [pool_1, pool_2, pool_3]


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, tar, running_mask=None):
        inp = inp.contiguous().view(inp.size()[0], -1)
        tar = tar.contiguous().view(tar.size()[0], -1)
        mask = running_mask.contiguous().view(running_mask.size()[0], -1)
        # print(torch.sum(tar + mask))
        if running_mask is not None:
            inp = inp * mask
            tar = tar * mask
        # print(torch.sum(tar))
        a = torch.sum(inp * tar, 1)
        b = torch.sum(inp * inp, 1) + 0.0001
        c = torch.sum(tar * tar, 1) + 0.0001
        d = (2 * a) / (b + c)

        return 1 - torch.mean(d)


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        N, C, H, W = feature.shape
        feature = feature.view(N, C, H * W)
        gram_mat = torch.bmm(feature, torch.transpose(feature, 1, 2))
        return gram_mat / (C * H * W)

    def forward(self, results, targets):
        loss = 0.
        for i, (ress, tars) in enumerate(zip(results, targets)):
            loss += self.l1loss(self.gram(ress), self.gram(tars))
        return loss / len(results)


class PerceptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, results, targets):
        loss = 0.
        for i ,(ress, tars) in enumerate(zip(results, targets)):
            loss += self.l1loss(ress, tars)
        return loss / len(results)


class TotalVariationLoss(nn.Module):
    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img
    
        kernel = torch.FloatTensor([
            [0,1,0],
            [1,-2,0],
            [0,0,0]
        ]).view(1,1,3,3)

        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)
    
    def gradient(self, x):
        return nn.functional.conv2d(x, self.kernel, stride=1, padding=1, groups=self.c_img)
    
    def forward(self, results, mask):
        loss = 0.
        for res in results:
            grad = self.gradient(res) * resize_like(mask, res)
            loss += torch.mean(torch.abs(grad))
        return loss / len(results)


class AdversialLoss(nn.Module):
    def __init__(self, type='hinge', target_real_label=1.0, target_fake_label=0.0):
        super(AdversialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()
        else:
            raise NotImplementedError
    
    def forward(self, outputs, is_real, is_disc=False):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                # print(output.mean())
                return (-outputs).mean()
        
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    # print(score.shape, gt_text.shape, training_mask.shape)
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    # neg_score = score[gt_text <= 0.5]
    bce_score = gt_text * np.log(score + 0.0001) + (1 - gt_text) * np.log(1 - score + 0.0001)
    neg_score = bce_score[gt_text <= 0.5]
    neg_score_sorted = np.sort(neg_score)
    # print(neg_score_sorted)
    threshold = neg_score_sorted[-neg_num]

    selected_mask = ((score >= threshold)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask
def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.squeeze().data.cpu().numpy()
    gt_texts = gt_texts.squeeze().data.cpu().numpy()
    training_masks = training_masks.squeeze().data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks

def init_folders(*folders):
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)


def load_globals(nets_path, globals_dict, override=True):
    save_set = {
                'vm_tag', 'images_root', 'vm_root', 'vm_size', 'image_size', 'image_size_w', 'image_size_h', 'patch_size', 'perturbate', 'opacity_var',
                'use_rgb', 'weight', 'shared_depth', 'num_blocks', 'batch_size', 'use_vm_decoder', 'rotate_vm', 'TDBmode'
                'scale_vm', 'crop_vm', 'batch_vm', 'font', 'text_border', 'blur', 'dis_channels', 'gen_channels', 'dilation_depth'
                }
    to_save = False
    params_file = '%s/train_params.pkl' % nets_path
    __opt = Opt()
    if os.path.isfile(params_file):
        print('loading options from %s/' % nets_path)
        with open(params_file, 'rb') as f:
            save_globals_dict = pickle.load(f)
    else:
        save_globals_dict = {}
    print(save_globals_dict)
    for item in save_set:
        if item not in save_globals_dict and item in globals_dict:
            to_save = True
            save_globals_dict[item] = globals_dict[item]
        if item in save_globals_dict:
            setattr(__opt, item, save_globals_dict[item])
            print('%s: %s' % (item, str(save_globals_dict[item])))
    if to_save and override:
        with open(params_file, 'wb') as f:
            pickle.dump(save_globals_dict, f, pickle.HIGHEST_PROTOCOL)
    return __opt


def init_loaders(opt, cache_root='', ds='IC15'):
    if ds != 'IC15':
        train_dataset = CacheLoader(cache_root, train=True, patch_size=opt.patch_size)
        test_dataset = CacheLoader(cache_root, train=False, patch_size=None)
    else:
        train_dataset = IC15Loader(cache_root, train=True, patch_size=opt.patch_size)
        test_dataset = IC15Loader(cache_root, train=False, patch_size=None)
    # print(train_dataset.root)
    # print(len(train_dataset))
    _train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    if opt.patch_size:
        batch_scale_w = int(opt.image_size_w / opt.patch_size)
        batch_scale_h = int(opt.image_size_h / opt.patch_size)
        batch_scale = batch_scale_w * batch_scale_h
    else:
        batch_scale = 1
    _test_data_loader = DataLoader(test_dataset, batch_size=1,
                                   shuffle=False, num_workers=1)
    return _train_data_loader, _test_data_loader


def init_nets(opt, net_path, device, tag='', para=False):
    # net_baseline = UnetBaselineD(shared_depth=opt.shared_depth, use_vm_decoder=opt.use_vm_decoder,
    #                              blocks=opt.num_blocks)
    net_baseline = UnetBaselineD(shared_depth=opt.shared_depth, use_vm_decoder=opt.use_vm_decoder, blocks=opt.num_blocks)
    if para:
        net_baseline = net_baseline.to(0)
        net_baseline = nn.DataParallel(net_baseline, device_ids=[0,1], output_device=0)
    # if tag != '':
    #     tag = '_' + str(tag)
    cur_path = '%s/net_baseline%s.pth' % (net_path, tag)
    print(cur_path)
    if os.path.isfile(cur_path):
        print('loading baseline from %s/' % net_path)
        net_baseline.load_state_dict(torch.load(cur_path))
    net_baseline = net_baseline.to(device)
    return net_baseline


def save_test_images(net, loader, image_name, device):
    with torch.no_grad():
        net.eval()
        synthesized, images, vm_mask, vm_area, _ = next(iter(loader))
        # print(synthesized.shape)
        # exit(-1)
        # expanded_real_mask = vm_mask.repeat(1, 3, 1, 1)
        vm_mask = vm_mask.to(device)
        synthesized = synthesized.to(device)
        images = images.to(device)
        expanded_real_mask = vm_mask.repeat(1, 3, 1, 1)
        # output = net(synthesized, 1 - expanded_real_mask)
        output = net(synthesized)
        # print(output[0].shape)
        # exit(-1)
        guess_images, guess_mask = output[0], output[-1]
        # print(guess_mask)
        expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
        print(torch.mean(guess_images, (0,2,3)))
        # debug
        # for i in range(guess_images.shape[0]):
        #     a = guess_images[i, 0, :, :].cpu().numpy()
        #     b = guess_images[i, 1, :, :].cpu().numpy()
        #     c = guess_images[i, 2, :, :].cpu().numpy()
        #     print(np.corrcoef(a, b), np.corrcoef(a, c), np.corrcoef(b, c))
        reconstructed_pixels = guess_images * expanded_guess_mask
        reconstructed_images = synthesized * (1 - expanded_guess_mask) + reconstructed_pixels
        real_pixels = images * expanded_real_mask
        transformed_guess_mask = expanded_guess_mask * 2 - 1
        expanded_real_mask = expanded_real_mask * 2 - 1
        if len(output) == 3:
            guess_vm = output[2]
            reconstructed_vm = (guess_vm - 1) * expanded_guess_mask + 1
            images_un = (torch.cat((synthesized, reconstructed_images, images, reconstructed_vm, transformed_guess_mask), 0))
        else:
            images_un = (torch.cat((synthesized, reconstructed_images, guess_images, transformed_guess_mask, expanded_real_mask), 0))
        # print(torch.sum(guess_mask), torch.sum(vm_mask))
        images_un = torch.clamp(images_un.data, min=-1, max=1)
        images_un = make_grid(images_un, nrow=synthesized.shape[0], padding=5, pad_value=1)
        print(image_name)
        save_image(images_un, image_name)
    net.train()
    return images_un



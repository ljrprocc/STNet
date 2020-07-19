import sys
sys.path.append('/home/jingru.ljr/Motif-Removal')
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils.train_utils import *
from train.eval import *
from torch import nn
import multiprocessing

# paths
root_path = '..'
train_tag = 'demo_coco_1'


# datasets paths
cache_root = ['/data/jingru.ljr/COCO/syn_output/']

# dataset configurations
patch_size = 128
image_size_w = 600
image_size_h = 720

# network
nets_path = '%s/checkpoints/%s' % (root_path, train_tag)
images_path = '%s/images' % nets_path

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = False
use_rgb = True


# train configurations
gamma1 = 2   # L1 image
gamma2 = 1   # L1 visual motif
gamma3 = 10  # L1 style loss
gamma4 = 0.02 # Perceptual
gamma5 = 10   # L1 valid
epochs = 50
batch_size = 32
print_frequency = 20
save_frequency = 5
device = torch.device('cuda:0')


def l1_relative(reconstructed, real, batch, area):
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1

def dice_loss(guess_mask, vm_mask, dice_criterion, training_masks=None):
    
    if training_masks is None:
        training_masks = torch.ones(vm_mask.size())
    selected_masks = ohem_batch(guess_mask, vm_mask, training_masks)
    
    selected_masks = selected_masks.to(device)
    # print(torch.sum(vm_mask - selected_masks))
    loss = dice_criterion(guess_mask, vm_mask, selected_masks)
    return loss, selected_masks

def train(net, train_loader, test_loader):
    bce = nn.BCELoss()
    style = StyleLoss()
    per = PerceptionLoss()
    tv = TotalVariationLoss(3)
    dice = DiceLoss()
    vgg_feas = VGGFeature().to(device)
    net.set_optimizers()
    losses = []
    style_losses = []
    print('Training Begins')
    selected_masks = None
    for epoch in range(epochs):
        real_epoch = epoch + 1
        for i, data in enumerate(train_loader, 0):
            # exit(-1)
            synthesized, images, vm_mask, vm_area, total_area = data
            synthesized, images, = synthesized.to(device), images.to(device)
            vm_mask, vm_area, total_area = vm_mask.to(device), vm_area.to(device), total_area.to(device)
            results = net(synthesized)
            expanded_vm_mask = vm_mask.repeat(1, 3, 1, 1)
            # results = net(synthesized, 1 - expanded_vm_mask)
            guess_images, guess_mask = results[0], results[1]
            # print('hiahiahia')
            # print(guess_mask.shape, guess_images.shape)
            # exit(-1)
            # print(synthesized.shape)
            # exit(-1)
            
            expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_vm_mask
            reconstructed_images = synthesized * (1 - expanded_guess_mask) + reconstructed_pixels
            real_pixels = images * expanded_vm_mask
            batch_cur_size = vm_mask.shape[0]
            # total_area = vm_mask.shape[-1] * vm_mask.shape[-2]
            net.zero_grad_all()
            loss_l1_images = l1_relative(reconstructed_pixels, real_pixels, batch_cur_size, vm_area)
            loss_l1_holes = l1_relative(synthesized * (1 - expanded_guess_mask), images * (1 - expanded_vm_mask), batch_cur_size, total_area-vm_area)
            # print(vm_mask.dtype, guess_mask.dtype)
            loss_mask = bce(guess_mask, vm_mask)
            # loss_mask, selected_masks = dice_loss(guess_mask, vm_mask, dice, selected_masks)
            # print(loss_mask, loss_l1_images)
            # Construct Sytle Loss
            loss_style = style(vgg_feas(reconstructed_images), vgg_feas(images))
            loss_perceptual = per(vgg_feas(reconstructed_images), vgg_feas(images))
            # loss_style = 0
            loss_l1_vm = 0
            if len(results) == 3:
                guess_vm = results[2]
                reconstructed_motifs = guess_vm * expanded_vm_mask
                real_vm = motifs.to(device) * expanded_vm_mask
                loss_l1_vm = l1_relative(reconstructed_motifs, real_vm, batch_cur_size, vm_area)
            # loss = gamma1 * loss_l1_images + gamma2 * loss_l1_vm + gamma3 * loss_style + gamma4 * loss_perceptual + gamma5 * loss_l1_holes+ loss_mask
            loss = gamma1 * loss_l1_images + loss_mask + gamma5 * loss_l1_holes
            loss.backward()
            net.step_all()
            losses.append(loss.item())
            # style_losses.append(loss_l1_holes.item())
            # print
            if (i + 1) % print_frequency == 0:
                print('%s [%d, %3d], baseline loss: %.4f' % (train_tag, real_epoch, batch_size * (i + 1), sum(losses) / len(losses)))
                losses = []
                style_losses = []
        # savings
        if real_epoch % save_frequency == 0:
            print("checkpointing...")
            image_name = '%s/%s_%d.png' % (images_path, train_tag, real_epoch)
            _ = save_test_images(net, test_loader, image_name, device)
            torch.save(net.state_dict(), '%s/net_baseline.pth' % nets_path)
            torch.save(net.state_dict(), '%s/net_baseline_%d.pth' % (nets_path, real_epoch))


    print('Training Done:)')


def run():
    init_folders(nets_path, images_path)
    opt = load_globals(nets_path, globals(), override=True)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)
    base_net = init_nets(opt, nets_path, device)
    train(base_net, train_loader, test_loader)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run()

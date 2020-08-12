import sys
sys.path.append('/home/jingru.ljr/Motif-Removal')
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from networks.gan_model import *
from utils.train_utils import *
from train.eval import *
from torch import nn
import multiprocessing

# paths
root_path = '..'
# train_tag = 'coco_gan'
train_tag = 'demo_coco'
# train_tag = 'demo_msra_1'

# datasets paths
cache_root = ['/data/jingru.ljr/COCO/syn_output/']
# cache_root = ['/data/jingru.ljr/icdar2015/syn_ds_root_1280_4x/']
# cache_root = ['/data/jingru.ljr/MSRA-TD500/syn_ds_root/']

# dataset configurations
patch_size = 128
image_size_w = 640
image_size_h = 720

# network
nets_path = '%s/checkpoints/%s' % (root_path, train_tag)
images_path = '%s/images' % nets_path

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = False
use_rgb = True
gen_only = True
dis_channels = 32
gen_channels = 24
dilation_depth = 0
TDBmode = True

# train configurations
gamma1 = 2   # L1 image
gamma2 = 1   # L1 visual motif
gamma3 = 10  # L1 style loss
gamma4 = 0.02 # Perceptual
gamma5 = 3   # L1 valid
gamma_dis = 0.5
gamma_gen = 5
gamma_coarse = 1
gamma_coarse_hole = 0.2
epochs = 50
batch_size = 16
print_frequency = 10
save_frequency = 5
device = torch.device('cuda:0')

def l1_relative(reconstructed, real, batch, area):
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1

def l1_loss(predict, target):
    return torch.abs(predict - target).mean()


def dice_loss(guess_mask, vm_mask, dice_criterion, training_masks=None):
    
    if training_masks is None:
        training_masks = torch.ones(vm_mask.size())
    selected_masks = ohem_batch(guess_mask, vm_mask, training_masks)
    
    selected_masks = selected_masks.to(device)
    # print(torch.sum(vm_mask - selected_masks))
    loss = dice_criterion(guess_mask, vm_mask, selected_masks)
    return loss, selected_masks

def train(net, train_loader, test_loader):
    # net = nets.module
    bce = nn.BCELoss()
    style = StyleLoss()
    per = PerceptionLoss()
    tv = TotalVariationLoss(3)
    dice = DiceLoss()
    vgg_feas = VGGFeature().to(device)
    net.set_optimizers()
    losses = []
    D_losses = []
    G_losses = []
    start_epoch = 0
    # if start_epoch > 0:
    #     net.load(start_epoch)
    print('Training Begins')
    selected_masks = None
    for epoch in range(start_epoch, epochs):
        real_epoch = epoch + 1
        for i, data in enumerate(train_loader, 0):
            # exit(-1)
            with torch.autograd.set_detect_anomaly(True):
                synthesized, images, vm_mask, vm_area, total_area = data
                synthesized, images, = synthesized.to(device), images.to(device)
                vm_mask, vm_area, total_area = vm_mask.to(device), vm_area.to(device), total_area.to(device)
                # results = net(synthesized)
                if TDBmode:
                    results = net(synthesized)
                    guess_images, guess_mask = results[0], results[1]
                    gen_loss, dis_loss = 0., 0.
                else:
                    guess_images, gen_loss, dis_loss, guess_mask, coarse_images = net(synthesized)
                expanded_vm_mask = vm_mask.repeat(1, 3, 1, 1)

                # results = net(synthesized, 1 - expanded_vm_mask)
                # guess_images, guess_mask = results[0], results[1]
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
                # loss_l1_images = l1_relative(reconstructed_pixels, real_pixels, batch_cur_size, vm_area)
                # loss_l1_holes = l1_relative(synthesized * (1 - expanded_guess_mask), images * (1 - expanded_vm_mask), batch_cur_size, total_area-vm_area)
                loss_l1_recon = l1_loss(reconstructed_pixels, real_pixels)
                loss_l1_outer = l1_loss(reconstructed_images * (1 - expanded_vm_mask), images * (1 - expanded_vm_mask))
                # print(loss_l1_recon, loss_l1_outer)
                loss_mask, loss_coarse, loss_coarse_hole = 0., 0., 0.
                if TDBmode or (not TDBmode and not gen_only):
                    # print(vm_mask.dtype, guess_mask.dtype)
                    loss_mask = bce(guess_mask, vm_mask)
                    if not TDBmode:
                        loss_coarse = l1_loss(coarse_images * expanded_vm_mask, real_pixels)
                        loss_coarse_hole = l1_loss(coarse_images * (1 - expanded_vm_mask), images * (1 - expanded_vm_mask))
                    # loss_mask, selected_masks = dice_loss(guess_mask, vm_mask, dice, selected_masks)
                    # print(loss_mask, loss_l1_images)
                    # Construct Sytle Loss
                    # loss_style = style(vgg_feas(reconstructed_images), vgg_feas(images))
                    # loss_perceptual = per(vgg_feas(reconstructed_images), vgg_feas(images))
                    # loss_style = 0
                    loss_l1_vm = 0
                # if len(results) == 3:
                #     guess_vm = results[2]
                #     reconstructed_motifs = guess_vm * expanded_vm_mask
                #     real_vm = motifs.to(device) * expanded_vm_mask
                #     loss_l1_vm = l1_relative(reconstructed_motifs, real_vm, batch_cur_size, vm_area)
                # loss = gamma1 * loss_l1_images + gamma2 * loss_l1_vm + gamma3 * loss_style + gamma4 * loss_perceptual + gamma5 * loss_l1_holes+ loss_mask
                loss = gamma1 * loss_l1_recon + loss_mask + gamma5 * loss_l1_outer + gamma_dis * dis_loss + gamma_gen * gen_loss + gamma_coarse * loss_coarse + gamma_coarse_hole * loss_coarse_hole
                loss.backward()
                net.step_all()
                losses.append(loss.item())
                if not TDBmode:
                    D_losses.append(dis_loss.item())
                    G_losses.append(gen_loss.item())
            # print
            if (i + 1) % print_frequency == 0:
                if TDBmode:
                    print('%s [%d, %3d], total loss: %.4f' % (train_tag, real_epoch, batch_size * (i + 1), sum(losses) / len(losses)))
                else:
                    print('%s [%d, %3d], total loss: %.4f, D loss: %.4f, G loss: %.4f' % (train_tag, real_epoch, batch_size * (i + 1), sum(losses) / len(losses), sum(D_losses)/len(D_losses), sum(G_losses) / len(G_losses)))
                losses = []
                style_losses = []
        # savings
        if real_epoch % save_frequency == 0:
            print("checkpointing...")
            image_name = '%s/%s_%d_fine.png' % (images_path, train_tag, real_epoch)
            _ = save_test_images(net, test_loader, image_name, device)
            if not TDBmode and not os.path.exists('%s/epoch%d'%(nets_path, real_epoch)):
                os.mkdir('%s/epoch%d'%(nets_path , real_epoch))
            if not TDBmode:
                torch.save(net.generator.state_dict(), '%s/epoch%d/net_baseline_G.pth' % (nets_path, real_epoch))
                torch.save(net.discriminator.state_dict(), '%s/epoch%d/net_baseline_D.pth' % (nets_path, real_epoch))
            if TDBmode:
                torch.save(net.state_dict(), '%s/net_baseline%d.pth' % (nets_path, real_epoch))
            if (not TDBmode and not gen_only):
                torch.save(net.mask_generator.state_dict(), '%s/net_baseline%d.pth' % (nets_path, real_epoch))


    print('Training Done:)')


def run():
    init_folders(nets_path, images_path)
    opt = load_globals(nets_path, globals(), override=True)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)
    # base_net = init_nets(opt, nets_path, device)
    base_net = InpaintModel(opt, nets_path, device, tag='1000').to(device) if not TDBmode else init_nets(opt, nets_path, device, tag='1000')
    train(base_net, train_loader, test_loader)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run()

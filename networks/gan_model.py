import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch.optim as optim

from networks.inpaint_model import *
from utils.train_utils import *
# from networks.baselines import weight_init

def weight_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

class Discriminator(nn.Module):
    def __init__(self, hidden_channels=64, patch_size=128):
        super(Discriminator, self).__init__()
        self.hidden_channels = hidden_channels
        self.dis_convs = []
        self.dis_bns = []
        # Build discriminator
        self.dis_convs.append(spectral_norm(nn.Conv2d(3, hidden_channels, kernel_size=5, stride=2, padding=2)))
        self.dis_bns.append(nn.BatchNorm2d(hidden_channels))
        self.dis_convs.append(spectral_norm(nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=5, stride=2, padding=2)))
        self.dis_bns.append(nn.BatchNorm2d(hidden_channels*2))
        self.dis_convs.append(spectral_norm(nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=5, stride=2, padding=2)))
        self.dis_bns.append(nn.BatchNorm2d(hidden_channels*4))
        self.dis_convs.append(spectral_norm(nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=5, stride=2, padding=2)))
        self.dis_bns.append(nn.BatchNorm2d(hidden_channels * 8))
        # self.dis_convs.append(spectral_norm(nn.Conv2d(hidden_channels*4, hidden_channels*4, kernel_size=5, stride=2, padding=2)))
        # self.dis_convs.append(spectral_norm(nn.Conv2d(hidden_channels*4, 1, kernel_size=3, stride=1, padding=1)))
        self.dis_convs = nn.ModuleList(self.dis_convs)
        self.dis_bns = nn.ModuleList(self.dis_bns)
        now_size = patch_size // (2 ** len(self.dis_convs))
        self.linear = nn.Linear(now_size * now_size * 8 * hidden_channels, 1)
        # self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        for i, conv in enumerate(self.dis_convs):
            # print(x.shape)
            x = conv(x)
            # print(x.shape)
            # if conv.out_channels != 1:
            x = self.dis_bns[i](x)
            x = self.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        # x = torch.tanh(x)
        # x = torch.clamp(self.relu(x), min=-1., max=1.)
        # print(x)
        # out = torch.flatten(x)
        # out = torch.sigmoid(out)
        return x


class InpaintModel(nn.Module):
    def __init__(self, opt, net_path, device, tag='', gen_only=True, gate=False):
        super(InpaintModel, self).__init__()
        self.adversial_loss = AdversialLoss('lsgan')
        self.discriminator = Discriminator(opt['gen']['hidden_channels'], patch_size=opt['patch_size'])
        self.mask_generator = init_nets(opt, net_path, device, tag)
        self.generator = Coarse2FineModel(opt['dis']['hidden_channels'], opt['dilation_depth']) if not gate else GatedCoarse2FineModel(opt['dis']['hidden_channels'], opt['dilation_depth'])
        self.gate = gate
        # print(self.generator)
        self.gen_only = gen_only
        self.net_path = net_path
        self.use_coarse = opt['open_image'] == 'open'
        self.tag = tag
        self.device = device
        self.set_optimizers(tag=='')
        print('**Parameter calculation:')
        print('# mask_generator parameters:', sum(param.numel() for param in self.mask_generator.parameters()))
        print('# Generator parameters:', sum(param.numel() for param in self.generator.parameters()))
        print('# Discriminator parameters:', sum(param.numel() for param in self.discriminator.parameters()))
        # self.update_device(device)
        if tag != '':
            if not gen_only:
                self.load(int(tag))

    def update_device(self, device):
        self.discriminator = self.discriminator.to(device)
        self.mask_generator = self.mask_generator.to(device)
        self.generator = self.generator.to(device)
    
    def dis_gen_forward(self, x, x_real=None):
        # Only support critic_iter=1
        self.discriminator.zero_grad()
        set_requires_grad(self.mask_generator, False)
        # set_requires_grad(self.generator, False)
        # set_requires_grad(self.discriminator, True)
        results_mask_gen = self.mask_generator(x)
        corase_image, result_mask = results_mask_gen[0], results_mask_gen[1]
        # print(torch.mean(corase_image))
        # print(torch.mean(x))
        hard_mask = (result_mask.repeat(1,3,1,1) > 0.9).float()
        x_out, offsests = self.generator(corase_image if self.use_coarse else x, xori=x, mask=result_mask)
        # print(torch.mean(corase_image), torch.mean(result_mask))
        # print(torch.mean(x))
        # print(torch.mean(x_out))
            
        hard_mask = result_mask
        fine_image = x_out * hard_mask + x * (1 - hard_mask)
        # discriminator loss
        dis_input_real = x_real
        dis_input_fake = fine_image
        # dis_input_fake = x_out.detach()
        # print(torch.mean(dis_input_real), torch.mean(dis_input_fake))
        dis_real = self.discriminator(dis_input_real)
        dis_fake = self.discriminator(dis_input_fake.detach())
        # print(torch.mean(dis_real), torch.mean(dis_fake))
        # print(dis_fake)
        # print(torch.mean(fine_image))
        dis_real_loss = self.adversial_loss(dis_real, True, True)
        dis_real_loss.backward()
        dis_fake_loss = self.adversial_loss(dis_fake, False, True)
        dis_fake_loss.backward()
        # print(dis_real_loss, dis_fake_loss)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2
        self.dis_optimzer.step()
        self.generator.zero_grad()
        # set_requires_grad(self.generator, True)
        # set_requires_grad(self.discriminator, False)
        gen_fake = self.discriminator(dis_input_fake)
        gen_gan_loss = self.adversial_loss(gen_fake, True, False)
        # print(gen_gan_loss)
        gen_loss = gen_gan_loss
        return x_out, gen_loss, dis_loss, result_mask, corase_image, offsests



    def forward(self, x, update='gen', x_real=None):
        assert update in ['gen', 'dis', 'mask']
        # with torch.no_grad():
            # results_mask_gen = self.mask_generator(x)
            # corase_image, result_mask = results_mask_gen[0], results_mask_gen[1]
        # if update == 'gen':    
        #     x_out, offsests = self.generator(corase_image, xori=x, mask=result_mask)
        # else:
        #     with torch.no_grad():
        #         x_out, offsests = self.generator(corase_image, xori=x, mask=result_mask)
        # hard_mask = (result_mask.repeat(1,3,1,1) > 0.9).int()
        # fine_image = x_out * hard_mask + x * (1 - hard_mask)
        # print(x_out[0][:, :, 0].mean(), x_out[0][:, :, 1].mean(), x_out[0][:, :, 2].mean())
        gen_loss = 0.
        dis_loss = 0.
        if update=='dis':
            set_requires_grad(self.mask_generator, False)
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)
            results_mask_gen = self.mask_generator(x)
            corase_image, result_mask = results_mask_gen[0], results_mask_gen[1]
            # print(torch.mean(corase_image))
            # print(torch.mean(x))
            hard_mask = (result_mask.repeat(1,3,1,1) > 0.9).float()
            x_out, offsests = self.generator(corase_image if self.use_coarse else x, xori=x, mask=result_mask)
            # print(torch.mean(corase_image), torch.mean(result_mask))
            # print(torch.mean(x))
            # print(torch.mean(x_out))
            
            hard_mask = result_mask
            fine_image = x_out * hard_mask + x * (1 - hard_mask)
            # discriminator loss
            dis_input_real = x_real
            dis_input_fake = fine_image
            # dis_input_fake = x_out.detach()
            # print(torch.mean(dis_input_real), torch.mean(dis_input_fake))
            dis_real = self.discriminator(dis_input_real)
            dis_fake = self.discriminator(dis_input_fake.detach())
            # print(torch.mean(dis_real), torch.mean(dis_fake))
            # print(dis_fake)
            # print(torch.mean(fine_image))
            dis_real_loss = self.adversial_loss(dis_real, True, True)
            dis_fake_loss = self.adversial_loss(dis_fake, False, True)
            # print(dis_real_loss, dis_fake_loss)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            # print(torch.mean(corase_image - x))
        elif update=='gen':
            set_requires_grad(self.mask_generator, False)
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
            results_mask_gen = self.mask_generator(x)
            corase_image, result_mask = results_mask_gen[0], results_mask_gen[1]
            hard_mask = (result_mask.repeat(1,3,1,1) > 0.9).float()
            x_out, offsests = self.generator(corase_image if self.use_coarse else x, xori=x, mask=result_mask)
            # print(torch.mean(x))
            # print(torch.mean(corase_image), torch.mean(result_mask))
            # hard_mask = (result_mask.repeat(1,3,1,1) > 0.8).float()
            # hard_mask = result_mask
            fine_image = x_out * hard_mask + x * (1 - hard_mask)
            # print(torch.mean(fine_image), torch.mean(x_out))

            gen_input_fake = fine_image
            # gen_input_fake = x_out
            gen_fake = self.discriminator(gen_input_fake)
            gen_gan_loss = self.adversial_loss(gen_fake, True, False)
            # print(gen_gan_loss)
            gen_loss += gen_gan_loss
            # print(x_out)
            return x_out, gen_loss, fine_image, result_mask, corase_image, offsests
        else:
            # set_requires_grad(self.mask_generator, False)
            # set_requires_grad(self.mask_generator.mask_decoder, True)
            set_requires_grad(self.mask_generator, True)
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, False)
            results_mask_gen = self.mask_generator(x)
            corase_image, result_mask = results_mask_gen[0], results_mask_gen[1]
            hard_mask = (result_mask.repeat(1,3,1,1) > 0.8).float()
            x_out, offsests = self.generator(corase_image if self.use_coarse else x, xori=x, mask=result_mask)
            # print(torch.mean(x))
            # hard_mask = result_mask
            # print(torch.mean(corase_image), torch.mean(hard_mask))
            
            fine_image = x_out * hard_mask + x * (1 - hard_mask)
            return x_out, fine_image, dis_loss, result_mask, corase_image, offsests

        return x_out, gen_loss, dis_loss, result_mask, corase_image, offsests
        
    def zero_grad_all(self):
        self.dis_optimzer.zero_grad()
        self.gen_optimzer.zero_grad()
        if not self.gen_only:
            self.mask_generator.zero_grad_all()

    def step_all(self):
        self.dis_optimzer.step()
        self.gen_optimzer.step()
        if not self.gen_only:
            self.mask_generator.step_all()

    def set_optimizers(self, init=True):
        self.dis_optimzer = optim.Adam(
            params = self.discriminator.parameters(),
            lr = 0.0002,
            betas = (0.0, 0.99)
        )
        # print(self.generator.parameters()[0])
        self.gen_optimzer = optim.Adam(
            params = self.generator.parameters(),
            lr = 0.0002,
            betas = (0.0, 0.99)
        )
        # if self.gen_only:
        #     for para in self.mask_generator.parameters():
        #         para.requires_grad = False
        # else:
        #     self.mask_generator.set_optimizers()
            # Default finetune
            # for para in self.mask_generator.shared_decoder.parameters():
            #     para.requires_grad = False

            # for para in self.mask_generator.encoder.parameters():
            #     para.requires_grad = False
        if init:
            self.apply(weight_init('kaiming'))
        else:
            self.generator.apply(weight_init('kaiming'))
            self.discriminator.apply(weight_init('kaiming'))

    def translate_simple(self, x):
        results_mask_gen = self.mask_generator(x)
        corase_image, result_mask = results_mask_gen[0], results_mask_gen[1]
        # print(torch.mean(corase_image - x))
        x_out, offsests = self.generator(corase_image if self.use_coarse else x, xori=x, mask=result_mask)
        hard_mask = (result_mask.repeat(1,3,1,1) > 0.8).float()
        fine_image = x_out * hard_mask + x * (1 - hard_mask)
        return x_out, fine_image, corase_image, result_mask

    def load(self, epoch):
        appendix = 'g' if self.gate else ''
        pathD = '%s/epoch%d/net_baseline_%sD.pth' % (self.net_path, epoch, appendix)
        pathG = '%s/epoch%d/net_baseline_%sG.pth' % (self.net_path, epoch, appendix)
        path_baseline = '%s/net_baseline%d.pth' % (self.net_path, epoch)
        print('Loading parameters of IIM generator....')
        print(pathD)
        self.generator.load_state_dict(torch.load(pathG))
        print('Loading parameters of IIM discriminator...')
        self.discriminator.load_state_dict(torch.load(pathD))
        # print('Loading parameters of TDM..')
        # self.mask_generator.load_state_dict(torch.load())

        
    

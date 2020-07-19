from networks.unet_components import *
from networks.pconv_component import *


class UnetDecoderD(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, is_pconv=False):
        super(UnetDecoderD, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.is_pconv = is_pconv
        outs = in_channels
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            if is_pconv:
                up_conv = UpPConvD(ins, outs, blocks, residual=residual, norm=batch_norm, concat=concat)
            else:
                up_conv = UpConvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                                concat=concat)
            self.up_convs.append(up_conv)
        if is_final:
            self.conv_final = pconv1x1(outs, out_channels) if is_pconv else conv1x1(outs, out_channels)
        else:
            if is_pconv:
                up_conv = UpPConvD(outs, out_channels, blocks, residual=residual, norm=batch_norm, concat=concat)
            else:
                up_conv = UpConvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                                concat=concat)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def __call__(self, x, encoder_outs=None, mask=None, encoder_masks=None):
        return self.forward(x, encoder_outs, mask, encoder_masks)

    def forward(self, x, encoder_outs=None, mask=None, encoder_masks=None):
        # print(len(encoder_masks))
        lastx = x
        for i, up_conv in enumerate(self.up_convs):
            # if isinstance(x, tuple):
            #     print(x)
            before_pool = None
            mask_before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            if encoder_masks is not None:
                mask_before_pool = encoder_masks[-(i+2)]
            # print(i, before_pool.shape, x.shape)
            if self.is_pconv:
                x, mask = up_conv(lastx, mask, before_pool, mask_before_pool)
                # print(x.shape)
                # print('**')
            else:
                x = up_conv(x, before_pool)
            # print(x.shape)
            # print(lastx.shape)
        # print('**')
        if self.conv_final is not None:
            if self.is_pconv:
                x, mask = self.conv_final(x, mask)
            else:
                x = self.conv_final(x)
        return x, mask


class UnetEncoderD(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True, is_pconv=False):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        self.is_pconv = is_pconv
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            pooling = True if i < depth-1 else False
            if is_pconv:
                down_conv = DownPConvD(ins, outs, blocks, pooling=pooling, residual=residual, norm=batch_norm)
            else:
                down_conv = DownConvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def __call__(self, x, mask=None):
        return self.forward(x, mask)

    def forward(self, x, mask=None):
        encoder_outs = []
        encoder_masks = []
        mask_in = mask
        # print(self.down_convs)
        for d_conv in self.down_convs:
            if self.is_pconv:
                # print(mask)
                # wdconv = DownPConvD(3, 32, 5).to(0)
                # print(x.shape, mask_in.shape)
                x, before_pool, mask_in, mask_before_pool = d_conv(x, mask_in)
                # print(wdconv(x, mask))
                # exit(-1)
            else:
                x, before_pool = d_conv(x)
                mask_before_pool = None
            encoder_outs.append(before_pool)
            encoder_masks.append(mask_before_pool)
        return x, encoder_outs, mask_in, encoder_masks


class UnetEncoderDecoderD(nn.Module):
    def __init__(self, in_channels=3, depth=5, blocks_encoder=1, blocks_decoder=1, out_channels=3,
                 start_filters=32, residual=True, batch_norm=True, transpose=True, concat=True, transfer_data=True,
                 activation=f.tanh):
        super(UnetEncoderDecoderD, self).__init__()
        self.transfer_data = transfer_data
        self.__activation = activation
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks_encoder,
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm)
        self.decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1),
                                    out_channels=out_channels, depth=depth, blocks=blocks_decoder, residual=residual,
                                    batch_norm=batch_norm, transpose=transpose, concat=concat)

    def __call__(self, synthesized):
        return self.forward(synthesized)

    def forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            reconstructed = self.decoder(image_code, before_pool)
        else:
            reconstructed = self.decoder(image_code)
        return self.__activation(reconstructed)
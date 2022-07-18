import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------------

def _init_conv_layer(conv, activation, mode='fan_out'):
    if isinstance(activation, nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       a=activation.negative_slope,
                                       nonlinearity='leaky_relu',
                                       mode=mode)
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       nonlinearity='relu',
                                       mode=mode)
    else:
        pass
    if conv.bias != None:
        torch.nn.init.zeros_(conv.bias)

def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out

#----------------------------------------------------------------------------

def same_padding(images, ksizes, strides, rates):
    """Implements tensorflow "SAME" padding as defined in:
       https://github.com/tensorflow/tensorflow/blob/8eaf671025e8cd5358278f91f7e89e2fbbe6a26b/tensorflow/core/kernels/ops_util.cc#L65
       see also: https://www.tensorflow.org/api_docs/python/tf/nn#same_padding_2
    """
    in_height, in_width = images.shape[2:]
    out_height = -(in_height // -strides[0])  # ceil(a/b) = -(a//-b)
    out_width = -(in_width // -strides[1])
    filter_height = (ksizes[0]-1)*rates[0] + 1
    filter_width = (ksizes[1]-1)*rates[1] + 1
    pad_along_height = max(
        (out_height-1)*strides[0] + filter_height - in_height, 0)
    pad_along_width = max(
        (out_width-1)*strides[1] + filter_width - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    paddings = (pad_left, pad_right, pad_top, pad_bottom)
    padded_images = torch.nn.ZeroPad2d(paddings)(images)
    return padded_images

#----------------------------------------------------------------------------

def downsampling_nn_tf(images, n=2):
    """NN downsampling with tensorflow option align_corners=True \\
       Args:
           images: input
           n: downsampling factor
    """
    in_height, in_width = images.shape[2:]
    out_height, out_width = in_height // n, in_width // n
    height_inds = torch.linspace(0, in_height-1, steps=out_height, device=images.device).add_(0.5).floor_().long()
    width_inds = torch.linspace(0, in_width-1, steps=out_width, device=images.device).add_(0.5).floor_().long()
    return images[:, :, height_inds][..., width_inds]

#----------------------------------------------------------------------------

#################################
########### GENERATOR ###########
#################################

class GConv(nn.Module):
    """Implements the gated 2D convolution introduced in 
       `Free-Form Image Inpainting with Gated Convolution`(Yu et al., 2019) \\
        Uses the SAME padding from tensorflow.
    """

    def __init__(self,
                 cnum_in,
                 cnum_out,
                 ksize,
                 stride=1,
                 rate=1,
                 padding='same',
                 activation=nn.ELU()
                 ):

        super().__init__()

        self.activation = activation
        self.cnum_out = cnum_out
        num_conv_out = cnum_out if self.cnum_out == 3 or self.activation is None else 2*cnum_out
        self.conv = nn.Conv2d(cnum_in,
                              num_conv_out,
                              kernel_size=ksize,
                              stride=stride,
                              padding=0,
                              dilation=rate)

        _init_conv_layer(self.conv, activation=self.activation)

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.padding = padding

    def forward(self, x):
        x = same_padding(x, [self.ksize, self.ksize], [self.stride, self.stride],
                         [self.rate, self.rate])

        x = self.conv(x)
        if self.cnum_out == 3 or self.activation is None:
            return x
        x, y = torch.split(x, self.cnum_out, dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

#----------------------------------------------------------------------------

class GDeConv(nn.Module):
    """Upsampling followed by convolution"""

    def __init__(self, cnum_in,
                 cnum_out,
                 padding=1):
        super().__init__()
        self.conv = GConv(cnum_in, cnum_out, 3, 1,
                          padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest',
                           recompute_scale_factor=False) 
        x = self.conv(x)
        return x

#----------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, cnum_in=5, cnum=48, return_flow=False, checkpoint=None):
        super().__init__()

        # stage 1
        self.conv1 = GConv(cnum_in, cnum//2, 5, 1, padding=2)

        self.conv2_downsample = GConv(cnum//2, cnum, 3, 2)
        self.conv3 = GConv(cnum, cnum, 3, 1)
        self.conv4_downsample = GConv(cnum, 2*cnum, 3, 2)
        self.conv5 = GConv(2*cnum, 2*cnum, 3, 1)

        self.conv6 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv7_atrous = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
        self.conv8_atrous = GConv(2*cnum, 2*cnum, 3, rate=4, padding=4)
        self.conv9_atrous = GConv(2*cnum, 2*cnum, 3, rate=8, padding=8)
        self.conv10_atrous = GConv(2*cnum, 2*cnum, 3, rate=16, padding=16)
        self.conv11 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv12 = GConv(2*cnum, 2*cnum, 3, 1)

        self.conv13_upsample = GDeConv(2*cnum, cnum)
        self.conv14 = GConv(cnum, cnum, 3, 1)
        self.conv15_upsample = GDeConv(cnum, cnum//2)
        self.conv16 = GConv(cnum//2, cnum//4, 3, 1)

        self.conv17 = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

        # stage 2
        # conv branch
        self.xconv1 = GConv(3, cnum//2, 5, 1, padding=2)

        self.xconv2_downsample = GConv(cnum//2, cnum//2, 3, 2)
        self.xconv3 = GConv(cnum//2, cnum, 3, 1)
        self.xconv4_downsample = GConv(cnum, cnum, 3, 2)
        self.xconv5 = GConv(cnum, 2*cnum, 3, 1)

        self.xconv6 = GConv(2*cnum, 2*cnum, 3, 1)
        self.xconv7_atrous = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
        self.xconv8_atrous = GConv(2*cnum, 2*cnum, 3, rate=4, padding=4)
        self.xconv9_atrous = GConv(2*cnum, 2*cnum, 3, rate=8, padding=8)
        self.xconv10_atrous = GConv(2*cnum, 2*cnum, 3, rate=16, padding=16)

        # attention branch
        self.pmconv1 = GConv(3, cnum//2, 5, 1, padding=2)

        self.pmconv2_downsample = GConv(cnum//2, cnum//2, 3, 2)
        self.pmconv3 = GConv(cnum//2, cnum, 3, 1)
        self.pmconv4_downsample = GConv(cnum, 2*cnum, 3, 2)
        self.pmconv5 = GConv(2*cnum, 2*cnum, 3, 1)

        self.pmconv6 = GConv(2*cnum, 2*cnum, 3, 1, activation=nn.ReLU())
        self.contextual_attention = ContextualAttention(ksize=3,
                                                        stride=1,
                                                        rate=2,
                                                        fuse_k=3,
                                                        softmax_scale=10,
                                                        fuse=False,
                                                        device_ids=None,
                                                        n_down=2,
                                                        return_flow=return_flow)

        self.pmconv9 = GConv(2*cnum, 2*cnum, 3, 1)
        self.pmconv10 = GConv(2*cnum, 2*cnum, 3, 1)

        self.allconv11 = GConv(4*cnum, 2*cnum, 3, 1)
        self.allconv12 = GConv(2*cnum, 2*cnum, 3, 1)

        self.allconv13_upsample = GDeConv(2*cnum, cnum)
        self.allconv14 = GConv(cnum, cnum, 3, 1)
        self.allconv15_upsample = GDeConv(cnum, cnum//2)
        self.allconv16 = GConv(cnum//2, cnum//4, 3, 1)

        self.allconv17 = GConv(cnum//4, 3, 3, 1, activation=None)

        self.return_flow = return_flow

        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint)['G']
            self.load_state_dict(generator_state_dict, strict=True)
        self.eval();

    def forward(self, x, mask):
        xin = x

        # stage 1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)

        x = self.conv17(x)
        x = self.tanh(x)
        x_stage1 = x

        # stage2, paste result as input
        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)

        # conv branch
        xnow = x
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)

        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)

        x = self.pmconv6(x)
        x, offset_flow = self.contextual_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)

        x = self.allconv17(x)
        x = self.tanh(x)
        x_stage2 = x

        if self.return_flow:
            return x_stage1, x_stage2, offset_flow
            
        return x_stage1, x_stage2

    @torch.inference_mode()
    def infer(self,
              image,
              mask,
              return_vals=['inpainted', 'stage1'], 
              device='cuda'):
        """
        Args:
            image: 
            mask:
            return_vals: inpainted, stage1, stage2, flow
        Returns:

        """

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        image = (image*2 - 1.)  # map image values to [-1, 1] range
        # 1.: masked 0.: unmasked
        mask = (mask > 0.).to(dtype=torch.float32)

        image_masked = image * (1.-mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # sketch channel
        x = torch.cat([image_masked, ones_x, ones_x*mask],
                      dim=1)  # concatenate channels

        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)

        image_compl = image * (1.-mask) + x_stage2 * mask

        output = []
        for return_val in return_vals:
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')

        return output


#----------------------------------------------------------------------------

####################################
####### CONTEXTUAL ATTENTION #######
####################################

"""
adapted from: https://github.com/daa233/generative-inpainting-pytorch/blob/c6cdaea0427b37b5b38a3f48d4355abf9566c659/model/networks.py
"""
class ContextualAttention(nn.Module):
    """ Contextual attention layer implementation. \\
        Contextual attention is first introduced in publication: \\
        `Generative Image Inpainting with Contextual Attention`(Yu et al., 2019) \\
        Args:
            ksize: Kernel size for contextual attention
            stride: Stride for extracting patches from b
            rate: Dilation for matching
            softmax_scale: Scaled softmax for attention
    """

    def __init__(self,
                 ksize=3,
                 stride=1,
                 rate=1,
                 fuse_k=3,
                 softmax_scale=10.,
                 n_down=2,
                 fuse=True,
                 return_flow=False,
                 device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device_ids = device_ids
        self.n_down = n_down
        self.return_flow = return_flow

    def forward(self, f, b, mask=None):
        """
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
        """
        device = f.device
        # get shapes
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                         strides=[self.rate*self.stride,
                                                  self.rate*self.stride],
                                         rates=[1, 1], padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = downsampling_nn_tf(f, n=self.rate)
        b = downsampling_nn_tf(b, n=self.rate)
        int_fs, int_bs = list(f.size()), list(b.size())   # b*c*h*w
        # split tensors along the batch dimension
        f_groups = torch.split(f, 1, dim=0)
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                     strides=[self.stride, self.stride],
                                     rates=[1, 1], padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            mask = downsampling_nn_tf(mask, n=(2**self.n_down)*self.rate)
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1], padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        
        # mm shape: [L, 1, 1, 1]
        mm = (torch.mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k, device=device).view(1, 1, k, k)  # 1*1*k*k

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(torch.sum(torch.pow(wi, 2), dim=[1, 2, 3], keepdim=True)).clamp_min(1e-4)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                # (B=1, I=1, H=32*32, W=32*32)
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                # (B=1, C=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])                
                yi = F.conv2d(yi, fuse_weight, stride=1)
                # (B=1, 32, 32, 32, 32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            # (B=1, C=32*32, H=32, W=32)
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            
            # softmax to match
            yi = yi * mm 
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            if self.return_flow:
                offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

                if int_bs != int_fs:
                    # Normalize the offset value to match foreground dimension
                    times = (int_fs[2]*int_fs[3])/(int_bs[2]*int_bs[3])
                    offset = ((offset + 1) * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'),
                                    offset % int_fs[3]], dim=1)  # 1*2*H*W

                offsets.append(offset)
           
            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            
        y = torch.cat(y, dim=0)  # back to the mini-batch
        y = y.contiguous().view(raw_int_fs)

        if not self.return_flow:
            return y, None

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2], device=device).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        # to flow image
        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).detach().cpu().numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).detach().cpu().numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate, mode='bilinear', align_corners=True)

        return y, flow

#----------------------------------------------------------------------------

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extracts sliding local blocks \\
    see also: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    """

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
        padding = 0

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             stride=strides,
                             padding=padding,
                             dilation=rates
                             )
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

#----------------------------------------------------------------------------

def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

#----------------------------------------------------------------------------

def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img

#----------------------------------------------------------------------------

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

#----------------------------------------------------------------------------

#################################
######### DISCRIMINATOR #########
#################################

class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in,
                 cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        size = self.weight.size()
        weight_orig = self.weight.view(size[0], -1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x

#----------------------------------------------------------------------------

class DConv(nn.Module):
    def __init__(self, cnum_in, cnum_out, ksize=5, stride=2):
        super().__init__()
        self.conv_sn = Conv2DSpectralNorm(cnum_in, cnum_out, ksize, stride)
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        x = same_padding(x, [self.ksize, self.ksize],
                         [self.stride, self.stride],
                         [1, 1])
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x

#----------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)
        self.conv6 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.Flatten()(x)

        return x

#----------------------------------------------------------------------------
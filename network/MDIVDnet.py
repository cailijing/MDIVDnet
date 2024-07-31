import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from math import cos, pi, sqrt

## for mmcv-full 1.7.0 and mmedit 0.16.0
# from mmcv.cnn import constant_init
# from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
# from mmcv.runner import load_checkpoint
# from mmedit.models.backbones.sr_backbones.basicvsr_net import (
#     ResidualBlocksWithInputConv)
# from mmcv.cnn import ConvModule
# from mmedit.models.common import PixelShufflePack, flow_warp
# from mmedit.utils import get_root_logger

## for mmcv 2.2.0 and mmagic 0.10.4
from mmengine.model import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.runner import load_checkpoint
from mmagic.models.editors.basicvsr.basicvsr_net import ResidualBlocksWithInputConv
from mmcv.cnn import ConvModule
from mmagic.models.archs import PixelShufflePack
from mmagic.models.utils import flow_warp
import logging

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1) #offset_1.shape=torch.Size([1, 144, 180, 320])
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

class feat_reconstruct(nn.Module):
    def __init__(self, filter_in=64, filter_out=1):
        super(feat_reconstruct, self).__init__()
        self.filter_in = filter_in
        if self.filter_in == 128:
            group = 2
        else:
            group = 1
        self.conv_hr1 = nn.Conv2d(in_channels=filter_in, out_channels=64, kernel_size=3, stride=1, padding=1, groups=group)
        self.conv_hr2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, filter_out, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, hr):
        hr = self.lrelu(self.conv_hr1(hr))
        hr = self.lrelu(self.conv_hr2(hr))
        hr = self.conv_last(hr)
        return hr

class matpadding(nn.Module):
    def __init__(self,pad_vertical, pad_horizontal):
        super().__init__()
        self.pad_vertical = pad_vertical
        self.pad_horizontal = pad_horizontal

    def forward(self, meanmat_ori):
        N, C ,H, W = meanmat_ori.shape
        meanmat_new = torch.zeros((N, C ,H+2*self.pad_vertical, W+2*self.pad_horizontal)).to('cuda')
        meanmat_new[:, :, self.pad_vertical:self.pad_vertical + H, self.pad_horizontal:self.pad_horizontal + W] = meanmat_ori[:, :, :, :]

        for i in range(self.pad_horizontal):
            meanmat_new[:, :, self.pad_vertical:self.pad_vertical + H, i] = meanmat_ori[:, :, :, abs(i-self.pad_horizontal)]
            meanmat_new[:, :, self.pad_vertical:self.pad_vertical + H, W + i + self.pad_horizontal] = meanmat_ori[:, :, :, abs(self.pad_horizontal-i-1)]

        for i in range(self.pad_vertical):
            meanmat_new[:, :, i, :] = meanmat_new[:, :, 2*self.pad_vertical-i, :]
            meanmat_new[:, :, i + self.pad_vertical + H, :] = meanmat_new[:, :, self.pad_vertical + H - 2 - i, :]
        return meanmat_new

class FMEkernel_v1(nn.Module):
    def __init__(self, kernelsize):
        super().__init__()
        self.FMEkernel = nn.Parameter(torch.autograd.Variable(torch.ones((1, 1, kernelsize, kernelsize))/pow(kernelsize,2),
                                                              requires_grad=True))

    def forward(self):
        y = self.FMEkernel
        return y

class FMEblock_v1(nn.Module):

    def __init__(self):
        super(FMEblock_v1, self).__init__()
        self.kernel = FMEkernel_v1(5)

    def forward(self, meanmat):
        meanmat_padding = F.pad(meanmat, [2,2,2,2], 'reflect')
        y = F.conv2d(meanmat_padding, self.kernel())

        return y

class linemean_mat(nn.Module):

    def __init__(self):
        super(linemean_mat, self).__init__()

    def forward(self, x):
        meanmat = x.mean(dim=-1).permute(0, 2, 3, 1) #NCHF
        return meanmat

class LNPPBlock_rowcol(nn.Module):

    def __init__(self):
        super(LNPPBlock_rowcol, self).__init__()
        self.fmeblock_line = FMEblock_v1()
        self.fmeblock_col = FMEblock_v1()
        self.meanmat = linemean_mat()

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        N, F, C, H, W = x.shape
        if C == 3:
            x = x[:, :, 1:-1, :, :]

        meanmat = self.meanmat(x)
        fmemat_hat = self.fmeblock_line(meanmat)
        diff_mat = meanmat - fmemat_hat
        x = x - diff_mat.permute(0, 3, 1, 2).unsqueeze(-1).repeat(1, 1, 1, 1, W)
        x = x.permute(0, 1, 2, 4, 3)
        meanmat = self.meanmat(x)
        fmemat_hat = self.fmeblock_col(meanmat)
        diff_mat = meanmat - fmemat_hat
        x = x - diff_mat.permute(0, 3, 1, 2).unsqueeze(-1).repeat(1, 1, 1, 1, H)
        x_result = x.permute(0, 1, 2, 4, 3)
        if C == 3:
            x_result = x_result.repeat(1, 1, 3, 1, 1)
        return x_result

class SPyNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()

        ref = [ref]
        supp = [supp]

        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):

        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow

class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=52,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)

class ScaleLayer(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.kernel = nn.Parameter(torch.tensor(s))

    def forward(self, input):
        y = input * self.kernel
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class kernel_weight(nn.Module):
    def __init__(self,patchsize):
        super().__init__()
        # self.ReLU = nn.ReLU()
        self.it_weights = nn.Parameter(torch.autograd.Variable(torch.ones((patchsize**2, 1, 1, 1)),
                                                               requires_grad=True))

    def forward(self, input):
        y = input.to('cuda') * self.it_weights
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class Kernel_IDCT(nn.Module):
    def __init__(self,patchsize):
        super().__init__()
        conv_shape = (patchsize**2, patchsize**2, 1, 1)
        kernel = torch.zeros(conv_shape).cuda()
        r1 = sqrt(1.0 / patchsize)
        r2 = sqrt(2.0 / patchsize)
        for i in range(patchsize):
            _u = 2 * i + 1
            for j in range(patchsize):
                _v = 2 * j + 1
                index = i * patchsize + j
                for u in range(patchsize):
                    for v in range(patchsize):
                        index2 = u * patchsize + v
                        t = cos(_u * u * pi / (2*patchsize)) * cos(_v * v * pi / (2*patchsize))
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index2, index, 0, 0] = t
        self.kernel = torch.autograd.Variable(kernel)

    def forward(self):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

class Kernel_DCT(nn.Module):
    def __init__(self,patchsize):
        super().__init__()
        conv_shape = (patchsize ** 2, patchsize ** 2, 1, 1)
        kernel = torch.zeros(conv_shape).cuda()
        r1 = sqrt(1.0 / patchsize)
        r2 = sqrt(2.0 / patchsize)
        for u in range(patchsize):
            for v in range(patchsize):
                index = u * patchsize + v
                for i in range(patchsize):
                    _u = 2 * i + 1
                    for j in range(patchsize):
                        _v = 2 * j + 1
                        index2 = i * patchsize + j
                        t = cos(_u * u * pi / (2*patchsize)) * cos(_v * v * pi / (2*patchsize))
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index2, index, 0, 0] = t
        self.kernel = torch.autograd.Variable(kernel)

    def forward(self):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

class adaptive_IDCT_trans(nn.Module):
    def __init__(self,patchsize):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = kernel_weight(patchsize=patchsize)
        self.kernel = Kernel_IDCT(patchsize=patchsize)

    def forward(self, inputs):
        self.kernel1 = self.it_weights(self.kernel())
        y = F.conv2d(inputs, self.kernel1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class adaptive_DCT_trans(nn.Module):
    def __init__(self,patchsize):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = kernel_weight(patchsize=patchsize)
        self.kernel = Kernel_DCT(patchsize=patchsize)

    def forward(self, inputs):
        self.kernel0 = self.it_weights(self.kernel())
        y = F.conv2d(inputs, self.kernel0)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class CRBlock(nn.Module):
    '''(Conv2d => ReLU)'''
    def __init__(self, in_ch,  filters, kernel, padding=1, stride=1, dilation = 1 ):
        super(CRBlock, self).__init__()
        self.in_ch = in_ch
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        if kernel == 1:
            padding = 0
        elif self.stride == 2:
            stride = self.stride
        elif self.dilation == 1:
            pass
        elif self.dilation == 2:
            padding = self.dilation
        elif self.dilation == 3:
            padding = self.dilation

        self.convblock = nn.Sequential(
            nn.Conv2d(self.in_ch, self.filters, self.kernel, padding=padding, stride=stride, dilation=self.dilation),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(
                negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class MFFBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, patchsize, pretrained=None):
        super().__init__()
        self.nFilters = patchsize * patchsize
        self.filterlayer = len(dilation)
        self.dilation = dilation
        self.patchsize = patchsize

        self.conv01 = nn.Conv2d(in_channels=in_ch, out_channels=self.nFilters, kernel_size=1, padding=0)

        self.adaptive_implicit_trans01 = adaptive_DCT_trans(self.patchsize)

        self.conv_lrelu = nn.ModuleList(
            [CRBlock(self.nFilters * (i+1), self.nFilters, 3, padding=1, dilation=self.dilation[i]) for i in
             range(self.filterlayer)])

        self.feature_merge = nn.Conv2d(self.nFilters * (self.filterlayer+1), self.nFilters, 3, padding=1)

        self.adaptive_implicit_trans1 = adaptive_IDCT_trans(self.patchsize)

        self.feature_convert = nn.Conv2d(self.nFilters, out_ch, kernel_size=1, padding=0)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

    def forward(self, x):
        N, C, H, W = x.shape
        if C == 1:
            x_onechannel = x
        elif C == 3:
            x_onechannel = x[:, 1:-1, :, :]

        _x = self.conv01(x_onechannel)

        t = self.adaptive_implicit_trans01(_x)

        for level in range(self.filterlayer):
            _t = self.conv_lrelu[level](t)
            t = torch.cat([_t, t], dim=-3)

        t = self.feature_merge(t)
        t = self.lrelu(t)

        w = 1 / (1 + torch.sum(t, dim=1, keepdim=True))

        t = self.adaptive_implicit_trans1(t)
        t = w * t

        image_feat = t
        noise_feat = None

        return image_feat, noise_feat

class MDIVDnet(nn.Module):

    def __init__(self,
                 mid_channels=64,
                 num_blocks=15,
                 max_residue_magnitude=10,
                 spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                                   'basicvsr/spynet_20210409-c6c1bd09.pth',
                 cpu_cache_length=200,
                 downsampling_scale=4):

        super().__init__()
        self.dct_patchsize = 5
        self.mid_channels = mid_channels
        self.cpu_cache_length = cpu_cache_length
        self.downsampling_scale = downsampling_scale

        self.ldpp = LNPPBlock_rowcol()

        self.dct_feature = MFFBlock(in_ch=1, out_ch=self.mid_channels, dilation=[1, 2, 3, 2, 1], patchsize=self.dct_patchsize, pretrained=None)

        self.spynet = SPyNet(pretrained=None)

        if self.downsampling_scale == 0:
            self.feat_extract = ResidualBlocksWithInputConv(1, mid_channels, 5)
        elif self.downsampling_scale == 2:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))
        elif self.downsampling_scale == 4:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))
        elif self.downsampling_scale == 8:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)

        if self.downsampling_scale >= 2:
            self.upsample1 = PixelShufflePack(
                mid_channels, 64, 2, upsample_kernel=3)
        if self.downsampling_scale >= 4:
            self.upsample2 = PixelShufflePack(
                mid_channels, mid_channels, 2, upsample_kernel=3)
        if self.downsampling_scale >= 8:
            self.upsample3 = PixelShufflePack(
                mid_channels, mid_channels, 2, upsample_kernel=3)

        self.dct_feature_convert = nn.Conv2d(in_channels=self.dct_patchsize * self.dct_patchsize,
                                             out_channels=self.mid_channels,
                                             kernel_size=3,
                                             padding=1)

        self.feat_reconstruct = feat_reconstruct(filter_in=128, filter_out=1)

        # batchnorm
        self.batchnorm = nn.BatchNorm2d(self.mid_channels)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False


        filename = os.path.abspath(__file__)
        net_name = os.path.splitext(os.path.split(filename)[1])[0]
        print(net_name)

    def check_if_mirror_extended(self, lqs):

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1) #torch.Size([1, 128, 180, 320])
            feat_prop = feat_prop + self.backbone[module_name](feat)  #torch.Size([1, 64, 180, 320])
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats, feats_dct = None):

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)

            if self.downsampling_scale >= 8:
                hr = self.lrelu(self.upsample3(hr))
            if self.downsampling_scale >= 4:
                hr = self.lrelu(self.upsample2(hr))
            if self.downsampling_scale >= 2:
                hr = self.lrelu(self.upsample1(hr))

            if feats_dct != None:
                dct_feature = self.dct_feature_convert(feats_dct[:, i, :, :, :])

                hr = torch.cat([hr, dct_feature], dim=1)
            hr = self.feat_reconstruct(hr)

            hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()  #torch.Size([1, 21, 3, 180, 320])

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False


        lqs = self.ldpp(lqs)

        dct_feat, dct_noise_feat = self.dct_feature(lqs.view(n*t, c, h, w))
        dct_feat = dct_feat.view(n, t, self.dct_patchsize*self.dct_patchsize, h, w)
        if self.downsampling_scale == 0:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=1/self.downsampling_scale,
                mode='bicubic').view(n, t, c, h // self.downsampling_scale, w // self.downsampling_scale)
            dct_feat_downsample = F.interpolate(
                dct_feat.view(-1, self.dct_patchsize*self.dct_patchsize, h, w), scale_factor=1 / self.downsampling_scale,
                mode='bicubic').view(n, t, self.dct_patchsize*self.dct_patchsize, h // self.downsampling_scale, w // self.downsampling_scale)


        feats = {}

        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w)) #torch.Size([21, 64, 180, 320])
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w) #torch.Size([1, 21, 64, 180, 320])
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(dct_feat_downsample)

        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        result = self.upsample(lqs, feats, dct_feat)

        return result

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')



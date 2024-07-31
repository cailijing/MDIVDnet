import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from math import log10
from utils import strred


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def _binarize(y_data, threshold):

    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data


def ssim(img1, img2, window_size=11, size_average=False):
    _, channel, h, w = img1.size()

    img1_ = img1[:, :, 8: -8, 8: -8]
    img2_ = img2[:, :, 8: -8, 8: -8]
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1_.get_device())
    window = window.type_as(img1_)
    return _ssim(img1_, img2_, window, window_size, channel, size_average)

def psnr(img1, img2):
    # N C L H W

    img1 = (img1 * 255.0).int()
    img2 = (img2 * 255.0).int()
    img1 = img1.float() / 255.0
    img2 = img2.float() / 255.0
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 10 * log10(1 / mse)
    return psnr

def cal_ssim(video1, video2, window_size=11, size_average=False):
    N,C,L,H,W=np.shape(video1)
    ssim_avg=0
    for i in range(N):
       for j in range(L):
          frame_noisy=video1[i,:,j,:,:]
          frame_noisy=frame_noisy.view(1,C,H,W)
          frame_gt=video2[i,:,j,:,:]
          frame_gt=frame_gt.view(1,C,H,W)
          ssim_frame=ssim(frame_noisy.detach(),frame_gt.detach())
          ssim_avg=ssim_avg+ssim_frame.cpu().numpy()
    return ssim_avg/(N*L)

def cal_strred(video1,video2):
    N,C,L,H,W=np.shape(video1)
    video1=video1.view(C,L,H,W).permute(1,2,3,0).detach().cpu().numpy()
    video2=video2.view(C,L,H,W).permute(1,2,3,0).detach().cpu().numpy()
    if C == 3:
        video1_gray=np.zeros((L,H,W,1),dtype=np.float)
        video2_gray=np.zeros((L,H,W,1),dtype=np.float)
        for i in range(L):
            video1_gray[i,:,:,:]=np.expand_dims((video1[i,:,:,0]*0.2989+video1[i,:,:,1]*0.5870+video1[i,:,:,2]*0.1140),2)*255
            video2_gray[i,:,:,:]=np.expand_dims(video2[i,:,:,0]*0.2989+video2[i,:,:,1]*0.5870+video2[i,:,:,2]*0.1140,2)*255
    elif C == 1:
        video1_gray = video1
        video2_gray = video2
    _, strred_score, _ = strred(video1_gray, video2_gray)
    return strred_score

def calc_metrics(denoise_seq, gt):

    denoise_seq = denoise_seq.unsqueeze(dim=0).permute(0, 2, 1, 3, 4) #N C F H W
    gt = gt.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)
    PSNR = psnr(denoise_seq.detach(), gt.detach())
    SSIM = float(cal_ssim(denoise_seq, gt))
    STTRED = cal_strred(denoise_seq, gt)

    return PSNR, SSIM, STTRED

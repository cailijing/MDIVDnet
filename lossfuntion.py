from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def charbonnier_loss(pred, target, eps=1e-12):
        """Charbonnier loss.

        Args:
            pred (Tensor): Prediction Tensor with shape (n, c, h, w).
            target ([type]): Target Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Calculated Charbonnier loss.
        """
        return torch.sqrt((pred - target) ** 2 + eps)

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * self.charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)

class L2_LOSS(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L2_LOSS, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        square = torch.square(diff)
        loss = torch.sum(square) / X.size(0)
        return loss


class L1_LOSS(nn.Module):
    def __init__(self):
        super(L1_LOSS, self).__init__()
        self.eps = 1e-6

    def forward(self, Ximage, Ytarget):
        diff = torch.add(Ximage, -Ytarget)
        # square = torch.square(diff)
        abs = torch.abs(diff)
        loss = torch.sum(abs) / Ximage.size(0)
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        _reduction_modes = ['none', 'mean', 'sum']
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def charbonnier_loss(self, pred, target, eps=1e-12):
        """Charbonnier loss.

        Args:
            pred (Tensor): Prediction Tensor with shape (n, c, h, w).
            target ([type]): Target Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Calculated Charbonnier loss.
        """
        return torch.sqrt((pred - target) ** 2 + eps)

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = self.loss_weight * self.charbonnier_loss(
            pred,
            target,
            eps=self.eps)
        loss = torch.mean(loss)

        return loss


class L1_Sobel_Loss_C3(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(L1_Sobel_Loss_C3, self).__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3, 3, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3, 3, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)


        # diff = torch.add(edge_outputs, -edge_Y)
        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(error) / outputs.size(0) ##output.size(0)은 batch size
        return loss

class L1_Sobel_Loss_C1(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(L1_Sobel_Loss_C1, self).__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(1, 1, 3, bias=False)
        self.conv_op_y = nn.Conv2d(1, 1, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)


        # diff = torch.add(edge_outputs, -edge_Y)
        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(error) / outputs.size(0) ##output.size(0)은 batch size
        return loss

# class L1_Advanced_Sobel_Loss(nn.Module):
#     def __init__(self, device=torch.device('cuda')):
#         super().__init__()
#         self.device = device
#         self.conv_op_x = nn.Conv2d(1,1, 3, bias=False)
#         self.conv_op_y = nn.Conv2d(1,1, 3, bias=False)
#
#         sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0,-1, -2]],
#                                    [[2, 1, 0], [1, 0, -1], [0,-1, -2]],
#                                    [[2, 1, 0], [1, 0, -1], [0,-1, -2]]], dtype='float32')
#         sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
#                                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
#                                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
#         sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
#         sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
#
#         self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
#         self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
#         self.conv_op_x.weight.requires_grad = False
#         self.conv_op_y.weight.requires_grad = False
#
#     # def forward(self, edge_outputs, image_target):
#     def forward(self, outputs, image_target):
#         edge_Y_xoutputs = self.conv_op_x(outputs)
#         edge_Y_youtputs = self.conv_op_y(outputs)
#         edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)
#
#         edge_Y_x = self.conv_op_x(image_target)
#         edge_Y_y = self.conv_op_y(image_target)
#         edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)
#
#         diff = torch.add(edge_Youtputs, -edge_Y)
#         error = torch.abs(diff)
#         loss = torch.sum(error) / outputs.size(0)
#         return loss

class edge_making(nn.Module):
    def __init__(self):
        super(edge_making, self).__init__()
        self.conv_op_x, self.conv_op_y = self.make_sobel_layer()

    def forward(self, output):
        output = (output*2)-1
        edge_X_x = self.conv_op_x(output)
        edge_X_y = self.conv_op_y(output)
        edge_X = torch.abs(edge_X_x) + torch.abs(edge_X_y)
        return edge_X

    def make_sobel_layer(self):
        conv_op_x = nn.Conv2d(3, 1, 3, bias=False)
        conv_op_y = nn.Conv2d(3, 1, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        conv_op_x.weight.requires_grad = False
        conv_op_y.weight.requires_grad = False

        return conv_op_x, conv_op_y




class Sobelloss_L1(nn.Module):
    """edge_loss"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, image, target, cuda=True):
        x_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # x_filter3 = np.zeros_like(x_filter)
        # y_filter3 = np.zeros_like(y_filter)
        x_filter3 = np.zeros((3,3,3))
        y_filter3 = np.zeros((3,3,3))
        x_filter3[:,:,0] = x_filter
        x_filter3[:,:,1] = x_filter
        x_filter3[:,:,2] = x_filter
        y_filter3[:,:,0] = y_filter
        y_filter3[:,:,1] = y_filter
        y_filter3[:,:,2] = y_filter

        convx = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        convy = nn.Conv2d(3, 3, kernel_size=3 , stride=1, padding=1, bias=False)
        weights_x = torch.from_numpy(x_filter3).float().unsqueeze(0).unsqueeze(0)
        weights_y = torch.from_numpy(y_filter3).float().unsqueeze(0).unsqueeze(0)

        if cuda:
            weights_x = weights_x.cuda()
            weights_y = weights_y.cuda()

        convx.weight = nn.Parameter(weights_x)
        convy.weight = nn.Parameter(weights_y)

        convx.weight.requires_grad = False
        convy.weight.requires_grad = False

        g1_x = convx(image)
        g2_x = convx(target)
        g1_y = convy(image)
        g2_y = convy(target)

        g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
        g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

        loss = torch.sqrt((g_1 - g_2).pow(2))
        loss = torch.sum(loss) / image.size(0)

        # return torch.mean((g_1 - g_2).pow(2)) # L2, MSE loss
        # return torch.sqrt((g_1 - g_2).pow(2)) # L1sobel loss
        return loss # L1sobel loss


# class L1_Charbonnier_loss(nn.Module):
#     """L1 Charbonnierloss."""
#     def __init__(self):
#         super(L1_Charbonnier_loss, self).__init__()
#         self.eps = 1e-6
#     def forward(self, X, Y):
#         diff = torch.add(X, -Y)
#         error = torch.sqrt(diff * diff + self.eps)
#         loss = torch.sum(error) / X.size(0)
#         return loss




# def loss_function(recon_x, x, mu, logvar):
#     """
#     recon_x: generating images
#     x: origin images
#     mu: latent mean
#     logvar: latent log variance
#     """
#     BCE = reconstruction_function(recon_x, x)  # mse loss
#     # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.sum(KLD_element).mul_(-0.5)
#     # KL divergence
#     return BCE + KLD


# def edge_loss(out, target, cuda=True):
#     x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#     y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     convx = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding=1, bias=False)
#     convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding=1, bias=False)
#     weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
#     weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)
#
#     if cuda:
#         weights_x = weights_x.cuda()
#         weights_y = weights_y.cuda()
#
#     convx.weight = nn.Parameter(weights_x)
#     convy.weight = nn.Parameter(weights_y)
#
#     g1_x = convx(out)
#     g2_x = convx(target)
#     g1_y = convy(out)
#     g2_y = convy(target)
#
#     g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
#     g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))
#
#     return torch.mean((g_1 - g_2).pow(2))

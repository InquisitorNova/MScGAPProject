"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 
https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock. The ResBlock has been modified to include GroupNorm layers
and Gated Convolutions. The ResBlock is based on the ResBlock introduced in the paper:
https://arxiv.org/abs/1806.03589. The GatedUNet is designed to allow the network to control 
the information that flows through the network. The GatedUNet is designed to be used with 2D image data.

The original MIT License is as follows:

MIT License

Copyright (c) 2017 Jackson Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl

from torch.nn import init
import transformers
import numpy as np

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def find_group_size(out_channels, min_group_channels=4, max_group_channels=32):
    # Start from the number of channels and decrement to find the largest valid group size
    for num_groups in range(out_channels, 0, -1):
        group_channels = out_channels // num_groups
        if out_channels % num_groups == 0 and min_group_channels <= group_channels <= max_group_channels:
            return num_groups
    # If no valid number
    return 1

class GatedConvolutionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedConvolutionGate, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.gate = conv3x3(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        gate = self.sigmoid(self.gate(x))
        return out * gate

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        n_groups = find_group_size(F_int)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(n_groups, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(n_groups,F_int)
        )

        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1,1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, n_groups = 32):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.gated_convolution_gate = GatedConvolutionGate(self.out_channels, self.out_channels)

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)
        #self.BatchNorm_1 = nn.BatchNorm2d(self.out_channels)
        #self.BatchNorm_2 = nn.BatchNorm2d(self.out_channels)

        self.GroupNorm_1 = nn.GroupNorm(n_groups, self.out_channels)
        self.GroupNorm_2 = nn.GroupNorm(n_groups, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_skip = self.conv1(x)

        x = self.conv2(x_skip)
        x = self.GroupNorm_1(x)
        x = F.leaky_relu(x)

        x = self.conv3(x) + self.gated_convolution_gate(x_skip)
        x = self.GroupNorm_2(x)
        x = F.leaky_relu(x)

        before_pool = x 
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, n_groups = 32,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.n_groups = n_groups
        self.attention_gate = AttentionGate(self.out_channels, self.out_channels, self.out_channels // 2)
        self.gated_convolution_gate = GatedConvolutionGate(2 * self.out_channels if self.merge_mode == 'concat' else self.out_channels, self.out_channels)

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)

        self.GroupNorm_1 = nn.GroupNorm(self.n_groups, self.out_channels)
        self.GroupNorm_2 = nn.GroupNorm(self.n_groups, self.out_channels)

    def forward(self, from_down, from_up):
        """" Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        from_down = self.attention_gate(from_up, from_down)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x_skip = self.gated_convolution_gate(x)

        x = self.conv2(x_skip)
        x = self.GroupNorm_1(x)
        x = F.leaky_relu(x)

        x = self.conv3(x) + x_skip
        x = self.GroupNorm_2(x)
        x = F.leaky_relu(x)

        return x


class GatedResUNet(pl.LightningModule):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, levels, channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='add', dataset = None, val_dataset = None,
                 Learning_Rate = 0.001, Warm_Up_Epochs = 10, Epochs = 100, 
                 mini_batches = 100):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        self.save_hyperparameters()
        
        super(GatedResUNet, self).__init__()
        
        
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.levels = levels
        self.channels = channels
        self.start_filts = start_filts
        self.depth = depth
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.Learning_Rate = Learning_Rate
        self.Warm_Up_Epochs = Warm_Up_Epochs
        self.Epochs = Epochs
        self.mini_batches = mini_batches

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.channels * self.levels if i == 0 else outs
            outs = self.start_filts*(2**i)
            n_groups = find_group_size(outs)
            #print(n_groups, outs)
#            outs = self.start_filts
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, n_groups = n_groups)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            n_groups = find_group_size(outs)
#            outs = ins
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode, n_groups = n_groups)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        epsilon = 1
        
        stack = None
        
        factor = 10.0
        for i in range (self.levels):
            scale = x.clone()*(factor**(-i))
            scale = torch.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = torch.cat((stack,scale),1)
        
        x = stack
        
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        x = self.conv_final(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def photonLoss(self,result, target):
        expEnergy = torch.exp(result)
        perImage =  -torch.mean(result*target, dim =(-1,-2,-3), keepdim = True )
        perImage += torch.log(torch.mean(expEnergy, dim =(-1,-2,-3), keepdim = True ))*torch.mean(target, dim =(-1,-2,-3), keepdim = True )
        return torch.mean(perImage)
    
    def MSELoss(self,result, target):
        expEnergy = torch.exp(result)
        expEnergy /= (torch.mean(expEnergy, dim =(-1,-2,-3), keepdim = True ))
        target = target / (torch.mean(target, dim =(-1,-2,-3), keepdim = True ))
        return torch.mean((expEnergy-target)**2)
    
    def training_step(self, batch, batch_idx = None):
        img_input, _, target_img  = batch
        predicted = self.forward(img_input)
        train_loss = self.photonLoss(predicted, target_img)
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss

    def validation_step(self, batch, batch_idx = None):
        img_input, _, target_img = batch
        predicted = self.forward(img_input)
        valid_loss = self.photonLoss(predicted, target_img)
        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss

    def test_step(self, batch, batch_idx = None):
        img_input, _, target_img = batch
        predicted = self.forward(img_input)
        test_loss = self.photonLoss(predicted, target_img)
        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss
    
    def configure_optimizers(self):
        num_warm_steps = self.mini_batches * self.Warm_Up_Epochs
        num_training_steps = self.mini_batches * self.Epochs
        optimizer = optim.AdamW(self.parameters(), lr=self.Learning_Rate, weight_decay = 1e-4)

        Scheduler_1 = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.Learning_Rate, total_steps = num_training_steps, 
                                                      epochs = self.Epochs, pct_start = 0.1, anneal_strategy = "cos", 
                                                      div_factor = 10.0, final_div_factor = 1.0)
        
        Scheduler_1 = {
            'scheduler': Scheduler_1,
            "interval": "step",
            "monitor": "train_loss",
            "frequency":1
        }

        return [optimizer], [Scheduler_1]
"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 
https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock.

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

# Import relevant modules:
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

def find_group_number(channels, min_group_channels = 4, max_group_channels = 32):
    """
    This function is designed to find the largest valid group number for the ResBlock GroupNorm Layer:
    channels - The number of channels in the current image representation.
    min_group_channels - The minimum number of channels per group.
    ma_group_channels - The maximum number of channels per group.
    """

    # Start from the number of channels and decrease the number of groups
    # until the number of channels per group is less than the maximum number of channels per group.

    for num_groups in range(channels, 0, -1):
        group_channels = channels // num_groups
        if channels % num_groups == 0 and group_channels >= min_group_channels and group_channels <= max_group_channels:
            return num_groups
    
    # Return 1 if no valid number exists.
    return 1


class Gated_Convolution(nn.Module):
    """
    Defines the Gated Convolutional Layer used in the ResBlock. This layer
    is designed to allow the network to control the information that flows
    through the network, as described in the paper: https://arxiv.org/abs/1806.03589
    in_channels = The number of channels in the input image representation.
    out_channels = The number of channels in the output image representation.
    """
    def __init__(self, in_channels, out_channels):
        super(Gated_Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x) * F.sigmoid(self.gate(x))


class AttentionGate(nn.Module):
    """
    A mechanism introduced as described in the AttnUNet paper which uses
    the outputs from the previous decoder layer and the encoder layer to
    create a gating mechanism that allows the network to focus on the most
    important features.
    F_g = The number of channels in the encoder layer.
    F_l = The number of channels in the decoder layer.
    F_int = The number of channels in the intermediate layer.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # Initialise the layers of the block. 
        n_groups = find_group_number(F_int)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(n_groups, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(n_groups,F_int)
        )
        
        self.W_psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1,1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.W_psi(psi)
        return x * psi

class DownConvolutionBlock(nn.Module):
    """
    For the basic UNet, this block is used to downsample the input image representation.
    It consists of convolutional layers supported by normalisation layers and non-linear activation
    functions.
    in_channels = The number of channels in the input image representation.
    out_channels = The number of channels in the output image representation.
    pooling = A boolean value that determines whether pooling is used.
    n_groups = The number of groups used in the GroupNorm layers.
    num_layers = The number of convolutional layers in the block.
    activation = The activation function used in the block.
    dropout_rate = The dropout rate used in the block.
    """

    def __init__(self, in_channels, out_channels, pooling = True, num_layers = 2,
                 activation = nn.LeakyReLU(), dropout_rate = 0.5):
        super(DownConvolutionBlock, self).__init__()

        # Initialise the layers of the block.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Create the layers of the block.
        n_groups = find_group_number(out_channels, min_group_channels = 4, max_group_channels = 32)

        # The Skip Block provides the input and skip connection to the ResBlock.
        self.Skip_Block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.GroupNorm(n_groups, out_channels),
            activation,
        )

        # The Main Processing Block is the main part of the block.
        Main_Processing_Block = []
        
        Main_Processing_Block.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)

        Main_Processing_Block.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)
        
        self.Main_Processing_Block = nn.Sequential(*Main_Processing_Block)

        # If pooling is required, add a pooling layer.
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # If dropout is required, add a dropout layer.
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.gated_convolution_gate = Gated_Convolution(out_channels, out_channels)
    
    def forward(self, x):
        x_skip = self.gated_convolution_gate(self.Skip_Block(x))

        x = self.Main_Processing_Block(x_skip)

        if self.dropout_rate is not None:
            x = self.dropout(x)

        x+= x_skip

        before_pool = x
        if self.pooling:
            x = self.pool(x)
            return x, before_pool
        else:
            return x

class UpConvolutionBlock(nn.Module):
    """
    For the basic UNet, this block is used to upsample the input image representation.
    It consists of convolutional layers supported by normalisation layers and non-linear activation
    functions.
    in_channels = The number of channels in the input image representation.
    out_channels = The number of channels in the output image representation.
    merge_mode = The method used to merge the input and upsampled image representations.
    up_mode = The method used to upsample the input image representation.
    n_groups = The number of groups used in the GroupNorm layers.
    num_layers = The number of convolutional layers in the block.
    activation = The activation function used in the block.
    dropout_rate = The dropout rate used in the block.
    """

    def __init__(self, in_channels, out_channels, merge_mode = "concat", upsampling = True,
                 up_mode = "upsample", num_layers = 2, activation = nn.LeakyReLU(),
                 dropout_rate = 0.5):
        super(UpConvolutionBlock, self).__init__()

        # Initialise the layers of the block.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.upsampling = upsampling

        # Create the layers of the block.
        n_groups = find_group_number(out_channels, min_group_channels = 4, max_group_channels = 32)

        # The Skip Block provides the input and skip connection to the ResBlock.
        self.Skip_Block = nn.Sequential(
            nn.Conv2d(2*out_channels if merge_mode == "concat" and self.upsampling else out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.GroupNorm(n_groups, out_channels),
            activation,
        )

        # The Main Processing Block is the main part of the block.
        Main_Processing_Block = []

        Main_Processing_Block.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)

        Main_Processing_Block.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)

        self.Main_Processing_Block = nn.Sequential(*Main_Processing_Block)

        # If dropout is required, add a dropout layer.
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        
        # If upsampling is required, add an upsampling layer.
        if up_mode == "transpose" and upsampling:
            self.UpConvolutionBlock = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
                nn.GroupNorm(n_groups, out_channels),
                activation,
            ]
        elif up_mode == "upsample" and upsampling:
            self.UpConvolutionBlock = nn.Sequential(
                nn.Upsample(mode = "bilinear", scale_factor = 2, align_corners = True),
                nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                nn.GroupNorm(n_groups, out_channels),
                activation,
            )
        
        if upsampling:
            self.UpConvolutionBlock = nn.Sequential(*self.UpConvolutionBlock)

        self.attention_gate = AttentionGate(out_channels, out_channels, out_channels)
        self.gated_convolution_gate = Gated_Convolution(out_channels, out_channels)

    def forward(self, below, above):
        # Begin by matching upsampling the relevant layers
        # and merging them.
        if self.upsampling:
            below = self.UpConvolutionBlock(below)
            below = self.attention_gate(above, below)
            if self.merge_mode == "concat":
                x = torch.cat((below, above), 1)
            else:
                x = below + above
        else:
            x = below

        x_skip = self.Skip_Block(x)
        x = self.gated_convolution_gate(x_skip)

        x = self.Main_Processing_Block(x)

        x += x_skip

        if self.dropout_rate is not None:
            x = self.dropout(x)

        return x
    
class GatedUNet(pl.LightningModule):
    """
    The ResUNet class is based on the original UNet class introduced in
    https://arxiv.org/abs/1505.04597. The ResUNet class is designed
    to be incorporated within the GAP Framework. It is a fully-convolutional
    neural network with skip connections that allow for information to travel
    between the encoder and decoder pathways making up the network. The encoder
    pathway is designed to downsample the input image representation, while the
    decoder pathway is designed to upsample the input image representation. As the image
    is downsampled, spatial resolution is decreased while the number of channels is increased.
    As the image is upsampled, spatial resolution is increased while the number of channels is decreased.
    The ResUNet class is designed to be used with 2D image data.
    in_channels = The number of channels in the input image representation.
    """

    def __init__(
            self,
            levels: int = 10,
            channels: int = 3,
            depth: int = 5,
            up_mode: str = "upsample",
            merge_mode: str = "concat",
            num_layers: int = 2,
            activation: nn.Module = nn.LeakyReLU(),
            dropout_rate: float = 0.5,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            starting_filters: float = 32,
            num_blocks_per_layer: int = 2,
            epochs = 40,
            mini_batches = 512,
            warm_up_epochs = 10,
            average_face: torch.Tensor = None,
            device = "cuda"):
    
        self.save_hyperparameters()
        super(GatedUNet, self).__init__()

        if up_mode != "upsample" and up_mode != "transpose":
            raise ValueError("The up_mode parameter must be either 'upsample' or 'transpose'.")
        
        if merge_mode != "concat" and merge_mode != "add":
            raise ValueError("The merge_mode parameter must be either 'concat' or 'add'.")
        
        # Initialise the parameters of the ResUNet.
        self.levels = levels
        self.channels = channels
        self.depth = depth
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.starting_filters = starting_filters
        self.num_blocks_per_layer = num_blocks_per_layer
        self.epochs = epochs
        self.warm_up_epochs = warm_up_epochs
        self.mini_batches = mini_batches
        self.average_face = average_face
        self.eps = torch.Tensor([1e-6])

        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()
        
        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()
        
        # Define the Encoder Blocks.
        for index in range(depth):
            in_channels = channels * (levels+1) if index == 0 else out_channels
            out_channels = self.starting_filters if index == 0 else out_channels * 2  
            for block in range(num_blocks_per_layer):
                if block != self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                             pooling = False, num_layers = num_layers, 
                                                             activation = activation, dropout_rate = dropout_rate,
                                                             ))
                elif block == self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                         pooling = True, num_layers = num_layers, 
                                                         activation = activation, dropout_rate = dropout_rate))
                    
                in_channels = out_channels
                    
        # Create the Bottleneck Pathway.
        self.Bottleneck_Block = nn.ModuleList()
        pooling = False

        self.Bottleneck_Block.append(DownConvolutionBlock(out_channels, 2*out_channels,
                                                            pooling = pooling, num_layers = num_layers,
                                                            activation = activation, dropout_rate = dropout_rate))


        out_channels = out_channels*2
        # Create the Decoder Pathway.
        self.decoder = nn.ModuleList()
        for index in range(depth):
            in_channels = out_channels
            out_channels = in_channels // 2

            for block in range(num_blocks_per_layer):
                if block == 0:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = True,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate,
                                                        ))
                else:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = False,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate))
                in_channels = out_channels
                       
                    
        # Create the Output Layers.
        self.output = nn.Conv2d(out_channels, channels, kernel_size = 1)
        self.reset_parameters()

    @staticmethod
    def weight_init(m):

        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, nonlinearity = "leaky_relu")
            if m.bias is not None:
                init.constant(m.bias, 0)

        if isinstance(m, nn.GroupNorm):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

        if isinstance(m, nn.BatchNorm1d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

        if isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight, nonlinearity = "leaky_relu")
            if m.bias is not None:
                init.constant(m.bias, 0)

    def reset_parameters(self):
        for _, m in enumerate(self.modules()):
            if not isinstance(m, nn.LazyLinear):
                self.weight_init(m)

    def forward(self, x):

        stack = None
        factor = 10.0

        # Begin by encoding the input image representation with a 
        # sineusodial encoding.
        out = x.clone()
        for index in range(self.levels):
            scale = out*(factor**(-index))
            sin_encoding = torch.sin(scale)
            if stack is None:
                stack = sin_encoding
            else:
                stack = torch.cat((stack, sin_encoding), 1)
        
        x = stack
        x = torch.cat((x, self.average_face.repeat(x.size(0), 1, 1, 1)), 1)
        encoder = {}

        current_depth = 0
        # Begin by encoding the input image representation.
        for index, Encoder_Block in enumerate(self.encoder):
            out = Encoder_Block(x)

            if len(out) == 2:
                x, before_pool = out
                encoder[current_depth] = before_pool
                current_depth += 1
            else:
                x = out
        
        # Pass the encoding through the bottleneck of the networ
        for index, Bottleneck_Block in enumerate(self.Bottleneck_Block):
            x = Bottleneck_Block(x)
        
        # Begin by decoding the input image representation.
        index = 0
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool)
                current_depth -= 1
                index+=1
            else:
                x = Decoder_Block(x, None)
                index+=1
    
        # Pass the image representation through the output layer.
        output = self.output(x)
        return output

    def predict(self, x):
        return self.forward(x)
    
    def photonLoss(self,result, target):
        expEnergy = torch.exp(result)
        perImage =  -torch.mean(result*target, dim =(-1,-2,-3), keepdims = True )
        perImage += torch.log(torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))*torch.mean(target, dim =(-1,-2,-3), keepdims = True )
        return torch.mean(perImage)
    
    def MSELoss(self,result, target):
        expEnergy = torch.exp(result)
        expEnergy /= (torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))
        target = target / (torch.mean(target, dim =(-1,-2,-3), keepdims = True ))
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
        num_warm_steps = self.mini_batches * self.warm_up_epochs
        num_training_steps = self.mini_batches * self.epochs
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = 1e-4)

        Scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.learning_rate, total_steps = num_training_steps, 
                                                      epochs = self.epochs, pct_start = 0.1, anneal_strategy = "cos", 
                                                      div_factor = 10.0, final_div_factor = 1.0)
        
        Scheduler = {
            'scheduler': Scheduler,
            "interval": "step",
            "monitor": "train_loss",
            "frequency":1
        }

        return [optimizer], [Scheduler]
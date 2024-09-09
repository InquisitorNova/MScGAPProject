"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 
https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock. This implementation includes the use of GroupNorm layers
and the ability to use different numbers of channels in the ResBlock. The ResBlock is designed
with the incorporation of Gated Convolutional layers, Attention Gates and Hierarchical Decoders.
The ResBlock is designed to be used with 2D image data. This UNet is inspired by the
following papers:

1. "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al.
2. "Attention U-Net: Learning Where to Look for the Pancreas" by Oktay et al.
3. "Free-Form Image Inpainting with Gated Convolution by Yu et al.
4. "Progressive residual networks for image super-resolution" by Jin Wan et al.
5. "Dynamic Convolution: Attention over Convolution Kernels" by Chen et al.
6. "On The Power of Curriculum Learning in Training Deep Networks" by Hacohen et al.
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
import math
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

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

class Dynamic_Sequential(nn.Module):
    def __init__(self, *layers):
        super(Dynamic_Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, condition, temperature):
        for layer in self.layers:
            if isinstance(layer, DynamicConv2d):
                x = layer(x, condition, temperature)
            else:
                x = layer(x)
        return x

class CurriculumScheduling:
    # Inspired by the work of Hacohen et al. on Curriculum Learning:
    # https://arxiv.org/abs/1904.03626
    def __init__(self, dataset, val_dataset,initial_min_psnr, target_min_psnr, max_psnr, epochs_per_phase, total_epochs):
        super(CurriculumScheduling, self).__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.initial_min_psnr = initial_min_psnr
        self.target_min_psnr = target_min_psnr
        self.max_psnr = max_psnr
        self.epochs_per_phase = epochs_per_phase
        self.total_epochs = total_epochs
        self.current_phase = 0
        
        # Calculate phase count and PSNR step for each phase
        self.phase_count = math.ceil(total_epochs / epochs_per_phase)
        self.psnr_step = (target_min_psnr - initial_min_psnr) / self.phase_count

    def update_psnr_range(self):
        # Ensure we're not exceeding the phase count
        if self.current_phase < self.phase_count:
            self.current_phase += 1
            new_min_psnr = self.initial_min_psnr + self.current_phase * self.psnr_step

            if self.psnr_step > 0:  
                self.dataset.minPSNR = min(self.target_min_psnr, new_min_psnr)
                self.val_dataset.minPSNR = min(self.target_min_psnr, new_min_psnr)
            else:
                self.dataset.minPSNR = max(self.target_min_psnr, new_min_psnr)
                self.val_dataset.minPSNR = max(self.target_min_psnr, new_min_psnr)

            print(f"Current PSNR Range: {self.dataset.minPSNR} - {self.dataset.maxPSNR}")
            print(f"Current Phase: {self.current_phase}/{self.phase_count}")
            print(f"Psnr_Step {self.psnr_step}")
            print("Target PSNR: ", self.target_min_psnr)
        else:
            print("Final phase reached, no further PSNR adjustments.")

    def get_current_phase(self):
        return self.current_phase

class NoiseEstimator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NoiseEstimator, self).__init__()

        self.conv1x1 = nn.Conv2d(2*in_channels, 16, kernel_size = 1, stride = 1, padding = 0)
        self.conv3x3 = nn.Conv2d(2*in_channels, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv5x5 = nn.Conv2d(2*in_channels, 16, kernel_size = 5, stride = 1, padding = 2)

        self.groupnorm1 = nn.GroupNorm(find_group_number(48), 48)
        self.conv2 = nn.Conv2d(48, 64, 3, 1)
        self.groupnorm2 = nn.GroupNorm(find_group_number(64), 64)
        self.conv3 = nn.Conv2d(64, out_channels, 3, 1)

    def forward(self, ImageNoise, ReferenceImage):
        x = torch.cat([ImageNoise, ReferenceImage], dim=1)

        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        
        x = F.leaky_relu(self.groupnorm1(torch.cat([x1, x2, x3], dim=1)))
        x = F.leaky_relu(self.groupnorm2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))
        return x

class HierarchicalResBlock(nn.Module):
    # This block is inspired by the Progressive Hierarchical Learning introduced in the paper:
    # 'Progressive residual networks for image super-resolution’, by Jin Wan et al.
    def __init__(self,hidden_dim, num_outputs):
        super(HierarchicalResBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, padding = 1)
        self.fc = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_outputs),
            nn.Softmax(dim = -1)
        )

    def forward(self, output, intermediate_outputs, noise):
        
        avg_noise_context = torch.mean(noise, dim = 1, keepdims = True)
        meax_noise_context, _ = torch.max(noise, dim = 1, keepdims = True)
        noise_context = torch.cat([avg_noise_context, meax_noise_context], dim = 1)
        noise_context = torch.sigmoid(self.conv1(noise_context))
        noise = noise * noise_context

        noise = noise.mean(dim = [2,3])

        weights = self.fc(noise)
        #print(len(intermediate_outputs))
        upsampled = F.interpolate(intermediate_outputs[0], size = output.shape[2:], mode = 'bilinear')
        #print(upsampled.shape, weights[:,0].unsqueeze(-1).unsqueeze(-1).shape)
        out = upsampled * weights[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        Intermediates = [upsampled]

        for index in range(1, len(intermediate_outputs)):
            upsampled = F.interpolate(intermediate_outputs[index], size = output.shape[2:], mode = 'bilinear')
            out = out + upsampled * weights[:,index].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            Intermediates.append(upsampled)

        return out + output, Intermediates

class Gated_Convolution(nn.Module):
    """
    Defines the Gated Convolutional Layer used in the ResBlock. This layer
    is designed to allow the network to control the information that flows
    through the network, as described in the paper: https://arxiv.org/abs/1806.03589
    in_channels = The number of channels in the input image representation.
    out_channels = The number of channels in the output image representation.
    """
    def __init__(self, in_channels, out_channels, channels= 3):
        super(Gated_Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(in_channels+channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, refined_noise):
        return self.conv(x) * torch.sigmoid(self.gate(torch.cat([x, refined_noise], dim=1)))

class AttentionGate(nn.Module):
    """
    A mechanism introduced as described in the AttnUNet paper which uses
    the outputs from the previous decoder layer and the encoder layer to
    create a gating mechanism that allows the network to focus on the most
    important features.
    F_g = The number of channels in the encoder layer.
    F_l = The number of channels in the decoder layer.
    F_int = The number of channels in the intermediate layer.
    https://arxiv.org/abs/1804.03999
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

def softmax_temperature(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

def Obtain_Temperature(epoch, num_epochs, start_temperature, end_temperature):
    return start_temperature + (end_temperature - start_temperature) * (epoch / num_epochs)

# Define the Dynamic Convolution 2d Layer:
class Attention_Block(nn.Module):
    #https://arxiv.org/abs/1912.03458
    def __init__(self, bottleneck_units, out_channels, in_channels, noise_channels = 3):
        super(Attention_Block, self).__init__()
        
        # Initialise Hyperparameters:
        self.bottleneck_units = bottleneck_units
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.noise_channels = noise_channels

        # Define the conditioning network which takes the refined noise as input:
        self.Conditioning_Network = nn.Sequential(
            nn.Conv2d(noise_channels, bottleneck_units, kernel_size = 3, padding = 1),
            nn.GroupNorm(find_group_number(bottleneck_units), bottleneck_units),
            nn.LeakyReLU(),
        )

        # Define the SE Block layers:
        self.Dense = nn.Linear(in_channels, bottleneck_units)
        self.Dense_2 = nn.Linear(bottleneck_units, out_channels)
        self.GlobalPool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, noise_map, temperature):

        noise_background = self.Conditioning_Network(noise_map)
        noise_context = noise_background.mean(dim = (2, 3))

        x = self.GlobalPool(x)
        x = x.view(x.size(0), -1)

        x = self.Dense(x) + noise_context
        x = F.leaky_relu(x)
        x = self.Dense_2(x)

        x = softmax_temperature(x, temperature)
        return x
    
class DynamicConv2d(nn.Module):
    # This layer is inspired by the work of Chen et al. on Dynamic Convolution:
    # https://arxiv.org/abs/1912.03458
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1, reduction_factor=16, 
                 num_kernels=4, dilation=1, bias=True, noise_channels = 3):
        super(DynamicConv2d, self).__init__()

        # Softmax Parameters
        self.bottleneck_units = in_channels // reduction_factor
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.Entropy = 0.0

        # Convolution Parameters
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.eps = 1e-6
        
        self.Parallel_Kernels = nn.Parameter(torch.randn(num_kernels, out_channels, in_channels//groups, kernel_size, kernel_size), requires_grad=True)

        if self.bias:
            self.Parallel_Biases = nn.Parameter(torch.randn(num_kernels, out_channels))
        else:
            self.Parallel_Biases = None

        # Attention Network
        self.Attention = Attention_Block(
            bottleneck_units=self.bottleneck_units,
            out_channels=num_kernels,
            in_channels=in_channels,
            noise_channels= noise_channels
        )

    def forward(self, x, noise_map, temperature):
        batch_size, _, height, width = x.size()
        
        # Compute Attention Weights
        Attention_Scores = self.Attention(x, noise_map, temperature)

        # Compute the negative entropy for the attention scores
        Negative_Entropy = -torch.sum(Attention_Scores * torch.log(Attention_Scores + self.eps), dim = 1)
        entropy_value = -torch.mean(Negative_Entropy)
        self.Entropy = entropy_value.item() + self.Entropy

        # Aggregate the kernels and biases
        x = x.view(1, -1, height, width)
        Collapsed_Weights = self.Parallel_Kernels.view(self.num_kernels, -1)
        Kernels = torch.matmul(Attention_Scores, Collapsed_Weights).view(batch_size*self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            Aggregated_Biases = torch.matmul(Attention_Scores, self.Parallel_Biases).view(batch_size*self.out_channels)
            Biases = Aggregated_Biases.view(batch_size*self.out_channels)
        else:
            Biases = None

        # Apply the Convolution
        output = F.conv2d(x, Kernels, bias=Biases, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        return output
    
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
                 activation = nn.LeakyReLU(), dropout_rate = 0.5, kernel_size = 3,
                 stride = 1, padding = 1, reduction_factor = 16, num_kernels = 4):
        super(DownConvolutionBlock, self).__init__()

        # Initialise the layers of the block.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.reduction_factor = reduction_factor
        self.num_kernels = num_kernels

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
        
        Main_Processing_Block.append(
            DynamicConv2d(out_channels, out_channels, kernel_size = 3, padding = 1, 
                          groups = n_groups, reduction_factor = reduction_factor, 
                          num_kernels = num_kernels)
        )
        
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)

        Main_Processing_Block.append(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            )
        
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)
        
        self.Main_Processing_Block = Dynamic_Sequential(*Main_Processing_Block)

        # If pooling is required, add a pooling layer.
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # If dropout is required, add a dropout layer.
        if dropout_rate is not None:
            self.dropout = nn.Dropout2d(dropout_rate)
        
        self.gated_convolution_gate = Gated_Convolution(out_channels, out_channels)
    
    def forward(self, x, refined_noise, temperature):
        refined_noise = F.interpolate(refined_noise, size = x.shape[2:], mode = 'bilinear')
        x_skip = self.Skip_Block(x)

        x = self.Main_Processing_Block(x_skip, refined_noise, temperature)

        x = x + self.gated_convolution_gate(x_skip, refined_noise)

        if self.dropout is not None:
            x = self.dropout(x)

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
                 dropout_rate = 0.5, kernel_size = 3, stride = 1, padding = 1,
                 reduction_factor = 16, num_kernels = 4):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.reduction_factor = reduction_factor
        self.num_kernels = num_kernels

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
       
        Main_Processing_Block.append(
            DynamicConv2d(out_channels, out_channels, kernel_size = 3, padding = 1, 
                          groups = n_groups, reduction_factor = reduction_factor, 
                          num_kernels = num_kernels)
        )
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)

        Main_Processing_Block.append(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        )
        Main_Processing_Block.append(nn.GroupNorm(n_groups, out_channels))
        Main_Processing_Block.append(activation)

        self.Main_Processing_Block = Dynamic_Sequential(*Main_Processing_Block)

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

    def forward(self, below, above, refined_noise, temperature):
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
        refined_noise = F.interpolate(refined_noise, size = x_skip.shape[2:], mode = 'bilinear')

        x = self.Main_Processing_Block(x_skip, refined_noise, temperature)

        x = x + self.gated_convolution_gate(x_skip, refined_noise)

        if self.dropout_rate is not None:
            x = self.dropout(x)

        return x

class HierarchicalUNet(pl.LightningModule):
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
    levels = The number of levels in the UNet.
    channels = The number of channels in the input image representation.
    depth = The depth of the UNet.
    up_mode = The method used to upsample the input image representation.
    merge_mode = The method used to merge the input and upsampled image representations.
    num_layers = The number of convolutional layers in the block.
    activation = The activation function used in the block.
    dropout_rate = The dropout rate used in the block.
    learning_rate = The learning rate used in the block.
    weight_decay = The weight decay used in the block.
    starting_filters = The number of filters used in the first convolutional layer.
    num_blocks_per_layer = The number of blocks per layer.
    epochs = The number of epochs used in training.
    mini_batches = The number of mini-batches used in training.
    warm_up_epochs = The number of warm-up epochs used in training.
    device = The device used in training.
    alpha = The alpha value used in the contrastive loss function.
    temperature = The temperature used in the contrastive loss function.
    start_temperature = The starting temperature used in the temperature scheduler.
    final_temperature = The final temperature used in the temperature scheduler.
    warm_temperature_epochs = The number of warm-up epochs used in the temperature scheduler.
    reduction_factor = The reduction factor used in the dynamic convolution layer.
    embedding_dim = The embedding dimension used in the dynamic convolution layer.
    num_kernels = The number of kernels used in the dynamic convolution layer.
    train_dataset = The training dataset used in training.
    validation_dataset = The validation dataset used in training.
    minpsnr = The minimum PSNR value used in the curriculum scheduler.
    maxpsnr = The maximum PSNR value used in the curriculum scheduler.
    alpha_start = The starting alpha value used in the curriculum scheduler.
    alpha_end = The ending alpha value used in the curriculum scheduler.
    epochs_per_phase = The number of epochs per phase used in the curriculum scheduler.
    minpsnr_initial = The initial minimum PSNR value used in the curriculum scheduler.
    auxiliary = The auxiliary value used in the contrastive loss function.
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
            epochs : int =  40,
            mini_batches : int = 512,
            warm_up_epochs : int = 10,
            device : str = "cuda",
            alpha : float = 0.5,
            temperature : float = 1.0,
            start_temperature: float = 20.0,
            final_temperature: float = 1.0,
            warm_temperature_epochs: int = 10,
            reduction_factor: int = 16,
            embedding_dim: int = 16,
            num_kernels: int = 4,
            train_dataset: torch.utils.data.Dataset = None,
            validation_dataset: torch.utils.data.Dataset = None,
            minpsnr: float = -40.0,
            maxpsnr: float = 32.0,
            alpha_start: float = 0.01,
            alpha_end: float = 0.1,
            epochs_per_phase: int = 10,
            minpsnr_initial: float = 32.0,
            auxiliary: float = 0.2):
        
        super(HierarchicalUNet, self).__init__()
        if up_mode != "upsample" and up_mode != "transpose":
            raise ValueError("The up_mode parameter must be either 'upsample' or 'transpose'.")
        
        if merge_mode != "concat" and merge_mode != "add":
            raise ValueError("The merge_mode parameter must be either 'concat' or 'add'.")


        # Define the parameters of the model:
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
        self.eps = torch.Tensor([1e-6])
        self.start_temperature = start_temperature
        self.final_temperature = final_temperature
        self.temperature = temperature
        self.warm_temperature_epochs = warm_temperature_epochs
        self.reduction_factor = reduction_factor
        self.num_kernels = num_kernels
        self.alpha = alpha
        self.auxiliary = auxiliary
        self.minpsnr = minpsnr
        self.maxpsnr = maxpsnr
        self.epochs_per_phase = epochs_per_phase
        self.minpsnr_initial = minpsnr_initial
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.save_hyperparameters()

        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()

        # Define Noise Estimation Network:
        self.noise_estimator = NoiseEstimator(channels, channels)

        # Define the Hierarchical ResBlock:
        self.Hierarchical_ResBlock = HierarchicalResBlock(hidden_dim = 64, num_outputs = depth//2+2)

        # Define the Curriculum Scheduler:
        self.Curriculum_Scheduler = CurriculumScheduling(train_dataset, validation_dataset, minpsnr_initial, minpsnr, maxpsnr, epochs_per_phase, epochs)

        # Define the intermediary layers:
        self.Up_Intermediates_Convs = nn.ModuleList()
        self.Bottleneck_Intermediate = nn.ModuleList()

        # Create Temperature Scheduler:
        self.Obtain_Temperature = lambda epoch: Obtain_Temperature(epoch, self.warm_temperature_epochs, self.start_temperature, self.final_temperature)
        self.learnable_temperature = torch.nn.Parameter(torch.tensor(self.final_temperature, requires_grad = True)).to("cuda" if device == "cuda" else "cpu")
    
        # Define the Encoder Blocks.
        for index in range(depth):
            in_channels = channels * levels if index == 0 else out_channels
            out_channels = self.starting_filters if index == 0 else out_channels * 2  
            for block in range(num_blocks_per_layer):
                if block != self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                             pooling = False, num_layers = num_layers, 
                                                             activation = activation, dropout_rate = dropout_rate,
                                                             kernel_size= 3, stride = 1, padding = 1, reduction_factor = reduction_factor,
                                                             num_kernels = num_kernels, embedding_dim= embedding_dim
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
                                                            activation = activation, dropout_rate = dropout_rate,
                                                            kernel_size= 3, stride = 1, padding = 1, reduction_factor = reduction_factor,
                                                            num_kernels = num_kernels))
    
        self.Bottleneck_Intermediate.append(
            nn.Sequential(
                nn.Conv2d(2*out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.GroupNorm(find_group_number(out_channels), out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, self.channels, kernel_size = 1, stride = 1, padding = 0),
                nn.GroupNorm(1, self.channels),
                nn.LeakyReLU(),
            )
        )

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
                                                           dropout_rate = dropout_rate, kernel_size= 3, stride = 1, padding = 1, 
                                                           reduction_factor = reduction_factor, num_kernels = num_kernels,
                                                           ))
                    if index % 2 == 0:
                        self.Up_Intermediates_Convs.append(
                            nn.Sequential(
                                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                                    nn.GroupNorm(find_group_number(out_channels), out_channels),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(out_channels, self.channels, kernel_size = 1, stride = 1, padding = 0),
                                    nn.GroupNorm(1, self.channels),
                                    nn.LeakyReLU(),
                                )
                            )  
                else:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = False,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate, kernel_size= 3, stride = 1, padding = 1,
                                                           reduction_factor = reduction_factor, num_kernels = num_kernels,
                                                           ))
                    
                     
                in_channels = out_channels
                       
                    
        # Create the Output Layers.
        self.output = nn.Conv2d(out_channels, channels, kernel_size = 1)
        self.reset_parameters()

    @staticmethod
    def weight_init(m):

        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

        if isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def reset_parameters(self):
        for _, m in enumerate(self.modules()):
            if not isinstance(m, nn.LazyLinear):
                self.weight_init(m)

    def get_temperature(self):
        if self.current_epoch < self.warm_temperature_epochs:
            self.temperature = self.Obtain_Temperature(self.current_epoch)
        else:
            self.temperature = torch.clamp(self.learnable_temperature, min = 0.1, max = 2.0)
        return self.temperature

    def forward(self, x):

        stack = None
        factor = 10.0
        self.eps = self.eps.to(x.device)
        normalised_x = x / (torch.max(x) + self.eps)
        reference_prior_noise = torch.sqrt(torch.clamp(normalised_x, min = 0.0) + self.eps)
        refined_noise = self.noise_estimator(reference_prior_noise, reference_prior_noise)
        temperature = self.get_temperature()

        # Begin by encoding the input image representation with a 
        # sineusodial encoding.
        for index in range(self.levels):
            scale = x.clone()*(factor**(-index))
            sin_encoding = torch.sin(scale)
            if stack is None:
                stack = sin_encoding
            else:
                stack = torch.cat((stack, sin_encoding), 1)
        
        x = stack
        encoder = {}
        current_depth = 0
        # Begin by encoding the input image representation.
        for index, Encoder_Block in enumerate(self.encoder):
            out = Encoder_Block(x, refined_noise, temperature)

            if len(out) == 2:
                x, before_pool = out
                encoder[current_depth] = before_pool
                current_depth += 1
            else:
                x = out

        Intermedate_Outs = []
        # Pass the encoding through the bottleneck of the networ
        for index, Bottleneck_Block in enumerate(self.Bottleneck_Block):
            x = Bottleneck_Block(x, refined_noise, temperature)
            Intermedate_Outs.append(self.Bottleneck_Intermediate[index](x))
        # Begin by decoding the input image representation.
        index = 0
        
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool, refined_noise, temperature)

                if index % 2 == 0:
                    half_index = index // 2
                    Intermedate_Outs.append(self.Up_Intermediates_Convs[half_index](x))

                current_depth -= 1
                index+=1
            else:
                x = Decoder_Block(x, None)
                index+=1
    
        # Pass the image representation through the output layer.
        output = self.output(x)
        output, Intermediates = self.Hierarchical_ResBlock(output, Intermedate_Outs, refined_noise)
        return output, Intermediates
    
    def predict(self, x):
        return self.forward(x)
    
    def reset_entropy(self):
        for module in self.modules():
            if isinstance(module, DynamicConv2d):
                module.Entropy = 0.0
    
    def get_entropy(self):
        total_entropy = 0.0
        dynamic_count = 0
        if self.current_epoch < self.warm_up_epochs:
            self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start)*(self.current_epoch/self.warm_up_epochs)
        else:
            self.alpha = self.alpha_end
        for module in self.modules():
            if isinstance(module, DynamicConv2d):
                total_entropy += self.alpha*module.Entropy
                dynamic_count += 1

        if dynamic_count == 0:
            return 0.0
        else:
            return total_entropy/dynamic_count
    def on_train_epoch_start(self):
        self.reset_entropy()

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.epochs_per_phase == 0:
            self.Curriculum_Scheduler.update_psnr_range()
    
    def photonLoss(self,result, target):
        expEnergy = torch.exp(result)
        perImage = -torch.mean(result*target, dim =(-1,-2,-3), keepdims = True )
        perImage = perImage + torch.log(torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))*torch.mean(target, dim =(-1,-2,-3), keepdims = True )
        return torch.mean(perImage)
    
    def MSELoss(self,result, target):
        expEnergy = torch.exp(result)
        expEnergy = expEnergy / (torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))
        target = target / (torch.mean(target, dim =(-1,-2,-3), keepdims = True ))
        return torch.mean((expEnergy-target)**2)

    def training_step(self, batch, batch_idx = None):
        img_input, _, target_img  = batch
        predicted, Intermediate_Outs = self.forward(img_input)
        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)

        train_loss = photon_loss + mse_loss
        entropy_loss = self.get_entropy()
        
        intermediate_loss = 0.0
        for _, intermediate_out in enumerate(Intermediate_Outs):
            intermediate_loss = intermediate_loss + self.MSELoss(intermediate_out, target_img)
        
        Train_Loss = (1- self.auxiliary) * train_loss + self.auxiliary * (intermediate_loss + entropy_loss)

        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", Train_Loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("entropy_loss", entropy_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        #self.log("contrastive", contrastive_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("intermediate_loss",intermediate_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        return Train_Loss

    def validation_step(self, batch, batch_idx = None):
        img_input, _, target_img  = batch
        predicted, Intermediate_Outs = self.forward(img_input)
        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)

        validation_loss = photon_loss + mse_loss
        entropy_loss = self.get_entropy()
        
        intermediate_loss = 0.0
        for _, intermediate_out in enumerate(Intermediate_Outs):
            intermediate_loss = intermediate_loss + self.MSELoss(intermediate_out, target_img)
        
        Validation_Loss = (1- self.auxiliary) * validation_loss + self.auxiliary * (intermediate_loss + entropy_loss)

        self.log("val_loss", Validation_Loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_entropy_loss", entropy_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        #self.log("val_contrastive", contrastive_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_intermediate_loss", intermediate_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        return Validation_Loss
    
    def test_step(self, batch, batch_idx = None):
        img_input, _, target_img = batch
        predicted, _ = self.forward(img_input)
        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)
        test_loss = photon_loss + mse_loss

        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss
    
    def configure_optimizers(self):
        #num_warm_steps = self.mini_batches * self.warm_up_epochs
        num_training_steps = self.mini_batches * self.epochs
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

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






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


class NoiseEstimator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NoiseEstimator, self).__init__()
        self.conv1 = nn.Conv2d(2*in_channels, 32, 3, 1)
        self.groupnorm1 = nn.GroupNorm(find_group_number(32), 32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.groupnorm2 = nn.GroupNorm(find_group_number(64), 64)
        self.conv3 = nn.Conv2d(64, out_channels, 3, 1)

    def forward(self, ImageNoise, ReferenceImage):
        x = torch.cat([ImageNoise, ReferenceImage], dim=1)
        x = F.relu(self.groupnorm1(self.conv1(x)))
        x = F.relu(self.groupnorm2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x
    
class HierarchicalResBlock(nn.Module):
    # This block is inspired by the Progressive Hierarchical Learning introduced in the paper:
    # 'Progressive residual networks for image super-resolution’, by Jin Wan et al.
    def __init__(self,hidden_dim, num_outputs):
        super(HierarchicalResBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, padding = 1)
        self.fc = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
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
            self.dropout = nn.Dropout2d(dropout_rate)
        
        self.gated_convolution_gate = Gated_Convolution(out_channels, out_channels)
    
    def forward(self, x, refined_noise):
        refined_noise = F.interpolate(refined_noise, size = x.shape[2:], mode = 'bilinear')
        x_skip = self.gated_convolution_gate(self.Skip_Block(x), refined_noise)

        x = self.Main_Processing_Block(x_skip)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x + x_skip

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
            self.dropout = nn.Dropout2d(dropout_rate)
        
        # If upsampling is required, add an upsampling layer.
        if up_mode == "transpose" and upsampling:
            self.UpBlock = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
                nn.GroupNorm(n_groups, out_channels),
                activation,
            ]
        elif up_mode == "upsample" and upsampling:
            self.UpBlock = nn.Sequential(
                nn.Upsample(mode = "bilinear", scale_factor = 2, align_corners = True),
                nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                nn.GroupNorm(n_groups, out_channels),
                activation,
            )
        
        if upsampling:
            self.UpBlock = nn.Sequential(*self.UpBlock)

        self.attention_gate = AttentionGate(out_channels, out_channels, out_channels)
        self.gated_convolution_gate = Gated_Convolution(out_channels, out_channels)

    def forward(self, below, above, refined_noise):
        # Begin by matching upsampling the relevant layers
        # and merging them.
        if self.upsampling:
            below = self.UpBlock(below)
            below = self.attention_gate(above, below)
            if self.merge_mode == "concat":
                x = torch.cat((below, above), 1)
            else:
                x = below + above
        else:
            x = below

        x_skip = self.Skip_Block(x)
        refined_noise = F.interpolate(refined_noise, size = x_skip.shape[2:], mode = 'bilinear')
        x = self.gated_convolution_gate(x_skip, refined_noise)

        x = self.Main_Processing_Block(x)

        x = x + x_skip

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
    levels = The number of levels in the network.
    channels = The number of channels in the input image representation.
    depth = The depth of the network.
    up_mode = The method used to upsample the input image representation.
    merge_mode = The method used to merge the input and upsampled image representations.
    num_layers = The number of convolutional layers in the block.
    activation = The activation function used in the block.
    dropout_rate = The dropout rate used in the block.
    learning_rate = The learning rate used in the block.
    weight_decay = The weight decay used in the block.
    starting_filters = The number of starting filters used in the block.
    num_blocks_per_layer = The number of blocks per layer in the block.
    epochs = The number of epochs used in the block.
    mini_batches = The number of mini-batches used in the block.
    warm_up_epochs = The number of warm-up epochs used in the block.
    alpha = The alpha value used in the block.
    device = The device used in the block.
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
            epochs: int = 40,
            mini_batches: int = 512,
            warm_up_epochs: int = 10,
            alpha: int = 0.5,
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
        self.alpha = alpha
        self.eps = torch.Tensor([1e-6]).to("cuda" if device == "cuda" else "cpu")

        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()

        # Define Noise Estimation Network:
        self.noise_estimator = NoiseEstimator(channels, channels)

        # Define the Intermediary Modules:
        self.Bottleneck_Intermediate = nn.ModuleList()
        self.Up_Intermediates_Convs = nn.ModuleList()

        # Define the Hierarchical ResBlock:
        self.Hierarchical_ResBlock = HierarchicalResBlock(hidden_dim = 64, num_outputs = depth+1)
        
        # Define the Encoder Blocks.
        for index in range(depth):
            in_channels = channels * levels if index == 0 else out_channels
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
    
        self.Bottleneck_Intermediate.append(
            nn.Sequential(
                nn.Conv2d(2*out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
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
                                                           dropout_rate = dropout_rate,
                                                        ))
                    self.Up_Intermediates_Convs.append(
                        nn.Sequential(
                                nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
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
        self.eps = self.eps.to(x.device)
        normalised_x = x / (torch.max(x) + self.eps)
        reference_prior_noise = torch.sqrt(torch.clamp(normalised_x, min = 0.0) + self.eps)
        refined_noise = self.noise_estimator(reference_prior_noise, reference_prior_noise)

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
            out = Encoder_Block(x, refined_noise)

            if len(out) == 2:
                x, before_pool = out
                encoder[current_depth] = before_pool
                current_depth += 1
            else:
                x = out
        Intermedate_Outs = []
        # Pass the encoding through the bottleneck of the networ
        for index, Bottleneck_Block in enumerate(self.Bottleneck_Block):
            x = Bottleneck_Block(x, refined_noise)
            Intermedate_Outs.append(self.Bottleneck_Intermediate[index](x))
        # Begin by decoding the input image representation.
        index = 0
        
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool, refined_noise)

                Intermedate_Outs.append(self.Up_Intermediates_Convs[index](x))

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
    
    def photonLoss(self,result, target):
        expEnergy = torch.exp(result)
        perImage =  -torch.mean(result*target, dim =(-1,-2,-3), keepdims = True )
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
        intermediate_loss = 0.0
         
        #Compute Intermediate Losses to enforce DeepSupervision
        for index in range(len(Intermediate_Outs)):
            Intermediate = Intermediate_Outs[index]
            intermediate_loss = intermediate_loss + self.photonLoss(Intermediate, target_img)

        train_loss = (1-self.alpha) * train_loss + self.alpha*intermediate_loss

        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("intermediate_loss", intermediate_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss
    
    def validation_step(self, batch, batch_idx = None):
        img_input, _, target_img = batch
        predicted, Intermediate_Outs = self.forward(img_input)
        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)
        valid_loss = photon_loss + mse_loss
        intermediate_loss = 0.0

        #Compute Intermediate Losses to enforce DeepSupervision
        for index in range(len(Intermediate_Outs)):
            Intermediate = Intermediate_Outs[index]
            intermediate_loss = intermediate_loss + self.photonLoss(Intermediate, target_img)
        
        valid_loss = (1-self.alpha) * valid_loss + self.alpha*intermediate_loss

        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_intermediate_loss", intermediate_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss
    
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
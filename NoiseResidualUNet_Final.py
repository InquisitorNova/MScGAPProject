"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 
https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock. I have also added the ability to use GroupNorm layers in the ResBlock.
I have modified the UNet so that it is used to predict the residual between the noisy and clean image, taking
in a prior image representation and a PSNR value as input. The UNet is trained to predict the residual between the
noisy and clean image, and the predicted residual is added to the prior image representation to produce the final
denoised image. The UNet is trained using a photon loss, which is a modified version of the Poisson loss, and a mean
squared error loss. The UNet is trained using the AdamW optimizer and a OneCycle learning rate scheduler. This UNet 
is based of of the following papers:
1. U-Net: Convolutional Networks for Biomedical Image Segmentation by Ronneberger et al. https://arxiv.org/abs/1505.04597.pdf
2. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising by Zhang et al. https://arxiv.org/abs/1608.03981.pdf
3. A DISCIPLINED APPROACH TO NEURAL NETWORK
HYPER-PARAMETERS: PART 1 – LEARNING RATE,
BATCH SIZE, MOMENTUM, AND WEIGHT DECAY by Leslie N. Smith https://arxiv.org/pdf/1803.09820.pdf

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

class DownConvolutionBlock(nn.Module):
    """
    For the UNet, this block is used to downsample the input image representation.
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
    
    def forward(self, x):
        x_skip = self.Skip_Block(x)

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
    For the Residual UNet, this block is used to upsample the input image representation.
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

    def forward(self, below, above):

        # Begin by matching upsampling the relevant layers
        # and merging them.
        if self.upsampling:
            below = self.UpBlock(below)
            if self.merge_mode == "concat":
                x = torch.cat((below, above), 1)
            else:
                x = below + above
        else:
            x = below

        x_skip = self.Skip_Block(x)

        x = self.Main_Processing_Block(x_skip)

        x+= x_skip

        if self.dropout_rate is not None:
            x = self.dropout(x)
        
        return x
    
class ResUNet(pl.LightningModule):
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
    levels = The number of levels in the ResUNet.
    channels = The number of channels in the input image representation.
    depth = The depth of the ResUNet.
    up_mode = The method used to upsample the input image representation.
    merge_mode = The method used to merge the input and upsampled image representations.
    num_layers = The number of convolutional layers in the ResBlock.
    activation = The activation function used in the ResBlock.
    dropout_rate = The dropout rate used in the ResBlock.
    learning_rate = The learning rate used in the AdamW optimizer.
    weight_decay = The weight decay used in the AdamW optimizer.
    starting_filters = The number of starting filters used in the ResUNet.
    num_blocks_per_layer = The number of ResBlocks per layer in the ResUNet.
    mini_batches = The number of mini-batches used in the ResUNet.
    warm_up_epochs = The number of warm-up epochs used in the ResUNet.
    epochs = The number of epochs used in the ResUNet.
    average_face = The average face used in the ResUNet.
    bottleneck = The number of channels in the bottleneck of the ResUNet.
    device = The device used to train the ResUNet.
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
            dropout_rate: float = 0.3,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            starting_filters: float = 32,
            num_blocks_per_layer: int = 2,
            mini_batches: int = 512,
            warm_up_epochs: int = 10,
            epochs: int = 30,
            average_face: torch.Tensor = None,
            bottleneck: int = 64,
            device = "cuda"):
    
        self.save_hyperparameters()
        super(ResUNet, self).__init__()

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
        self.mini_batches = mini_batches
        self.warm_up_epochs = warm_up_epochs
        self.epochs = epochs
        self.average_face = average_face.to(device)
        self.bottleneck = bottleneck
        self.eps = torch.Tensor([1e-8])
        
        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()
        
        # Define the Encoder Blocks.
        for index in range(depth):
            in_channels = channels * (levels + 2) + 1 if index == 0 else out_channels
            out_channels = self.starting_filters if index == 0 else out_channels * 2  
            for block in range(num_blocks_per_layer):
                if block != self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                             pooling = False, num_layers = num_layers, 
                                                             activation = activation, dropout_rate = dropout_rate))
                elif block == self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                         pooling = True, num_layers = num_layers, 
                                                         activation = activation, dropout_rate = dropout_rate))
                    
                in_channels = out_channels
                    
        # Create the Bottleneck Pathway.
        self.decoder_intermediates = nn.ModuleList()
        self.Bottleneck_Block = nn.ModuleList()
        pooling = False

        self.Bottleneck_Block.append(DownConvolutionBlock(out_channels, 2*out_channels,
                                                            pooling = pooling, num_layers = num_layers,
                                                            activation = activation, dropout_rate = dropout_rate))
        self.decoder_intermediates.append(
            nn.Sequential(
                            nn.Conv2d(2*out_channels, self.channels, kernel_size = 1, stride = 1, padding = 0),
                            nn.GroupNorm(1, self.channels),
                            nn.LeakyReLU(),
                            nn.Dropout(),
                            nn.Conv2d(self.channels, self.channels, kernel_size = 3, stride = 1, padding = 1),
                            nn.GroupNorm(1, self.channels),
                            nn.LeakyReLU(),
                            ).to(device)
                            )

        out_channels = out_channels*2
        # Create the Decoder Pathway.
        self.decoder = nn.ModuleList()
        for index in range(depth):
            in_channels = out_channels
            out_channels = in_channels // 2

            for block in range(num_blocks_per_layer):
                if  index != self.depth-1:
                    self.decoder_intermediates.append(
                    nn.Sequential(
                                nn.Conv2d(in_channels, self.channels, kernel_size = 1, stride = 1, padding = 0),
                                nn.GroupNorm(1, self.channels),
                                nn.LeakyReLU(),
                                nn.Dropout(),
                                nn.Conv2d(self.channels, self.channels, kernel_size = 3, stride = 1, padding = 1),
                                nn.GroupNorm(1, self.channels),
                                nn.LeakyReLU(),
                            ).to(device)
                    )
                if block == 0:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = True,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate))
                else:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = False,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate))
                in_channels = out_channels
                       
                    
        # Create the Output Layers.
        self.output = nn.Conv2d(out_channels, channels, kernel_size = 1)

        n_groups_1 = find_group_number(self.bottleneck, min_group_channels = 4, max_group_channels = 32)
        n_groups_2 = find_group_number(len(self.decoder_intermediates), min_group_channels = 4, max_group_channels = 32)

        # Create the Residual Network for predicting the weights of the residuals.
        self.ResidualNetwork = nn.Sequential(
            nn.Conv2d(out_channels, self.bottleneck, kernel_size = 3, padding = 1),
            nn.GroupNorm(n_groups_1, self.bottleneck),
            nn.LeakyReLU(),
            nn.Conv2d(self.bottleneck, len(self.decoder_intermediates), kernel_size = 3, padding = 1),
            nn.GroupNorm(n_groups_2, len(self.decoder_intermediates)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(len(self.decoder_intermediates), len(self.decoder_intermediates)),
            nn.Softmax(dim = 1)
        )

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

    def forward(self, x, psnr_values):

        stack = None
        factor = 10.0
        self.eps = self.eps.to(x.device)
        noise_level = torch.sqrt(x + self.eps)
        psnr_values = psnr_values[:,0, :, :].unsqueeze(1)

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
        x = torch.cat((x, self.average_face.repeat(x.size(0), 1, 1, 1)), 1)
        x = torch.cat((x, psnr_values, noise_level), 1)
        #x = torch.cat((x, psnr_values), 1)
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
        
        #self.decoder_intermediates[0](x)
        
        # Begin by decoding the input image representation.
        index = 0
        Residuals = []
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                #if  index != self.depth-1:
                    #Residuals.append(self.decoder_intermediates[index+1](x))
                
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool)
                current_depth -= 1
                index+=1
            else:
                x = Decoder_Block(x, None)
                index+=1
    
        # Pass the image representation through the output layer.
        log_residual = self.output(x)
        self.normalised_weights = self.ResidualNetwork(x)
        for index, residual in enumerate(Residuals):
           residual = F.interpolate(residual, size = log_residual.size()[2:], mode = "bilinear", align_corners = False)
           log_residual += self.normalised_weights[:,index].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*residual

        return log_residual

    def predict(self, x, psnr_values):
        log_residual = self.forward(x, psnr_values)
        Predicted = x + torch.exp(log_residual)
        return Predicted
    
    def compute_psnr_on_batch(self, image, target):
        self.eps = self.eps.to(image.device)
        mse = F.mse_loss(image, target)
        maximum = image.max()
        psnr = 10 * torch.log10(maximum / torch.maximum(mse, self.eps))
        return psnr.mean()
    
    def photonLoss(self, result, target):
        perImage = -torch.mean(target*torch.log(result + self.eps.to(result.device)), dim =(-1,-2,-3), keepdim = True)
        perImage += torch.log(torch.mean(result + self.eps.to(result.device), dim =(-1,-2,-3), keepdim = True ))*torch.mean(target, dim =(-1,-2,-3), keepdim = True )
        return torch.mean(perImage)
    
    def MSELoss(self,result, target):
        self.eps = self.eps.to(result.device)
        expEnergy = result
        expEnergy /= (torch.mean(expEnergy, dim =(-1,-2,-3), keepdim = True ) + self.eps)
        target = target / (torch.mean(target, dim =(-1,-2,-3), keepdim = True ) + self.eps)
        #print(f"MSE_Loss", torch.mean((expEnergy-target)**2))
        return torch.clamp(torch.mean((expEnergy-target)**2), min = 0.0, max = 1000000.0)
    
    def training_step(self, batch, batch_idx = None):
        self.eps = self.eps.to(self.device)
        img_input,  psnr_values, target_img  = batch
        log_residual = self.forward(img_input, psnr_values)

        prior = img_input
        residual = torch.exp(log_residual)
        predicted = prior + residual
        residual_variance = torch.var(residual, dim = (-1,-2,-3), keepdim = True)
        varinace_penalty = 1.0/(self.eps + residual_variance.mean())

        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)
        train_loss = photon_loss + 0.4*varinace_penalty 
        psnr_values = self.compute_psnr_on_batch(predicted, target_img)

        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_psnr", psnr_values, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss
    
    def validation_step(self, batch, batch_idx = None):
        self.eps = self.eps.to(self.device)
        img_input, psnr_values, target_img = batch
        log_residual = self.forward(img_input, psnr_values)

        prior = img_input
        residual = torch.exp(log_residual)
        predicted = prior + residual
        residual_variance = torch.var(residual, dim = (-1,-2,-3), keepdim = True)
        varinace_penalty = 1.0/(self.eps + residual_variance.mean())

        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)

        valid_loss = photon_loss + 0.4*varinace_penalty
        psnr_values = self.compute_psnr_on_batch(predicted, target_img)

        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_psnr", psnr_values, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss
    
    def test_step(self, batch, batch_idx = None):
        self.eps = self.eps.to(self.device)
        img_input, psnr_values, target_img = batch
        log_residual = self.forward(img_input, psnr_values)

        prior = img_input
        residual = torch.exp(log_residual)
        predicted = prior + residual
        residual_variance = torch.var(residual, dim = (-1,-2,-3), keepdim = True)
        varinace_penalty = 1.0/(self.eps + residual_variance.mean())

        photon_loss = self.photonLoss(predicted, target_img)
        mse_loss = self.MSELoss(predicted, target_img)
        test_loss = photon_loss + 0.4*varinace_penalty
        psnr_values = self.compute_psnr_on_batch(predicted, target_img)

        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_photon_loss", photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_mse_loss", mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_psnr", psnr_values, on_step = False, on_epoch = True, prog_bar = True, logger = True)
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

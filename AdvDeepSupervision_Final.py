"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 
https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock. I have added the ability to use GroupNorm layers in the ResBlock
and the ability to use dropout layers in the ResBlock. I have also added the ability to use different
types of pooling layers in the ResBlock. The ResUNet is designed to be used with 2D image data.
This ResNet performs deep supervision, which means that the network is trained to predict the output
at multiple stages of the network. This is designed to improve the performance of the network.
The ResUNet also used a prior network to compute the input to the ResNet as a function of the 
noisy image and the dataset's average face. This is designed to improve the performance of the network.
This ResUNet is inspired by a number of different papers, including:
1. U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
2. Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
3. A Comprehensive Review on Deep Supervision: Theories and Applications:
    https://arxiv.org/abs/2207.02376

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
epilson = 1e-6

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

class NoiseEmbedder(nn.Module):
    def __init__(self, num_conditions, hidden_dim, height, width, channels):
        super(NoiseEmbedder, self).__init__()
        
        self.embedding_dim = channels * height * width

        self.Dense = nn.Linear(num_conditions, hidden_dim)
        group_num = find_group_number(hidden_dim)
        self.GroupNorm = nn.GroupNorm(group_num, hidden_dim)

        self.Dense_2 = nn.Linear(hidden_dim, self.embedding_dim)
        group_num = find_group_number(self.embedding_dim)
        self.GroupNorm_2 = nn.GroupNorm(group_num, self.embedding_dim)

        self.Conv3x3 = nn.Conv2d(channels, 3, kernel_size = 3, padding = 1, stride = 1)
        group_num = find_group_number(3)
        self.GroupNorm_3 = nn.GroupNorm(group_num, 3)
    
    def forward(self, image, condition):
        batch_size, channels, height, width = image.shape

        x = self.Dense(condition) # (-1,1) -> (-1, hidden_dim)
        x = self.GroupNorm(x)
        x = F.leaky_relu(x)

        x = self.Dense_2(x) # (-1, hidden_dim) -> (-1, embedding_dim)
        x = self.GroupNorm_2(x)
        x = F.leaky_relu(x)

        x = x.view(batch_size, channels, height, width)

        x = self.Conv3x3(x)
        x = self.GroupNorm_3(x)
        x = F.leaky_relu(x)

        # class label image = x
        #x = torch.cat((image, x), dim = 1)

        return x#, class label image


class PriorGate(nn.Module):
    def __init__(self, in_channels, out_channels, factor = 8):
        super(PriorGate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Conv_Input = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(),
        )

        self.Conv_Prior = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(),
        )

        self.Attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//factor, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels//factor),
            nn.LeakyReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels//factor, out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels),
            nn.Sigmoid(),
        )

        self.Psnr_Map = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(),
        )

        self.Weight_1 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.Weight_2 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.Weight_3 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.Bias = nn.Parameter(torch.randn(1, out_channels, 1, 1))

        nn.init.xavier_uniform_(self.Weight_1)
        nn.init.xavier_uniform_(self.Weight_2)
        nn.init.xavier_uniform_(self.Weight_3)
        nn.init.zeros_(self.Bias)
    
    def forward(self, x, prior, psnr_map):
        x_input = self.Conv_Input(x)
        x_prior = self.Conv_Prior(prior)
        x_psnr_map = self.Psnr_Map(psnr_map)
        
        Attention = torch.clamp(self.Attention(x_input), epilson, 1-epilson)
        
        x_input = x_input*Attention
        x_prior = x_prior*(1-Attention)
        #print(x_input.shape, x_prior.shape, x_psnr_map.shape)

        Gate = x_input * self.Weight_1 + x_prior * self.Weight_2 + x_psnr_map * self.Weight_3 + self.Bias
        Gate = torch.clamp(Gate, min = -10, max = 10)
        Gate = F.sigmoid(Gate)
        return x_prior*Gate + x_input*(1-Gate)

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


class AdvDeepSupResUNet(pl.LightningModule):
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
            mini_batches: int = 512,
            warm_up_epochs: int = 10,
            epochs: int = 30,
            auxiliary_weight = 0.2,
            batch_size: int = 32,
            average_face: torch.Tensor = None,
            device = "cuda"):
        
        self.save_hyperparameters()
        super(AdvDeepSupResUNet, self).__init__()

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
        self.auxiliary_weight = auxiliary_weight
        self.batch_size = batch_size
        self.average_face = average_face

        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()
        self.Down_Intermediates_Convs = nn.ModuleList()
        self.Up_Intermediates_Convs = nn.ModuleList()
        self.Priority_Gate = nn.ModuleList()

        # Define the Prior Gate:
        self.Priority_Gate.append(PriorGate(channels*levels, channels*levels))
        self.factor = nn.Parameter(torch.Tensor([10.0]))
        self.NoiseEmbedding = NoiseEmbedder(1, 16, 128, 128, 3)
        
        # Define the Encoder Blocks.
        for index in range(depth):
            in_channels = channels*levels if index == 0 else out_channels
            out_channels = self.starting_filters if index == 0 else out_channels * 2  
            for block in range(num_blocks_per_layer):
                if block != self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                             pooling = False, num_layers = num_layers, 
                                                             activation = activation, dropout_rate = dropout_rate))
                else:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                         pooling = True, num_layers = num_layers, 
                                                         activation = activation, dropout_rate = dropout_rate))
                    
                    if index % 2 == 0:
                        self.Down_Intermediates_Convs.append(
                            nn.Sequential(
                                nn.Conv2d(out_channels, self.channels, kernel_size = 1, stride = 1, padding = 0),
                                nn.GroupNorm(1, self.channels),
                            )
                        )
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
                                                           dropout_rate = dropout_rate))
                    
                    if index % 2 == 0:
                        self.Up_Intermediates_Convs.append(
                            nn.Sequential(
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

        # Create the Output Layer.
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

    def forward(self, x, psnr_map):
        
        batch_size, _, _, _ = x.shape
        stack_out = None
        stack_embedding = None
        factor = F.relu(self.factor)
        #condition = self.NoiseEmbedding(x, psnr_map[:,0,0,0].view(-1,1))

        #psnr = psnr_map[:, 0,0,0].view(-1,1)

        # Begin by encoding the input image representation with a 
        # sineusodial encoding.
        out = x.clone()
        embedding = self.average_face.repeat(batch_size, 1, 1, 1).clone()
        for index in range(self.levels):
            scale_out = out*(factor**(-index))
            embedding_out = embedding*(factor**(-index))
            sin_encoding = torch.sin(scale_out)
            cos_encoding = torch.cos(embedding_out)
            if stack_out is None:
                stack_out = sin_encoding
                stack_embedding = cos_encoding
            else:
                stack_out = torch.cat((stack_out, sin_encoding), 1)
                stack_embedding = torch.cat((stack_embedding, cos_encoding),1)
        
        x = stack_out
        self.embbeding = stack_embedding

        x = self.Priority_Gate[0](x, self.embbeding, psnr_map)
        encoder = {}

        current_depth = 0
        Down_Intermediate_Outs = [] 
        # Begin by encoding the input image representation.
        for index, Encoder_Block in enumerate(self.encoder):
            out = Encoder_Block(x)

            if len(out) == 2:
                x, before_pool = out
                encoder[current_depth] = before_pool
                current_depth += 1

                if index % 2 == 0:
                    half_index = index // 2
                    Down_Intermediate_Outs.append(self.Down_Intermediates_Convs[half_index](before_pool))
            else:
                x = out

        # Pass the encoding through the bottleneck of the network
        for index, Bottleneck_Block in enumerate(self.Bottleneck_Block):
            x = Bottleneck_Block(x)
        
        # Begin by decoding the input image representation.
        index = 0
        Up_Intermedate_Outs = []
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool)

                if index % 2 == 0:
                    half_index = index // 2
                    Up_Intermedate_Outs.append(self.Up_Intermediates_Convs[half_index](x))

                current_depth -= 1
                index+=1
            else:
                x = Decoder_Block(x, None)
                index+=1
        
        # Pass the image representation through the output layer.
        x = self.output(x)

        return x, Down_Intermediate_Outs, Up_Intermedate_Outs

    def predict(self, x, psnr_maps):
        return self.forward(x, psnr_maps)[0]
    
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
        img_input, psnr_map, target_img  = batch
        predicted, down_intermediates, up_intermediates = self.forward(img_input, psnr_map)
        train_photon_loss = self.photonLoss(predicted, target_img)
        train_mse_loss = self.MSELoss(predicted, target_img)
        train_loss = train_photon_loss + train_mse_loss
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        down_loss, up_loss = 0,0

        # Compute Intermediate Losses to enforce DeepSupervision
        for index in range(len(down_intermediates)):
            down_intermediate = down_intermediates[index]
            downsampled_target = F.interpolate(target_img, size = down_intermediate.shape[-2:], mode = 'bilinear')
            down_loss = down_loss + self.photonLoss(down_intermediate, downsampled_target)

        for index in range(len(up_intermediates)):
            up_intermediate = up_intermediates[index]
            upsampled_target = F.interpolate(target_img, size = up_intermediate.shape[-2:], mode = 'bilinear')
            up_loss = up_loss + self.photonLoss(up_intermediate, upsampled_target)

        train_loss = (1-self.auxiliary_weight) * train_loss + self.auxiliary_weight * (down_loss + up_loss)

        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_photon_loss", train_photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_mse_loss", train_mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)        
        return train_loss

    def validation_step(self, batch, batch_idx = None):
        img_input, psnr_maps, target_img = batch
        predicted, down_intermediates, up_intermediates = self.forward(img_input, psnr_maps)
        valid_photon_loss = self.photonLoss(predicted, target_img)
        valid_mse_loss = self.MSELoss(predicted, target_img)
        valid_loss = valid_photon_loss + valid_mse_loss

        down_loss, up_loss = 0,0

        # Compute Intermediate Losses to enforce DeepSupervision
        for index in range(len(down_intermediates)):
            down_intermediate = down_intermediates[index]
            downsampled_target = F.interpolate(target_img, size = down_intermediate.shape[-2:], mode = 'bilinear')
            down_loss = down_loss + self.photonLoss(down_intermediate, downsampled_target)

        for index in range(len(up_intermediates)):
            up_intermediate = up_intermediates[index]
            upsampled_target = F.interpolate(target_img, size = up_intermediate.shape[-2:], mode = 'bilinear')
            up_loss = up_loss + self.photonLoss(up_intermediate, upsampled_target)
        
        valid_loss = (1-self.auxiliary_weight) * valid_loss + self.auxiliary_weight * (down_loss + up_loss)

        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_photon_loss", valid_photon_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("val_mse_loss", valid_mse_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss

    def test_step(self, batch, batch_idx = None):
        img_input, psnr_maps, target_img = batch
        predicted, _, _ = self.forward(img_input, psnr_maps)
        test_photon_loss = self.photonLoss(predicted, target_img)
        test_mse_loss = self.MSELoss(predicted, target_img)
        test_loss = test_photon_loss + test_mse_loss
        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_photon_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("test_mse_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss
    
    def configure_optimizers(self):
        num_warm_steps = self.mini_batches * self.warm_up_epochs
        num_training_steps = self.mini_batches * self.epochs

        # Changed the optimize from Adam to AdamW to decople weight decay and reduce overfitting.
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = 1e-4)

        # Use the OneCycleLR learning rate scheduler.
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
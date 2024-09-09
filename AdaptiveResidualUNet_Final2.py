"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 

https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock. The ResBlock has been modified to include Group Normalisation
layers, and the FiLM and AdaIN layers have been added to the network. The FiLM and AdaIN layers
are used to modulate the features in the image based on the PSNR value. The NoiseEmbedder module
is used to form an embedding vector from the PSNR value that can be used to condition the network.
The AdaptiveResUNet class is designed
to be incorporated within the GAP Framework. It is a fully-convolutional
neural network with skip connections that allow for information to travel
between the encoder and decoder pathways making up the network. The encoder
pathway is designed to downsample the input image representation, while the
is designed to upsample the input image representation. 

The UNet is based  on the following papers:
1. U-Net: Convolutional Networks for Biomedical Image Segmentation
   https://arxiv.org/abs/1505.04597.pdf
2. FiLM: Visual Reasoning with a General Conditioning Layer
    https://arxiv.org/pdf/1709.07871.pdf
3. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
    https://arxiv.org/pdf/1703.06868.pdf    

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

class FiLM(nn.Module):
    """
    FiLM - Feature-wise Linear Modulation Layer
    num_features - Number of Features in the Image
    This layer is used to modulate the features in the image based on the PSNR value.
    https://arxiv.org/pdf/1709.07871.pdf
    """
    def __init__(self, embedding_dim, num_features):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.gamma = nn.Linear(embedding_dim, num_features)
        self.beta = nn.Linear(embedding_dim, num_features)
        #self.norm_1 = nn.InstanceNorm2d(num_features, affine = False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.gamma.weight, a=0.01)
        nn.init.kaiming_uniform_(self.beta.weight, a=0.01)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, embedding):

        # Compute the gamma and beta values for the FiLM layer.
        gamma = F.leaky_relu(self.gamma(embedding)).view(-1, self.num_features, 1, 1)
        beta = F.leaky_relu(self.beta(embedding)).view(-1, self.num_features, 1, 1)

        #Apply the FiLM layer to the input image representation.
        x = x * gamma + beta
        return x

class AdaIN(nn.Module):
    """
    AdaIN - Adaptive Instance Normalisation Layer
    num_features - Number of Features in the Image
    This layer is used to adaptively normalise the features in the image based on the PSNR value.
    https://arxiv.org/pdf/1703.06868.pdf
    """
    def __init__(self, embedding_dim, num_features):
        super(AdaIN, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.scale = nn.Linear(embedding_dim, num_features)
        self.shift = nn.Linear(embedding_dim, num_features)
        self.norm_1 = nn.InstanceNorm2d(num_features, affine = False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.scale.weight, a=0.01)
        nn.init.kaiming_uniform_(self.shift.weight, a=0.01)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)
    
    def forward(self, x, embedding):

        # Compute the scale and shift values for the AdaIN layer.
        scale = F.leaky_relu(self.scale(embedding)).view(-1, self.num_features, 1, 1)
        shift = F.leaky_relu(self.shift(embedding)).view(-1, self.num_features, 1, 1)

        return scale * self.norm_1(x) + shift
    
class NoiseEmbedder(nn.Module):
    """
    This class defines the NoiseEmbedder module.
    This module takes the psuedo psnr and forms an embedding vector from it
    that can be used to condition the network.
    num_conditions = The number of conditions to embed.
    hidden_dim = The dimensionality of the hidden layer.
    embedding_dim = The dimensionality of the embedding layer.
    dropout_rate = The dropout rate used in the module.
    """
    def __init__(self, num_conditions, hidden_dim = 16, embedding_dim = 64,
                 dropout_rate = 0.2):
        super(NoiseEmbedder, self).__init__()

        # Initilise the parameters and layers of the module.
        self.num_conditions = num_conditions
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Create the layers of the module.
        self.dense = nn.Linear(num_conditions, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, embedding_dim)

        # Add normalisation layers.
        n_groups = find_group_number(hidden_dim, min_group_channels = 4, max_group_channels = 32)
        self.norm = nn.GroupNorm(n_groups, hidden_dim)

        n_groups = find_group_number(embedding_dim, min_group_channels = 4, max_group_channels = 32)
        self.norm_2 = nn.GroupNorm(n_groups, embedding_dim)

        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, condition):
        x = self.dense(condition)
        x = self.norm(x)
        x = F.leaky_relu(x)

        if self.dropout_rate is not None:
            x = self.dropout(x)

        x = self.dense_2(x)
        x = self.norm_2(x)
        x = F.leaky_relu(x)

        return x

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
                 activation = nn.LeakyReLU(), dropout_rate = 0.5, embedding_dim = 64,
                 adain_enabled = False, film_enabled = False):
        super(DownConvolutionBlock, self).__init__()

        # Initialise the layers of the block.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.film_enabled = film_enabled
        self.adain_enabled = adain_enabled
        self.embedding_dim = embedding_dim

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

        # If AdaIN is enabled, add an AdaIN layer.
        if adain_enabled:
            self.adain = AdaIN(embedding_dim, out_channels)
        
        # If FiLM is enabled, add a FiLM layer.
        if film_enabled:
            self.film = FiLM(embedding_dim, out_channels)
    
    def forward(self, x, embedding):
        x_skip = self.Skip_Block(x)

        x = self.Main_Processing_Block(x_skip)

        if self.film_enabled:
            x = self.film(x, embedding)
        
        if self.adain_enabled:
            x = self.adain(x, embedding)
        
        if self.dropout is not None:
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
                 up_mode = "upsample", num_layers = 2, activation = nn.LeakyReLU(), embedding_dim = 64,
                 dropout_rate = 0.5, adain_enabled = False, film_enabled = False):
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
        self.adain_enabled = adain_enabled
        self.film_enabled = film_enabled
        self.embedding_dim = embedding_dim

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

        # If AdaIN is enabled, add an AdaIN layer.
        if adain_enabled:
            self.adain = AdaIN(embedding_dim, out_channels)
        
        # If FiLM is enabled, add a FiLM layer.
        if film_enabled:
            self.film = FiLM(embedding_dim, out_channels)

    def forward(self, below, above, embedding):

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

        if self.film_enabled:
            x = self.film(x, embedding)
        
        if self.adain_enabled:
            x = self.adain(x, embedding)

        x+= x_skip

        if self.dropout_rate is not None:
            x = self.dropout(x)
        
        return x
    
class AdaptiveResUNet(pl.LightningModule):
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
            film_enabled: bool = False,
            adain_enabled: bool = False,
            maxpsnr: float = 32.0,
            minpsnr: float = -40.0,
            embedding_dim: int = 64,
            units: int = 16,
            num_conditions: int = 1,
            device = "cuda"):
    
        self.save_hyperparameters()
        super(AdaptiveResUNet, self).__init__()

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
        self.film_enabled = film_enabled
        self.adain_enabled = adain_enabled
        self.eps = torch.Tensor([1e-8])
        self.minpsnr = minpsnr
        self.maxpsnr = maxpsnr
        self.embedding_dim: int = embedding_dim,
        self.units :int = units,
        self.num_conditions: int = num_conditions


        # Create the Noise Embedder Module.
        self.NoiseEmbedder = NoiseEmbedder(num_conditions, hidden_dim = units, embedding_dim = embedding_dim,
                                             dropout_rate = dropout_rate)
        
        # Create the Encoder Pathway.
        self.encoder = nn.ModuleList()
        
        # Define the Encoder Blocks.
        for index in range(depth):
            in_channels = channels * levels if index == 0 else out_channels
            out_channels = self.starting_filters if index == 0 else out_channels * 2  
            for block in range(num_blocks_per_layer):
                if block != self.num_blocks_per_layer - 1:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                             pooling = False, num_layers = num_layers, 
                                                             activation = activation, dropout_rate = dropout_rate,
                                                             adain_enabled = adain_enabled, film_enabled = film_enabled,
                                                             embedding_dim=embedding_dim))
                else:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                         pooling = True, num_layers = num_layers, 
                                                         activation = activation, dropout_rate = dropout_rate,
                                                         adain_enabled = adain_enabled, film_enabled = film_enabled,
                                                         embedding_dim=embedding_dim))
                in_channels = out_channels
                    
        # Create the Bottleneck Pathway.
        self.Bottleneck_Block = nn.ModuleList()
        pooling = False

        self.Bottleneck_Block.append(DownConvolutionBlock(out_channels, 2*out_channels,
                                                            pooling = pooling, num_layers = num_layers,
                                                            activation = activation, dropout_rate = dropout_rate,
                                                            adain_enabled = adain_enabled, film_enabled = film_enabled,
                                                            embedding_dim=embedding_dim))
        
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
                                                           dropout_rate = dropout_rate, adain_enabled = adain_enabled,
                                                           film_enabled = film_enabled, embedding_dim=embedding_dim))
                else:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = False,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate, film_enabled=film_enabled,
                                                           adain_enabled = adain_enabled, embedding_dim=embedding_dim))
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

        stack = None
        factor = 10.0
        psnr_values = psnr_map[:,0,0,0].view(-1,1)
        normalised_psnr = (psnr_values - self.minpsnr)/(self.maxpsnr - self.minpsnr)
        embedding = self.NoiseEmbedder(normalised_psnr)

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
            out = Encoder_Block(x, embedding)

            if len(out) == 2:
                x, before_pool = out
                encoder[current_depth] = before_pool
                current_depth += 1
            else:
                x = out
        
        # Pass the encoding through the bottleneck of the networ
        for index, Bottleneck_Block in enumerate(self.Bottleneck_Block):
            x = Bottleneck_Block(x, embedding)
        
        # Begin by decoding the input image representation.
        index = 0
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool, embedding)
                current_depth -= 1
                index+=1
            else:
                x = Decoder_Block(x, None, embedding)
                index+=1
    
        # Pass the image representation through the output layer.
        output = self.output(x)
        return output

    def predict(self, x, psnr_map):
        return self.forward(x, psnr_map)
    
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
        img_input, psnr_map, target_img  = batch
        predicted = self.forward(img_input, psnr_map)
        train_loss = self.photonLoss(predicted, target_img)
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss
    
    def validation_step(self, batch, batch_idx = None):
        img_input, psnr_map, target_img = batch
        predicted = self.forward(img_input, psnr_map)
        valid_loss = self.photonLoss(predicted, target_img)
        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss
    
    def test_step(self, batch, batch_idx = None):
        img_input, psnr_map, target_img = batch
        predicted = self.forward(img_input, psnr_map)
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
    

    




    

    
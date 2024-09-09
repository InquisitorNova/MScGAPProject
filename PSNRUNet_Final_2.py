"""
This is a modified Res-UNet implementation that builds upon the work of Jackson Haung
(https://github.com/jaxony/unet-pytorch/)
and Alex Krull. 
https://github.com/krulllab/GAP/blob/main/gap/GAP_UNET_ResBlock.py
I have modifield the UNet to be incorporated into the GAP Framework, adding additional features
such as the ability to use different activation functions, and the ability to use different
amounts of layers in the ResBlock. I have also added the ability to use GroupNorm layers in the ResBlock
and the ability to use a different number of channels per group in the GroupNorm layers.
For this UNet implementation, I have created a positional embedder to cinvert the PSNR value into a timestep
and then from a timestep into a positional embedding. This is used to encode the PSNR value into the image representation.
This is then used to condition the network on the PSNR value. This network is designed to be used with 2D image data.
and based on the the UNet architecture introduced in the original diffusion paper (https://arxiv.org/pdf/2006.11239)
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

# Define the Device Being Used:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# Introduced the PSNR to Timestep and Timestep to Positional Embedding transformations,
# as well as the Temporal Embedder class. These classes are used to encode the PSNR value
# into the image representation and condition the network on the PSNR value.
# Define the transformation for converting psnr into timesteps.
def psnr_to_timestep(psnr_map, minpsnr, maxpsnr, num_timesteps):
    """
    psnr - the psnr value to convert to a timestep
    minpsnr - the minimum psnr value that can be achieved
    maxpsnr - the maximum psnr value that can be achieved
    num_timesteps - the number of timesteps in the simulation
    This function takes a psnr as import, performs min-max normalisation and 
    then converts it into a timestep.
    """

    psnr_batch = psnr_map[:, 0,0,0]

    psnr_batch = torch.clamp(psnr_batch, minpsnr.item(), maxpsnr.item())
    normalised_psnr = ((psnr_batch - minpsnr) / (maxpsnr - minpsnr))
    timesteps = (num_timesteps-1)*normalised_psnr
    timesteps = torch.round(timesteps)
    return timesteps


# Define the transformation for converting the timestep into a positional embedding.
class Temporal_Embedder(nn.Module):
    """
    This is a class that defines the Temporal Embedder for the GAP Framework.
    It is based off the positional embeddings used in the Transformer architecture.
    It takes in a timestep and outputs a positional embedding.
    The code is based off the code from the PyTorch Transformer Library.
    """
    def __init__(self, n_channels):
        super(Temporal_Embedder, self).__init__()
        self.n_channels = n_channels
        self.Linear_1 = nn.Linear(self.n_channels//4, self.n_channels)
        self.Linear_2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels//8
        constant = torch.FloatTensor([10000.0])
        emb = torch.log(constant) / (half_dim -1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = 1)
        
        emb = F.leaky_relu(self.Linear_1(emb))
        emb = F.leaky_relu(self.Linear_2(emb))
        return emb

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
                 activation = nn.LeakyReLU(), dropout_rate = 0.5, time_channels = 32,
                 concatenate = False):
        super(DownConvolutionBlock, self).__init__()

        # Initialise the layers of the block.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.concatenate = concatenate
        self.time_channels = time_channels
        self.out_time_channels = 1

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
        
        Main_Processing_Block.append(nn.Conv2d(out_channels + self.out_time_channels if concatenate else out_channels, out_channels, kernel_size = 3, padding = 1))
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

        self.time_embedding = nn.Linear(time_channels, self.out_time_channels)
    
    def forward(self, x, t):
        batch_size, _, height, width = x.shape
        x_skip = self.Skip_Block(x)

        if self.concatenate:
            h = F.leaky_relu(self.time_embedding(t)).unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.out_time_channels, height, width)
            x = torch.cat((x_skip,h), 1)
        else:
            h = F.leaky_relu(self.time_embedding(t)[:,:,None,None])
            x = h + x_skip

        x = self.Main_Processing_Block(x)

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
                 up_mode = "upsample", num_layers = 2, activation = nn.LeakyReLU(),
                 dropout_rate = 0.5, concatenate = False, time_channels = 32):
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
        self.time_channels = time_channels
        self.out_time_channels = 1
        self.concatenate = concatenate

        # Create the layers of the block.
        n_groups = find_group_number(out_channels, min_group_channels = 4, max_group_channels = 32)

        # The Skip Block provides the input and skip connection to the ResBlock.

        if "concat" in merge_mode and self.upsampling:
            self.Skip_Block = nn.Sequential(
                nn.Conv2d(2*out_channels + self.out_time_channels if concatenate else 2*out_channels, out_channels, kernel_size = 3, padding = 1),
                nn.GroupNorm(n_groups, out_channels),
                activation,
            )
        else:
            self.Skip_Block = nn.Sequential(
                nn.Conv2d(out_channels + self.out_time_channels if concatenate else out_channels, out_channels, kernel_size = 3, padding = 1),
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

        self.time_embedding = nn.Linear(time_channels, self.out_time_channels)

    def forward(self, below, above, t):

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

        batch_size, channels, height, width = x.shape

        if self.concatenate:
            h = F.leaky_relu(self.time_embedding(t)).unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.out_time_channels, height, width)
            #print(h.shape, x.shape)
            x = torch.cat((x,h), 1)
        else:
            h = F.silu(self.time_embedding(t)[:,:,None,None])
            x += h

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
    """

    def __init__(
            self,
            levels: int = 10,
            channels: int = 3,
            initial_channels: int = 4,
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
            concatenate: bool = False,
            maxpsnr: int = 40,
            minpsnr: int = -40, 
            num_timesteps: int = 160,
            time_channels: int = 32,
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
        self.concatenate = concatenate
        self.time_channels = time_channels
        self.initial_channels = initial_channels
        
        self.time_channels = time_channels
        self.minpsnr = torch.FloatTensor([minpsnr]).to(device)
        self.maxpsnr = torch.FloatTensor([maxpsnr]).to(device)
        self.num_timesteps = num_timesteps


        # Define the functions needed to create the timestep and temporal embedding.
        self.Psnr_Converter = lambda psnr: psnr_to_timestep(psnr, self.maxpsnr, self.minpsnr, self.num_timesteps)
        self.Temporal_Embedding = Temporal_Embedder(self.time_channels)
        
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
                                                             concatenate = concatenate, time_channels = time_channels))
                else:
                    self.encoder.append(DownConvolutionBlock(in_channels, out_channels, 
                                                         pooling = True, num_layers = num_layers, 
                                                         activation = activation, dropout_rate = dropout_rate,
                                                         concatenate = concatenate, time_channels = time_channels))
                    
                in_channels = out_channels
                    
        # Create the Bottleneck Pathway.
        self.Bottleneck_Block = nn.ModuleList()
        pooling = False

        self.Bottleneck_Block.append(DownConvolutionBlock(out_channels, 2*out_channels,
                                                            pooling = pooling, num_layers = num_layers,
                                                            activation = activation, dropout_rate = dropout_rate,
                                                            concatenate = concatenate, time_channels = time_channels))
    
        out_channels = out_channels * 2
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
                                                           concatenate = concatenate, time_channels = time_channels))
                else:
                    self.decoder.append(UpConvolutionBlock(in_channels, out_channels, upsampling = False,
                                                           merge_mode = merge_mode, up_mode = up_mode,
                                                           num_layers = num_layers, activation = activation,
                                                           dropout_rate = dropout_rate,
                                                           concatenate = concatenate, time_channels = time_channels))
                    
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

    def forward(self, x, psnr_image):

        stack = None
        factor = 10.0
        t = self.Psnr_Converter(psnr_image)
        t = self.Temporal_Embedding(t)

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
            out = Encoder_Block(x, t)

            if len(out) == 2:
                x, before_pool = out
                encoder[current_depth] = before_pool
                current_depth += 1
            else:
                x = out
        
        # Pass the encoding through the bottleneck of the networ
        for index, Bottleneck_Block in enumerate(self.Bottleneck_Block):
            x = Bottleneck_Block(x, t)
        
        # Begin by decoding the input image representation.
        index = 0
        current_depth = self.depth -1
        while index <= len(self.decoder)-1:
            Decoder_Block = self.decoder[index]
            if Decoder_Block.upsampling:
                before_pool = encoder[current_depth]
                x = Decoder_Block(x, before_pool, t)
                current_depth -= 1
                index+=1
            else:
                x = Decoder_Block(x, None, t)
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
        img_input, psnr_maps, target_img  = batch
        predicted = self.forward(img_input, psnr_maps)
        train_loss = self.photonLoss(predicted, target_img)
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", learning_rate, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return train_loss
    
    def validation_step(self, batch, batch_idx = None):
        img_input, psnr_maps, target_img = batch
        predicted = self.forward(img_input, psnr_maps)
        valid_loss = self.photonLoss(predicted, target_img)
        self.log("val_loss", valid_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return valid_loss
    
    def test_step(self, batch, batch_idx = None):
        img_input, psnr_maps, target_img = batch
        predicted = self.forward(img_input, psnr_maps)
        test_loss = self.photonLoss(predicted, target_img)
        self.log("test_loss", test_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return test_loss
    
    def configure_optimizers(self):
        num_warm_steps = self.mini_batches * self.warm_up_epochs
        num_training_steps = self.mini_batches * self.epochs

        # Changed from the Adam Optimizer to the AdamW Optimizer t improve performance.
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = 1e-4)


        # Introduced the OneCycleLR Scheduler to improve performance.
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
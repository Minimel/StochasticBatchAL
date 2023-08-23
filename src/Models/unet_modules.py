"""
Author: Mélanie Gaillochet

Adapted from https://github.com/hiepph/unet-lightning/blob/master/Unet.py
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_shape(image, shp):
    """
    We pad the input image with zeroes to given shape.
    :param image: the tensor image that we want to pad (has to have dimension 5)
    :param shp: the desired output shape
    :return: zero-padded tensor
    """
    # Pad is a list of length 2 * len(source.shape) stating how many dimensions
    # should be added to the beginning and end of each axis.
    pad = []
    for i in range(len(image.shape) - 1, 1, -1):
        pad.extend([0, shp[i] - image.shape[i]])

    padded_img = torch.nn.functional.pad(image, pad)

    return padded_img

##### Alternative types of 2D convolutions #####
class WSConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


####### 2D UNet blocks ########
def conv_bloc(in_channels, out_channels, **kwargs):
    """
    We create a blocks with 3d convolution, batch norm, and ReLU activation
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    """
    conv_type = kwargs.get('conv_type', 'conv2d') # conv2d (normal), separable_conv2d, WS_conv2d
    normalization = kwargs.get('normalization', None)
    activation_fct = kwargs.get('activation_fct', None)
    kernel_size = kwargs.get('kernel_size', 3)
    stride = kwargs.get('stride', 1)
    padding = kwargs.get('padding', 1)
    momentum = kwargs.get('batch_norm_momentum')
    num_group_norm = kwargs.get('num_group_norm', 32)
    #weight_standardization = kwargs.get('weight_standardization', False)

    # We define type of batch normalization
    if normalization == "batch_norm":
        batch_norm = nn.BatchNorm2d(out_channels, momentum=momentum)
    elif normalization == "group_norm":
        batch_norm = nn.GroupNorm(num_group_norm, out_channels)

    # We define the activation function
    if activation_fct == "leakyReLU":
        activation = nn.LeakyReLU(inplace=True)
    elif activation_fct == "ReLU":
        activation = nn.ReLU(inplace=True)

    # We define type of convolution
    if conv_type == 'conv2d':
        conv_2d = nn.Conv2d
    elif conv_type == 'separable_conv2d':
        conv_2d = SeparableConv2d
    elif conv_type == 'WS_conv2d':
        conv_2d = WSConv2d

    conv_layer = [
        conv_2d(in_channels, out_channels, kernel_size, stride, padding),
        batch_norm,
        activation
    ]
    conv_layer = nn.Sequential(*conv_layer)

    return conv_layer


def double_conv_bloc(in_channels, middle_channels, out_channels, **kwargs):
    """
    We create a layer with 2 convolution blocks (3d conv., BN, and activation)
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    """
    conv_layer = [
        conv_bloc(in_channels, middle_channels, **kwargs),
        conv_bloc(middle_channels, out_channels, **kwargs)
    ]
    conv_layer = nn.Sequential(*conv_layer)

    return conv_layer


def max_pooling_2d(**kwargs):
    """
    We apply a max pooling with kernel size 2x2
    :return:
    """
    kernel_size = kwargs.get('kernel_size', 2)
    stride = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 0)

    return nn.MaxPool2d(kernel_size, stride, padding)


def up_conv_2d(in_channels, out_channels, **kwargs):
    """
    We apply and upsampling-convolution with kernel size 2x2
    :param in_channels: # of input channels
    :param out_channels: number of up-convolution channels
    :return:
    """
    kernel_size = kwargs.get('kernel_size', 2)
    stride = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 0)

    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                              padding)
                   

class encoderbottleneckUNet(pl.LightningModule):
    def __init__(self, config):
        """
        This modified UNet differs from the original one with the use of
        leaky relu (instead of ReLU) and the addition of residual connections.
        The idea is to help deal with fine-grained details
        :param in_channels: # of input channels (ie: 3 if  image in RGB)
        :param out_channels: # of output channels (# segmentation classes)
        """
        super(encoderbottleneckUNet, self).__init__()

        self.in_channels = config["in_channels"]
        self.channel_list = config["channel_list"] # channel_list=[64, 128, 256, 512]
        
        # Kernel size, stride, paddding, normalizatin and activation function will be passed as keyword arguments
        kwargs = config['structure']
            
       # We set the dropout
        self.dropout = nn.Dropout(kwargs['dropout_rate'])

        # Encoder part
        self.enc_1 = double_conv_bloc(self.in_channels,
                                      self.channel_list[0]//2,
                                      self.channel_list[0],
                                      **kwargs['conv_block'])
        self.pool_1 = max_pooling_2d(**kwargs['pooling'])

        self.enc_2 = double_conv_bloc(self.channel_list[0],
                                      self.channel_list[0],
                                      self.channel_list[1],
                                      **kwargs['conv_block'])
        self.pool_2 = max_pooling_2d(**kwargs['pooling'])
        self.enc_3 = double_conv_bloc(self.channel_list[1],
                                      self.channel_list[1],
                                      self.channel_list[2],
                                      **kwargs['conv_block'])
        self.pool_3 = max_pooling_2d(**kwargs['pooling'])
        
        # Bottleneck part
        self.center = double_conv_bloc(self.channel_list[-2],
                                       self.channel_list[-2],
                                       self.channel_list[-1],
                                       **kwargs['conv_block'])

    def forward(self, x):
        # Encoding
        enc_1 = self.enc_1(x)  # -> [BS, 64, x, y, z], if num_init_filters=64
        out = self.pool_1(enc_1)  # -> [BS, 64, x/2, y/2, z/2]

        # We put a dropout layer
        out = self.dropout(out)

        # print('\nEncoder 2')
        enc_2 = self.enc_2(out)  # -> [BS, 128, x/2, y/2, z/2]
        out = self.pool_2(enc_2)  # -> [BS, 128, x/4, y/4, z/4]

        # We put a dropout layer
        out = self.dropout(out)

        # print('\nEncoder 3')
        enc_3 = self.enc_3(out)  # -> [BS, 256, x/4, y/4, z/4]
        out = self.pool_3(enc_3)  # -> [BS, 256, x/8, y/8, z/8]

        # We put a dropout layer
        out = self.dropout(out)
        
        # Bottleneck
        center = self.center(out)  # -> [BS, 512, x/8, y/8, z/8]
        
        return center, [enc_1, enc_2, enc_3]


class bottleneckUNet(pl.LightningModule):
    def __init__(self, config) -> None:
        super(bottleneckUNet, self).__init__()
        self.channel_list = config["channel_list"] # channel_list=[64, 128, 256, 512]
        
        # Kernel size, stride, paddding, normalizatin and activation function will be passed as keyword arguments
        kwargs = config['structure']

        # Bottleneck part
        self.center = double_conv_bloc(self.channel_list[-2],
                                       self.channel_list[-2],
                                       self.channel_list[-1],
                                       **kwargs['conv_block'])
        
    def forward(self, x):
        out = self.center(x)  # -> [BS, 512, x/8, y/8, z/8]
        return out


class decoderUNet(pl.LightningModule):
    def __init__(self, config):
        """
        This modified UNet differs from the original one with the use of
        leaky relu (instead of ReLU) and the addition of residual connections.
        The idea is to help deal with fine-grained details
        :param in_channels: # of input channels (ie: 3 if  image in RGB)
        :param out_channels: # of output channels (# segmentation classes)
        """
        super(decoderUNet, self).__init__()

        self.out_channels = config["out_channels"]
        self.channel_list = config["channel_list"] # channel_list=[64, 128, 256, 512]
        
        # Kernel size, stride, paddding, normalizatin and activation function will be passed as keyword arguments
        kwargs = config['structure']
        
        # We set the dropout
        self.dropout = nn.Dropout(kwargs['dropout_rate'])
        
        # Decoder part
        self.up_1 = up_conv_2d(self.channel_list[-1],
                               self.channel_list[-1], 
                               **kwargs['upconv'])
        self.dec_1 = double_conv_bloc(self.channel_list[-1] + self.channel_list[-2],
                                      self.channel_list[-2],
                                      self.channel_list[-2],
                                      **kwargs['conv_block'])
        self.up_2 = up_conv_2d(self.channel_list[-2],
                               self.channel_list[-2], 
                               **kwargs['upconv'])
        self.dec_2 = double_conv_bloc(self.channel_list[-2] + self.channel_list[-3],
                                      self.channel_list[-3],
                                      self.channel_list[-3],
                                      **kwargs['conv_block'])
        self.up_3 = up_conv_2d(self.channel_list[-3],
                               self.channel_list[-3], 
                               **kwargs['upconv'])
        self.dec_3 = double_conv_bloc(self.channel_list[-3] + self.channel_list[-4],
                                      self.channel_list[-4],
                                      self.channel_list[-4],
                                      **kwargs['conv_block'])

        # Output
        self.out = nn.Conv2d(self.channel_list[0], self.out_channels,
                             kernel_size=1, padding=0)


    def forward(self, x, enc_1, enc_2, enc_3):
        # Decoding
        out = self.up_1(x)  # -> [BS, 512, x/4, y/4, z/4]
        out = pad_to_shape(out, enc_3.shape)
        out = torch.cat([out, enc_3], dim=1)  # -> [BS, 768, x/4, y/4, z/4]
        out = self.dropout(out)
        dec_1 = self.dec_1(out)  # -> [BS, 256, x/4, y/4, z/4]

        out = self.up_2(dec_1)  # -> [BS, 256, x/2, y/2, z/2]
        out = pad_to_shape(out, enc_2.shape)
        out = torch.cat([out, enc_2], dim=1)  # -> [BS, 384, x/2, y/2, z/2]
        out = self.dropout(out)
        dec_2 = self.dec_2(out)  # -> [BS, 128, x/2, y/2, z/2]

        out = self.up_3(dec_2)  # -> [BS, 128, x, y, z]
        out = pad_to_shape(out, enc_1.shape)
        out = torch.cat([out, enc_1], dim=1)  # -> [BS, 192, x, y, z]
        out = self.dropout(out)
        dec_3 = self.dec_3(out)  # -> [BS, 64, x, y, z]

        # Output
        out = self.out(dec_3)  # -> [BS, out_channels, x, y, z]
        
        return out, [dec_1, dec_2, dec_3]

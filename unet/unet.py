from typing import Union
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


class UnetDownModule(nn.Module):
    """ U-Net downsampling block. """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 downsample: bool = True):
        super().__init__()

        # layers: optional downsampling, 2 x (conv + bn + relu)
        self.maxpool = nn.MaxPool2d((2, 2)) if downsample else None
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UnetEncoder(nn.Module):
    """ U-Net encoder. https://arxiv.org/pdf/1505.04597.pdf """

    def __init__(self, num_channels: int):
        super().__init__()
        self.module1 = UnetDownModule(num_channels, 64, downsample=False)
        self.module2 = UnetDownModule(64, 128)
        self.module3 = UnetDownModule(128, 256)
        self.module4 = UnetDownModule(256, 512)
        self.module5 = UnetDownModule(512, 1024)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        return x


def get_backbone(name: str,
                 pretrained: Union[str, bool, None] = None):
    """ Loading backbone, defining names for skip-connections and encoder
    output. """

    if isinstance(pretrained, bool) and pretrained:
        pretrained = 'DEFAULT'

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(weights=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(weights=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(weights=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(weights=pretrained)
    elif name == 'resnet152':
        backbone = models.resnet152(weights=pretrained)
    elif name == 'vgg16':
        backbone = models.vgg16_bn(weights=pretrained).features
    elif name == 'vgg19':
        backbone = models.vgg19_bn(weights=pretrained).features
    elif name == 'inception_v3':
        backbone = models.inception_v3(weights=pretrained, aux_logits=False)
    elif name == 'densenet121':
        backbone = models.densenet121(weights=pretrained).features
    elif name == 'densenet161':
        backbone = models.densenet161(weights=pretrained).features
    elif name == 'densenet169':
        backbone = models.densenet169(weights=pretrained).features
    elif name == 'densenet201':
        backbone = models.densenet201(weights=pretrained).features
    elif name == 'unet_encoder':
        backbone = UnetEncoder(3)
    else:
        raise NotImplemented(
            '{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    elif name == 'inception_v3':
        feature_names = [None, 'Mixed_5d', 'Mixed_6e']
        backbone_output = 'Mixed_7c'
    elif name.startswith('densenet'):
        feature_names = [
            None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented(
            '{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0,
                 use_bn=True, parametric=False):
        super().__init__()

        self.parametric = parametric
        ch_out = ch_in / 2 if ch_out is None else ch_out

        # first convolution: either transposed conv,
        # or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        if self.parameters:
            x = self.up(x)
        else:
            x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                              align_corners=None)

        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class Unet(nn.Module):
    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with
    pre-trained torchvision backbones. """

    def __init__(self,
                 backbone_name: str = 'resnet50',
                 pretrained: Union[str, bool, None] = True,
                 encoder_freeze: bool = False,
                 classes: int = 21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling: bool = True,
                 shortcut_features='default',
                 decoder_use_batchnorm: bool = True):
        super().__init__()

        self.backbone_name = backbone_name
        self.decoder_filters = decoder_filters
        self.parametric_upsampling = parametric_upsampling
        self.decoder_use_batchnorm = decoder_use_batchnorm

        self.backbone, self.shortcut_features, self.bb_out_name = \
            get_backbone(backbone_name, pretrained=pretrained)
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = self.get_decoder()
        self.final_conv = nn.Conv2d(
            decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        # for accommodating inputs with different number of channels later
        self.replaced_conv1 = False

    def get_decoder(self):
        """build decoder"""
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        upsample_blocks = nn.ModuleList()
        # avoiding having more blocks than skip connections
        decoder_filters = self.decoder_filters[:len(self.shortcut_features)]

        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(
                zip(decoder_filters_in, decoder_filters)):

            upsample_blocks.append(
                UpsampleBlock(
                    filters_in, filters_out,
                    skip_in=shortcut_chs[num_blocks - i - 1],
                    parametric=self.parametric_upsampling,
                    use_bn=self.decoder_use_batchnorm
                )
            )
        return upsample_blocks

    def freeze_encoder(self):
        """ Freezing encoder parameters, the newly initialized decoder
        parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *inputs):
        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*inputs)

        for skip_name, upsample_block in zip(
                self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):
        """ Getting the number of channels at skip connections and at the
        output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = \
            self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        # only VGG has features at full resolution
        channels = [] if has_fullres_features else [0]

        # forward run in backbone to count channels
        # (dirty solution but works for *any* Module)
        out_channels = None
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels


class UnetTwoHead(Unet):
    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 classes_1=21,
                 classes_2=21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        nn.Module.__init__(self)

        self.backbone_name = backbone_name
        self.decoder_filters = decoder_filters
        self.parametric_upsampling = parametric_upsampling
        self.decoder_use_batchnorm = decoder_use_batchnorm

        self.backbone, self.shortcut_features, self.bb_out_name = \
            get_backbone(backbone_name, pretrained=pretrained)
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks1 = self.get_decoder()
        self.final_conv1 = nn.Conv2d(
            decoder_filters[-1], classes_1, kernel_size=(1, 1))

        self.upsample_blocks2 = self.get_decoder()
        self.final_conv2 = nn.Conv2d(
            decoder_filters[-1], classes_2, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        # for accommodating inputs with different number of channels later
        self.replaced_conv1 = False

    def forward(self, *inputs):
        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*inputs)

        x1 = x
        x2 = x

        for skip_name, upsample_block in zip(
                self.shortcut_features[::-1], self.upsample_blocks1):
            skip_features = features[skip_name]
            x1 = upsample_block(x1, skip_features)
        x1 = self.final_conv1(x1)

        for skip_name, upsample_block in zip(
                self.shortcut_features[::-1], self.upsample_blocks1):
            skip_features = features[skip_name]
            x2 = upsample_block(x2, skip_features)
        x2 = self.final_conv2(x2)
        return x1, x2

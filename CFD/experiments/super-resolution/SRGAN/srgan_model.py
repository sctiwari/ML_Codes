import copy
import torch
import torch.nn as nn
import torch.nn.functional as nnf


def mul_sigmoid(x):
    return x * torch.sigmoid(x)


class ResidualNetwork(nn.Module):
    def __init__(self, input, kernel_size, output, stride):
        super(ResidualNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input,
                               out_channels=output,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(num_features=output)
        self.conv2 = nn.Conv2d(in_channels=output,
                               out_channels=output,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(num_features=output)

    def forward(self, input):
        #y = mul_sigmoid((self.conv1(input)))
        #return (self.conv2(y)) + input

        y = mul_sigmoid(self.bn1(self.conv1(input)))
        return self.bn2(self.conv2(y)) + input


class UpSampleNetwork(nn.Module):
    def __init__(self, input):
        super(UpSampleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input,
                               out_channels=input * 4,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, input):
        return mul_sigmoid(self.shuffle(self.conv1(input)))


class Generator(nn.Module):
    def __init__(self, num_residual, upscale_factor, num_input=1, base_filter=64):
        super(Generator, self).__init__()
        self.num_residual = num_residual
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(in_channels=num_input,
                               out_channels=base_filter,
                               kernel_size=9,
                               stride=1,
                               padding=4)
        for i in range(self.num_residual):
            self.add_module('residual' + str(i + 1),
                            ResidualNetwork(input=base_filter,
                                            output=base_filter,
                                            kernel_size=3,
                                            stride=1))
        self.conv2 = nn.Conv2d(in_channels=base_filter,
                               out_channels=base_filter,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=base_filter)
        for i in range(self.upscale_factor // 2):
            self.add_module('upscale' + str(i + 1),
                            UpSampleNetwork(base_filter))
        self.conv3 = nn.Conv2d(in_channels=base_filter,
                               out_channels=num_input,
                               kernel_size=9,
                               stride=1,
                               padding=4)

    def forward(self, input):
        input = mul_sigmoid(self.conv1(input))

        x = input.clone()

        for i in range(self.num_residual):
            x = self.__getattr__('residual' + str(i + 1))(x)

        input = self.bn2(self.conv2(x)) + input

        for i in range(self.upscale_factor // 2):
            input = self.__getattr__('upscale' + str(i + 1))(input)

        return self.conv3(input)

    def weight_init(self, mean=0.0, std=0.02):
        for i in self._modules:
            if isinstance(i, nn.ConvTranspose2d) or isinstance(i, nn.Conv2d):
                i.weight.data.normal_(mean, std)
                i.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, num_input=1, base_filter=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_input,
                               out_channels=base_filter,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=base_filter,
                               out_channels=base_filter,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=base_filter)
        self.conv3 = nn.Conv2d(in_channels=base_filter,
                               out_channels=base_filter * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=base_filter * 2)
        self.conv4 = nn.Conv2d(in_channels=base_filter * 2,
                               out_channels=base_filter * 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=base_filter * 2)
        self.conv5 = nn.Conv2d(in_channels=base_filter * 2,
                               out_channels=base_filter * 4,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=base_filter * 4)
        self.conv6 = nn.Conv2d(in_channels=base_filter * 4,
                               out_channels=base_filter * 4,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=base_filter * 4)
        self.conv7 = nn.Conv2d(in_channels=base_filter * 4,
                               out_channels=base_filter * 8,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=base_filter * 8)
        self.conv8 = nn.Conv2d(in_channels=base_filter * 8,
                               out_channels=base_filter * 8,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=base_filter * 8)

        self.conv9 = nn.Conv2d(in_channels=base_filter * 8,
                               out_channels=num_input,
                               kernel_size=1,
                               stride=1,
                               padding=0)

    def forward(self, input):
        t = mul_sigmoid(self.conv1(input))
        #'''
        t = mul_sigmoid(self.bn2(self.conv2(t)))
        t = mul_sigmoid(self.bn3(self.conv3(t)))
        t = mul_sigmoid(self.bn4(self.conv4(t)))
        t = mul_sigmoid(self.bn5(self.conv5(t)))
        t = mul_sigmoid(self.bn6(self.conv6(t)))
        t = mul_sigmoid(self.bn7(self.conv7(t)))
        t = mul_sigmoid(self.bn8(self.conv8(t)))
        #'''
        '''
        t = mul_sigmoid((self.conv2(t)))
        t = mul_sigmoid((self.conv3(t)))
        t = mul_sigmoid((self.conv4(t)))
        t = mul_sigmoid((self.conv5(t)))
        t = mul_sigmoid((self.conv6(t)))
        t = mul_sigmoid((self.conv7(t)))
        t = mul_sigmoid((self.conv8(t)))
        '''
        t = self.conv9(t)
        return torch.sigmoid(nnf.avg_pool2d(t, t.size()[2:])).view(t.size()[0], -1)

    def weight_init(self, mean=0.0, std=0.02):
        for i in self._modules:
            if isinstance(i, nn.ConvTranspose2d) or isinstance(i, nn.Conv2d):
                i.weight.data.normal_(mean, std)
                i.bias.data.zero_()
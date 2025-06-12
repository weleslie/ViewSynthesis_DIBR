import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

CHANNEL = 32

class BinocularEncoder(nn.Module):
    def __init__(self, input_channels=None):
        super(BinocularEncoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, CHANNEL, 5, stride=1, padding=2),
                                   nn.LeakyReLU(negative_slope=0.1))  # 1

        self.conv2a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 5, stride=2, padding=2),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/2
        self.conv2b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/2

        self.pool3 = nn.MaxPool2d(2)  # 1/4
        self.conv3a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/4
        self.conv3b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/4

        self.pool4 = nn.MaxPool2d(2)  # 1/8
        self.conv4a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/8
        self.conv4b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/8

        self.pool5 = nn.MaxPool2d(2)  # 1/16
        self.conv5a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/16
        self.conv5b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/16

        self.pool6 = nn.MaxPool2d(2)  # 1/32
        self.conv6a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/32
        self.conv6b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1))  # 1/32

        # self.pool7 = nn.MaxPool2d(2)  # 1/64
        # self.conv7a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
        #                             nn.LeakyReLU(negative_slope=0.1))  # 1/64
        # self.conv7b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
        #                             nn.LeakyReLU(negative_slope=0.1))
        #
        # self.pool8 = nn.MaxPool2d(2)  # 1/64
        # self.conv8a = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
        #                             nn.LeakyReLU(negative_slope=0.1))  # 1/128
        # self.conv8b = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, 3, stride=1, padding=1),
        #                             nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2a = self.conv2a(conv1)
        conv2b = self.conv2b(conv2a)
        pool3 = self.pool3(conv2b)
        conv3a = self.conv3a(pool3)
        conv3b = self.conv3b(conv3a)
        pool4 = self.pool4(conv3b)
        conv4a = self.conv4a(pool4)
        conv4b = self.conv4b(conv4a)
        pool5 = self.pool5(conv4b)
        conv5a = self.conv5a(pool5)
        conv5b = self.conv5b(conv5a)
        pool6 = self.pool6(conv5b)
        conv6a = self.conv6a(pool6)
        conv6b = self.conv6b(conv6a)

        # pool7 = self.pool7(conv6b)
        # conv7a = self.conv7a(pool7)
        # conv7b = self.conv7b(conv7a)
        #
        # pool8 = self.pool8(conv7b)
        # conv8a = self.conv8a(pool8)
        # conv8b = self.conv8b(conv8a)

        # return (conv4b, conv3b, conv2b, conv1)
        # return (conv5b, conv4b, conv3b, conv2b, conv1)
        return (conv6b, conv5b, conv4b, conv3b, conv2b, conv1)
        # return (conv7b, conv6b, conv5b, conv4b, conv3b, conv2b, conv1)
        # return (conv8b, conv7b, conv6b, conv5b, conv4b, conv3b, conv2b, conv1)


class MultiscaleDecoder(nn.Module):
    def __init__(self):
        super(MultiscaleDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Upsample(scale_factor=2, mode='nearest'))  # 1/32

        self.conv2 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Upsample(scale_factor=2, mode='nearest'))  # 1/16
        self.flow1 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(CHANNEL, 2, kernel_size=3, stride=1, padding=1))

        self.conv3 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Upsample(scale_factor=2, mode='nearest'))  # 1/8
        self.flow2 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(CHANNEL, 2, kernel_size=3, stride=1, padding=1))

        self.conv4 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Upsample(scale_factor=2, mode='nearest'))  # 1/4
        self.flow3 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(CHANNEL, 2, kernel_size=3, stride=1, padding=1))

        # self.conv5 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
        #                            nn.LeakyReLU(negative_slope=0.1),
        #                            nn.Upsample(scale_factor=2, mode='nearest'))  # 1/2
        # self.flow4 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
        #                            nn.Conv2d(CHANNEL, 2, kernel_size=3, stride=1, padding=1))
        #
        # self.conv6 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
        #                            nn.LeakyReLU(negative_slope=0.1),
        #                            nn.Upsample(scale_factor=2, mode='nearest'))  # 1/2
        # self.flow5 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
        #                            nn.Conv2d(CHANNEL, 2, kernel_size=3, stride=1, padding=1))

        self.conv5 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Upsample(scale_factor=2, mode='nearest'))  # 1/2
        self.flow4 = nn.Sequential(nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(CHANNEL, 16, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, binocular_feature):
        integration0 = binocular_feature[0]  # conv6b
        conv1 = self.conv1(integration0)

        integration1 = binocular_feature[1] + conv1  # conv5b
        conv2 = self.conv2(integration1)
        flow1 = self.flow1(conv2)

        integration2 = binocular_feature[2] + conv2  # conv4b
        conv3 = self.conv3(integration2)
        flow2 = self.flow2(conv3)

        integration3 = binocular_feature[3] + conv3  # conv3b
        conv4 = self.conv4(integration3)
        flow3 = self.flow3(conv4)

        integration4 = binocular_feature[4] + conv4  # conv2b
        conv5 = self.conv5(integration4)
        flow4 = self.flow4(conv5)

        # integration5 = binocular_feature[5] + conv5  # conv2b
        # conv6 = self.conv6(integration5)
        # flow5 = self.flow5(conv6)
        #
        # integration6 = binocular_feature[6] + conv6  # conv2b
        # conv7 = self.conv6(integration6)
        # flow6 = self.flow5(conv7)

        # return (flow5, flow1)
        # return (flow5, flow2, flow1)
        return (flow4, flow3, flow2, flow1)
        # return (flow5, flow4, flow3, flow2, flow1)
        # return (flow6, flow5, flow4, flow3, flow2, flow1)


class FlowWarp(nn.Module):
    def __init__(self):
        super(FlowWarp, self).__init__()

    def forward(self, imgrs, flows):
        warped_imgls = []
        for i in range(len(imgrs)):
            warped_imgls.append(self.imgr2imgl_with_flow(imgrs[i], -flows[i]))
        return warped_imgls

    def imgr2imgl_with_flow(self, imgr, flow):
        batch_size, dim, height, width = flow.size()  # flow size: Bx2xHxW

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(imgr)
        y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(imgr)

        # Apply shift in X direction
        x_shifts = flow[:, 0, :, :] / width  # Normalize the U dimension
        y_shifts = flow[:, 1, :, :] / height  # Normalize the V dimension
        flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3) # [B, H, W, 2]
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(imgr, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

        return output
import torch.nn as nn
import torch.nn.functional as F
from models.modules import BinocularEncoder, MultiscaleDecoder, FlowWarp


class MultiscaleFlow(nn.Module):
    def __init__(self, binocualr_dim=None, training=True):
        super(MultiscaleFlow, self).__init__()
        self.binocular_encoder = BinocularEncoder(input_channels=binocualr_dim)
        self.mutliscale_decoder = MultiscaleDecoder()
        self.flow_warp = FlowWarp()

        self.training = training

    def forward(self, *args):
        if self.training:
            imgl, imgr, imglr = args[0], args[1], args[2]
        else:
            imglr = args[0]

        binocular_feature = self.binocular_encoder(imglr)
        flows = self.mutliscale_decoder(binocular_feature)

        if self.training:
            imgls = []
            imgrs = []
            scale = [1., 2., 4., 8.]
            for i, item in enumerate(scale):
                imgls.append(F.interpolate(imgl, scale_factor=1./item))
                imgrs.append(F.interpolate(imgr, scale_factor=1./item))
            warped_imgls = self.flow_warp(imgrs, flows)

            return warped_imgls, imgls, imgrs, flows
        else:
            return flows

    # def forward(self, imglr):
    #     binocular_feature = self.binocular_encoder(imglr)
    #
    #     # print('Multiscale decoder...')
    #     flows = self.mutliscale_decoder(binocular_feature)
    #
    #     return flows[0]

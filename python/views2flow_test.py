import torch.nn as nn
import torch.nn.functional as F
from models.modules import BinocularEncoder, MultiscaleDecoder, FlowWarp

class MultiscaleFlow(nn.Module):
    def __init__(self, binocualr_dim=None):
        super(MultiscaleFlow, self).__init__()
        self.binocular_encoder = BinocularEncoder(input_channels=binocualr_dim)
        self.mutliscale_decoder = MultiscaleDecoder()
        self.flow_warp = FlowWarp()

    def forward(self, imglr):
        binocular_feature = self.binocular_encoder(imglr)

        flows = self.mutliscale_decoder(binocular_feature)

        return flows
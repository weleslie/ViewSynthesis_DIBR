import torch
from torch import nn
from models.modules import FlowWarp

class SL1Loss(nn.Module):
    def __init__(self, ohem=False, topk=0.6):
        super(SL1Loss, self).__init__()
        self.ohem = ohem
        self.topk = topk
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), num_hard_samples)
        return torch.mean(loss)

class MultiscaleLoss(nn.Module):
    def __init__(self, multiscale_weights=None):
        super(MultiscaleLoss, self).__init__()
        self.weights = multiscale_weights
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, imgls, warped_imgls):
        multiscale_loss = 0.
        for i in range(len(imgls)):
            # loss = self.loss(imgls[i], warped_imgls[i])
            # loss_w = loss * self.weights[i]
            # multiscale_loss += loss_w
            multiscale_loss += torch.mean(self.loss(imgls[i] / 255., warped_imgls[i] / 255.)) * self.weights[i]
        return multiscale_loss

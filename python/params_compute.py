# ------------------------------------------
# Xiao Guo: 21/4/2022
# Compute #Params and FLOPs
# ------------------------------------------

from __future__ import print_function, division
import torch
from models.views2flow import MultiscaleFlow
import time


if __name__ == '__main__':
    net = MultiscaleFlow(binocualr_dim=6, training=False).cuda()
    from thop import profile

    input1 = torch.randn(10, 6, 1024, 512).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input1,))

    s = 0
    for i in range(100):
        t1 = time.time()
        _, _ = profile(net, inputs=(input1,))
        t2 = time.time()

        s += t2 - t1
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
    print('   time: %.4f' % (s / 100.0))

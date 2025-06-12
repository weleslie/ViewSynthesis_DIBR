from __future__ import print_function, division

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from data import StereoDataset
from models.views2flow import MultiscaleFlow
from models.loss import MultiscaleLoss

import time
import cv2
import numpy as np
import torch.nn.functional as F
import os

data_path = 'D:/training_model1_warp_cc'
model_path = './Log_L1/model-v2f-model-1.pth'
save_path = './results_1_series'
if not os.path.exists(save_path):
    os.mkdir(save_path)

weights = [1., .8, .6, .4, .2, .1]

alley_data = StereoDataset(data_path, (1024, 512))
dataloader = DataLoader(alley_data, batch_size=1, shuffle=False, num_workers=0)

model = MultiscaleFlow(binocualr_dim=6, training=False).cuda()
model.eval()
model.load_state_dict(torch.load(model_path))
multiscale_loss = MultiscaleLoss(multiscale_weights=weights)

example1 = torch.rand(1, 6, 512, 512).cuda()  # 生成一个随机输入维度的输入
traced_script_module = torch.jit.trace(model, example1)
traced_script_module.save("model_cpp_4.pt")

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))


CMAP = 'plasma'


def normalize_depth_for_display(depth, cmap=CMAP,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 ):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.
    # disp = 1.0 / (depth + 1e-6)
    disp = depth
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (np.percentile(disp, pc) + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    return disp


def gray2rgb(im, cmap=CMAP):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def backward_warp(imgr, flow):
    batch_size, dim, height, width = flow.size()  # flow size: Bx2xHxW

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(imgr).cuda()
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(imgr).cuda()

    # Apply shift in X direction
    x_shifts = flow[:, 0, :, :] / width  # Normalize the U dimension
    y_shifts = flow[:, 1, :, :] / height  # Normalize the V dimension

    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)  # [B, H, W, 2]
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(imgr, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output


if __name__ == '__main__':
    count = 0
    with torch.no_grad():
        for i_batch, stereo_batched in enumerate(dataloader):
            imgl_batch, imgr_batch = stereo_batched['imgl'], stereo_batched['imgr']
            imglr_batch = torch.cat([imgl_batch, imgr_batch], dim=1)

            tt0 = time.time()
            flows = model(imglr_batch.cuda())
            flow = flows[0]
            tt1 = time.time()
            print('cost time: ', tt1-tt0)

            # # warped_imgl_np = warped_imgl.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            imgl_np = imgl_batch.permute(0, 2, 3, 1).squeeze(0).numpy()
            imgr_np = imgr_batch.permute(0, 2, 3, 1).squeeze(0).numpy()
            flow_np = flow.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

            novel_view_1 = backward_warp(imgr_batch.cuda(), -flow * 0.16)
            novel_view_2 = backward_warp(imgr_batch.cuda(), -flow * 0.32)
            novel_view_3 = backward_warp(imgr_batch.cuda(), -flow * 0.48)
            novel_view_4 = backward_warp(imgr_batch.cuda(), -flow * 0.64)
            novel_view_5 = backward_warp(imgr_batch.cuda(), -flow * 0.8)
            novel_view_6 = backward_warp(imgr_batch.cuda(), -flow)

            novel_view_1 = novel_view_1.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            novel_view_2 = novel_view_2.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            novel_view_3 = novel_view_3.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            novel_view_4 = novel_view_4.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            novel_view_5 = novel_view_5.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            novel_view_6 = novel_view_6.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

            # plt.figure(), plt.imshow(-flow_np[:, :, 0], cmap='plasma')
            # plt.figure(), plt.imshow(flow_np[:, :, 1])
            # plt.figure(), plt.imshow(novel_view_5 / 255.)
            # plt.show()

            view_0 = cv2.cvtColor(imgl_np, cv2.COLOR_BGR2RGB) / 255.
            novel_view_1 = cv2.cvtColor(novel_view_1, cv2.COLOR_BGR2RGB) / 255.
            novel_view_2 = cv2.cvtColor(novel_view_2, cv2.COLOR_BGR2RGB) / 255.
            novel_view_3 = cv2.cvtColor(novel_view_3, cv2.COLOR_BGR2RGB) / 255.
            novel_view_4 = cv2.cvtColor(novel_view_4, cv2.COLOR_BGR2RGB) / 255.
            novel_view_5 = cv2.cvtColor(novel_view_5, cv2.COLOR_BGR2RGB) / 255.
            novel_view_6 = cv2.cvtColor(novel_view_6, cv2.COLOR_BGR2RGB) / 255.
            view_1 = cv2.cvtColor(imgr_np, cv2.COLOR_BGR2RGB) / 255.

            ## model 1 & 2
            view_0 = cv2.resize(view_0, (1152, 648))
            novel_view_1 = cv2.resize(novel_view_1, (1152, 648))
            novel_view_2 = cv2.resize(novel_view_2, (1152, 648))
            novel_view_3 = cv2.resize(novel_view_3, (1152, 648))
            novel_view_4 = cv2.resize(novel_view_4, (1152, 648))
            novel_view_5 = cv2.resize(novel_view_5, (1152, 648))
            novel_view_6 = cv2.resize(novel_view_6, (1152, 648))
            view_1 = cv2.resize(view_1, (1152, 648))

            # ## model 3
            # view_0 = cv2.resize(view_0, (1280, 960))
            # novel_view_1 = cv2.resize(novel_view_1, (1280, 960))
            # novel_view_2 = cv2.resize(novel_view_2, (1280, 960))
            # novel_view_3 = cv2.resize(novel_view_3, (1280, 960))
            # novel_view_4 = cv2.resize(novel_view_4, (1280, 960))
            # novel_view_5 = cv2.resize(novel_view_5, (1280, 960))
            # novel_view_6 = cv2.resize(novel_view_6, (1280, 960))
            # view_1 = cv2.resize(view_1, (1280, 960))

            if count == 0:
                cv2.imwrite(save_path + '/img' + str(count) + '.png', view_0 * 255)
            cv2.imwrite(save_path + '/img' + str(count + 1) + '.png', novel_view_5 * 255)
            cv2.imwrite(save_path + '/img' + str(count + 2) + '.png', novel_view_4 * 255)
            cv2.imwrite(save_path + '/img' + str(count + 3) + '.png', novel_view_3 * 255)
            cv2.imwrite(save_path + '/img' + str(count + 4) + '.png', novel_view_2 * 255)
            cv2.imwrite(save_path + '/img' + str(count + 5) + '.png', novel_view_1 * 255)
            cv2.imwrite(save_path + '/img' + str(count + 6) + '.png', view_1 * 255)
            count += 6

            cv2.imwrite('results_1/img' + str(i_batch) + '.png', novel_view_6 * 255)

    print('Finished Testing!')


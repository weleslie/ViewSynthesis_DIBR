from __future__ import print_function, division

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from data import StereoDataset
from models.views2flow import MultiscaleFlow
from models.loss import MultiscaleLoss

import time
import cv2

# imgl = cv2.imread('D:/training/6/24.bmp')
# imgr = cv2.imread('D:/training/6/25.bmp')
# imgl = cv2.resize(imgl, (512, 256))
# imgr = cv2.resize(imgr, (512, 256))
#
# imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
# imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
#
# imgl_ycrcb = cv2.cvtColor(imgl, cv2.COLOR_RGB2YCrCb)
# imgr_ycrcb = cv2.cvtColor(imgr, cv2.COLOR_RGB2YCrCb)
# kernel = np.array((
#             [1, 2, 1],
#             [0, 0, 0],
#             [-1, -2, -1]), dtype="float32")
# imgl_y_x = cv2.Sobel(imgl_ycrcb[:, :, 0], -1, 1, 1)
# imgl_y_y = cv2.Sobel(imgl_ycrcb[:, :, 0], -1, 0, 1)
# imgl_y = cv2.filter2D(imgl_ycrcb[:, :, 0], -1, kernel)
# imgr_y = cv2.filter2D(imgr_ycrcb[:, :, 0], -1, kernel)
# _, imgl_yx = cv2.threshold(imgl_y_x, 125, 255, cv2.THRESH_BINARY)
# _, imgl_yy = cv2.threshold(imgl_y_y, 125, 255, cv2.THRESH_BINARY)
#
# cv2.namedWindow('win')
# cv2.namedWindow('win_')
# cv2.namedWindow('win1')
# cv2.namedWindow('win2')
# cv2.imshow('win', imgl_y)
# cv2.imshow('win_', imgr_y)
# cv2.imshow('win1', imgl_yx)
# cv2.imshow('win2', imgl_yy)
# cv2.waitKey(0)


data_path = 'D:/training_model1_warp_cc'
model_path = './Log_L1'

weights = [1, .8, .6, .4, .2, .1]

alley_data = StereoDataset(data_path, (1024, 512))
dataloader = DataLoader(alley_data, batch_size=4, shuffle=False, num_workers=1)

model = MultiscaleFlow(binocualr_dim=6).cuda()
model.train()

multiscale_loss = MultiscaleLoss(multiscale_weights=weights)

optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25,
                                            gamma=0.5)

if __name__ == '__main__':
    for epoch in range(21):
        start = time.time()
        running_loss = 0.
        counter = 0

        for i_batch, stereo_batched in enumerate(dataloader):
            imgl_batch, imgr_batch = stereo_batched['imgl'], stereo_batched['imgr']
            imglr_batch = torch.cat([imgl_batch, imgr_batch], dim=1)

            warped_imgls, imgls, _, _ = model(imgl_batch.cuda(), imgr_batch.cuda(), imglr_batch.cuda())
            loss = multiscale_loss(imgls, warped_imgls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            counter += 1

        scheduler.step()

        print('Epoch: %d,  loss: %f, time: %.3f' % (epoch, running_loss / counter, time.time() - start))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(model.state_dict(), model_path + '/model-v2f-' + str(epoch) + '.pth')

    print('Finished Training!')


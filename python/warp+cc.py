# ------------------------------------------
# Xiao Guo: 21/4/2022
# Warp is basic function, CC is optional for comparison experiment,
# trans_ccm_interval_2.mat is opencv CCM,
# trans_r20_interval_2.mat is our method.
# ------------------------------------------


import numpy as np
import cv2 as cv
import os
from scipy.io import loadmat


def main():
    img_file = 'D:/training_model1/'
    ## only for warps
    img_warp_file = 'D:/training_model1_warp_2'
    ## warp+cc
    # img_warp_file = 'D:/training_model3_warp_cc'
    img_path = os.listdir(img_file)

    data = loadmat('corners.mat', verify_compressed_data_integrity=False)
    corners = np.array(data['corners'])

    target_corners = np.mean(corners[10:31:2, :, :], axis=0)

    ## -------------CCM----------------
    # data = loadmat('trans_ccm_interval_2.mat', verify_compressed_data_integrity=False)
    data = loadmat('trans_r20_interval_2.mat', verify_compressed_data_integrity=False)

    trans = data['mat']
    counter = 0

    for i in range(10, 31, 2):
        WarpMat = cv.getPerspectiveTransform(np.float32(corners[i]), np.float32(target_corners))

        for j in img_path:
            n = img_file + j + '/' + str('%2d' % (i + 1)) + '.jpg'
            train = cv.imread(n)

            ## -------------CCM----------------
            # if i != 20:
            #     mat = trans[counter, :, :]
            #     ## ------------ours------------
            #     h, w, _ = train.shape
            #     img = train.reshape(h * w, 3)
            #     img = img.transpose(1, 0)
            #
            #     img_trans = np.matmul(mat, img)
            #
            #     img_trans = img_trans.transpose(1, 0)
            #     train = img_trans.reshape(h, w, 3)
            #     ## -----------------------------
            #
            #     ## ------------ccm--------------
            #     # img = train.astype(np.float32) / 255.
            #     # img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     #
            #     # h, w, c = img.shape
            #     # img_ = img_.reshape(h * w, c)
            #     #
            #     # img_ccm = np.matmul(img_, mat)
            #     # out = img_ccm.reshape(h, w, c) * 255.
            #     # out[out < 0] = 0
            #     # out[out > 255] = 255
            #     # out = out.astype(np.uint8)
            #     # train = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            ## -------------CCM end------------

            temp_warp_img = cv.warpPerspective(train, WarpMat, (train.shape[1], train.shape[0]))

            if not os.path.exists(img_warp_file + '/' + j):
                os.mkdir(img_warp_file + '/' + j)

            cv.imwrite(img_warp_file + '/' + j + '/' + str('%.2d' % (int(i) + 1)) + '.jpg',
                       temp_warp_img)

        ## -------------CCM----------------
        if i != 20:
            counter += 1
        ## -------------CCM end------------


if __name__ == "__main__":
    main()

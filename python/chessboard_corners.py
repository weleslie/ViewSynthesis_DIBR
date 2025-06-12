# ------------------------------------------
# Xiao Guo: 21/4/2022
# Choose corner points from chessboard with click operation,
# and save them as a corner.mat file
# ------------------------------------------

import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat


def clickMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        temp.append([x, y])
        print([x, y])


def main():
    global corners, target_corners, temp_corners, temp
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", clickMouse)

    w1 = 4
    h1 = 4

    img1 = 'D:/training_model_chessboard/'
    corners = []
    target_corners = np.zeros((4, 2), dtype=np.float32)

    # 导入资源
    for i in range(40):
        temp = []

        img = img1 + str(i) + '.png'
        image = cv.imread(img)

        while True:
            cv2.imshow("img", image)
            if cv2.waitKey() == ord('q'):
                target_corners[0:1, :] += temp[0]
                target_corners[1:2, :] += temp[1]
                target_corners[2:3, :] += temp[2]
                target_corners[3:, :] += temp[3]

                corners.append(temp)
                print(i)
                break

        # cv2.drawChessboardCorners(image, (w1, h1), cp_img, ret)
        # cv2.imshow("img", image)
        # cv2.waitKey(0)

    corners = np.array(corners)
    savemat('../corners.mat', {'corners': corners})
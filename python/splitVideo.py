import cv2
import os

path = 'E:/date 7.19/test4(00h00m15s-00h00m30s).mp4'
v = cv2.VideoCapture(path)

H = 10
W = 4

rval, frame = v.read()
dir_idx = 0

while rval:
    rval, frame = v.read()

    if rval:
        path = 'D:/training_model1_1/'+str(dir_idx)
        if not os.path.exists(path):
            os.mkdir(path)
        idx = 1
        m, n, _ = frame.shape

        h = m // H
        w = n // W

        for i in range(H):
            for j in range(W):
                img = frame[i*h:(i+1)*h, j*w:(j+1)*w, :]
                cv2.imwrite(path + '/%.2d' % idx + '.jpg', img)

                idx += 1

        dir_idx += 1

print(1)
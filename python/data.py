from __future__ import print_function, division

import numpy as np
import os
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch
import PIL.Image as Image
import cv2


class StereoDataset(Dataset):
    def __init__(self, root_dir=None, output_size=(256, 512)):
        self.root_dir = root_dir
        self.output_size = output_size
        self.transform = transforms.Compose([Rescale(output_size), ToTensor()])
        self.session = os.listdir(self.root_dir)

        ## train
        self.session = self.session[::4]
        ## test model1 & model2 & 3
        # self.session = self.session[:1]

        self.image_dir = []
        for img_path in self.session:
            n_img = os.listdir(os.path.join(self.root_dir, img_path))
            for i in range(0, (len(n_img) - 1)):
                img_l = os.path.join(self.root_dir, img_path, n_img[i])
                img_r = os.path.join(self.root_dir, img_path, n_img[i+1])
                img = {'img_l': img_l, 'img_r': img_r}
                self.image_dir.append(img)

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        img = self.image_dir[idx]
        imgl_dir = img['img_l']
        imgr_dir = img['img_r']

        imgl = cv2.imread(imgl_dir)
        imgr = cv2.imread(imgr_dir)
        imgl = cv2.resize(imgl, self.output_size)
        imgr = cv2.resize(imgr, self.output_size)

        imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

        imgl = torch.from_numpy(imgl).permute(2, 0, 1).float()
        imgr = torch.from_numpy(imgr).permute(2, 0, 1).float()

        stereo = {'imgl': imgl, 'imgr': imgr}

        return stereo


class Rescale(object):
    def __init__(self, output_size):
        self.transform = torchvision.transforms.Resize(output_size)

    def __call__(self, stereo):
        imgl = self.transform(stereo['imgl'])
        imgr = self.transform(stereo['imgr'])
        return {'imgl': imgl, 'imgr': imgr}


class Rescale_back(object):
    def __init__(self, output_size):
        self.transform = torchvision.transforms.Resize(output_size)

    def __call__(self, stereo):
        imgl = self.transform(stereo['backl'])
        imgr = self.transform(stereo['backr'])
        return {'backl': imgl, 'backr': imgr}


class RandomCrop(object):
    """
    torchvision.transforms.RandomCrop just handle with one image, we need to
    crop two images at same area, then rewrite the function.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, stereo):
        imgl = np.array(stereo['imgl'])
        imgr = np.array(stereo['imgr'])
        imglr = np.concatenate([imgl, imgr], axis=-1)
        h, w, c = imglr.shape
        new_h, new_w = self.output_size
        if new_h == h and new_w == new_w:
            index = c // 2
            imgll = Image.fromarray(imglr[:, :, :index])
            imgrr = Image.fromarray(imglr[:, :, index:])
            return {'imgl': imgll, 'imgr': imgrr}
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            imglr = imglr[top: top + new_h, left: left + new_w]
            index = c // 2
            imgll = Image.fromarray(imglr[:, :, :index])
            imgrr = Image.fromarray(imglr[:, :, index:])
            return {'imgl': imgll, 'imgr': imgrr}


class GrayScale(object):
    def __init__(self):
        self.transform = torchvision.transforms.Grayscale()
    def __call__(self, stereo):
        imgl_gray = self.transform(stereo['imgl'])
        imgr_gray = self.transform(stereo['imgr'])
        return {'imgl': imgl_gray, 'imgr': imgr_gray}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, stereo):
        imgl = self.transform(stereo['imgl'])
        imgr = self.transform(stereo['imgr'])
        return {'imgl': imgl, 'imgr': imgr}


class ToTensorBack(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, stereo):
        imgl = self.transform(stereo['backl'])
        imgr = self.transform(stereo['backr'])
        return {'backl': imgl, 'backr': imgr}

#
# if __name__ == '__main__':
#     StereoDataset('I:/datasets/光流估计文件/MPI-Sintel-training_images/training/albedo/')

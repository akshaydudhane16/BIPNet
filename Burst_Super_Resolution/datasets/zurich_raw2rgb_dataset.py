import torch
import os
import cv2
import random
import numpy as np



class ZurichRAW2RGB(torch.utils.data.Dataset):
    """ Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """
    def __init__(self, root, split='train'):
        super().__init__()

        if split in ['train', 'test']:
            self.img_pth = os.path.join(root, split, 'canon')
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list(split)
        self.split = split

    def _get_image_list(self, split):
        if split == 'train':
            image_list = ['{:d}.jpg'.format(i) for i in range(46839)]#46839
        elif split == 'test':
            image_list = ['{:d}.jpg'.format(i) for i in range(1204)]
        else:
            raise Exception

        return image_list

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = cv2.imread(path)
        if random.randint(0,1) == 1 and self.split=='train':
            flag_aug = random.randint(1,7)
            img = self.data_augmentation(img, flag_aug)
        else:
            img = img
        return img

    def get_image(self, im_id):
        frame = self._get_image(im_id)

        return frame

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        frame = self._get_image(index)

        return frame

    def data_augmentation(self, image, mode):
        """
        Performs data augmentation of the input image
        Input:
            image: a cv2 (OpenCV) image
            mode: int. Choice of transformation to apply to the image
                    0 - no transformation
                    1 - flip up and down
                    2 - rotate counterwise 90 degree
                    3 - rotate 90 degree and flip up and down
                    4 - rotate 180 degree
                    5 - rotate 180 degree and flip
                    6 - rotate 270 degree
                    7 - rotate 270 degree and flip
        """
        if mode == 0:
            # original
            out = image
        elif mode == 1:
            # flip up and down
            out = np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            out = np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            # rotate 180 degree
            out = np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            # rotate 270 degree
            out = np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out.copy()

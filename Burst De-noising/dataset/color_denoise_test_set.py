# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np


class ColorDenoiseTestSet(torch.utils.data.Dataset):
    """
    Test set for the color burst denoising dataset introduced in [1]. The dataset can be downloaded from
    https://drive.google.com/file/d/1rXmauXa_AW8ZrNiD2QPrbmxcIOfsiONE/view

    [1] Basis Prediction Networks for Effective Burst Denoising with Large Kernels. Zhihao Xia, Federico Perazzi,
    Michael Gharbi, Kalyan Sunkavalli, and Ayan Chakrabarti, CVPR 2020
    """
    def __init__(self, root=None, noise_level=1, initialize=True):
        """
        args:
            root - Path to root dataset directory
            noise_level - The noise_level to use. Can be [1, 2, 4, 8], which higher level means more noise.
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        self.root = root

        self.noise_level = noise_level

        if initialize:
            self.initialize()

    def initialize(self):
        root = self.root

        data = np.load('{}/{}.npz'.format(root, self.noise_level))

        self.gt = np.ascontiguousarray(np.transpose(data['truth'], (0, 3, 1, 2)))
        self.burst = np.ascontiguousarray(np.transpose(data['noisy'], (0, 3, 4, 1, 2)))
        self.white_level = data['white_level']
        self.sig_shot = data['sqrt_sig_shot']
        self.sig_read = data['sig_read']

        # self.noise_levels = {0: 0.0114296, 1: 0.03428832, 2: 0.0520008, 3: 0.078863139, 4: 0.11960187, 5: 0.1813852}

    def __len__(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return self.gt.shape[0]

    def get_burst_info(self, burst_id):
        burst_name = '{}_{:02d}'.format(self.noise_level, burst_id)
        burst_info = {'burst_size': 8, 'burst_name': burst_name,
                      'sig_shot': torch.tensor(self.sig_shot[burst_id]).float(),
                      'sig_read': torch.tensor(self.sig_read[burst_id]).float(),
                      'white_level': torch.tensor(self.white_level[burst_id]).float(),
                      'gamma': torch.tensor([2.4, ]).float()}
        return burst_info

    def get_burst(self, burst_id, im_ids, info=None):
        # assert len(im_ids) == 8 and im_ids[0] == 0
        frames = [self.burst[burst_id, i] for i in im_ids]

        gt = self.gt[burst_id]

        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info

    def __getitem__(self, index):
        burst, gt, info = self.get_burst(index, list(range(8)))

        burst = np.stack(burst)
        burst = torch.from_numpy(burst).float()

        gt = torch.from_numpy(gt).float()

        sigma_read_est = info['sig_read'].view(1, 1).expand_as(gt)
        sigma_shot_est = info['sig_shot'].view(1, 1).expand_as(gt) ** 2.0

        sigma_estimate = torch.sqrt(sigma_read_est ** 2 + sigma_shot_est * burst.clamp(0.0))

        info['sigma_estimate'] = sigma_estimate

        return burst, gt, info

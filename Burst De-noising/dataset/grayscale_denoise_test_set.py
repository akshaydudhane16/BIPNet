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
import math
import numpy as np


class GrayscaleDenoiseTestSet(torch.utils.data.Dataset):
    """
    Test set for the grayscale burst denoising dataset introduced in [1]. The dataset can be downloaded from
    https://drive.google.com/file/d/1UptBXV4f56wMDpS365ydhZkej6ABTFq1/view

    [1] Burst Denoising with Kernel Prediction Networks. Ben Mildenhall, Jonathan T. Barron,
    Jiawen Chen, Dillon Sharlet, Ren Ng, and Robert Carroll, CVPR 2018
    """
    def __init__(self, root=None, noise_level=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            noise_level - The noise_level to use. Can be [1, 2, 4, 8], which higher level means more noise.
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        self.root = root

        # Convert noise level
        noise_level = 2 + math.log2(noise_level)
        self.noise_level = int(noise_level)

        if initialize:
            self.initialize()

    def initialize(self):
        noise_level = self.noise_level
        root = self.root

        data = np.load(root)

        self.gt = data['truth']
        self.burst = data['noisy']
        self.white_level = data['white_level']
        self.sig_shot = data['sig_shot']
        self.sig_read = data['sig_read']

        if noise_level is not None:
            self.gt = self.gt[noise_level*73: (noise_level+1)*73]
            self.burst = self.burst[noise_level * 73: (noise_level + 1) * 73]
            self.white_level = self.white_level[noise_level * 73: (noise_level + 1) * 73]
            self.sig_shot = self.sig_shot[noise_level * 73: (noise_level + 1) * 73]
            self.sig_read = self.sig_read[noise_level * 73: (noise_level + 1) * 73]

    def __len__(self):
        return self.gt.shape[0]

    def get_burst_info(self, burst_id):
        if self.noise_level is None:
            noise_level = burst_id // 73
            burst_name = '{}_{:02d}'.format(noise_level, burst_id % 73)
        else:
            noise_level = self.noise_level
            burst_name = '{}_{:02d}'.format(noise_level, burst_id)

        burst_info = {'burst_length': 8, 'burst_name': burst_name, 'noise_level': noise_level,
                      'sig_shot': torch.tensor(self.sig_shot[burst_id]).float(),
                      'sig_read': torch.tensor(self.sig_read[burst_id]).float(),
                      'white_level': torch.tensor(self.white_level[burst_id]).float(),
                      'gamma': torch.tensor([2.4, ]).float()}
        return burst_info

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self.burst[burst_id, :, :, i] for i in im_ids]

        gt = self.gt[burst_id]

        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info

    def __getitem__(self, index):
        burst, gt, info = self.get_burst(index, list(range(8)))

        burst = np.stack(burst)
        burst = torch.from_numpy(burst).unsqueeze(1).float()

        gt = torch.from_numpy(gt).unsqueeze(0).float()

        sigma_read_est = info['sig_read'].view(1, 1).expand_as(gt)
        sigma_shot_est = info['sig_shot'].view(1, 1).expand_as(gt) ** 2.0

        sigma_estimate = torch.sqrt(sigma_read_est ** 2 + sigma_shot_est * burst.clamp(0.0))

        info['sigma_estimate'] = sigma_estimate

        return burst, gt, info

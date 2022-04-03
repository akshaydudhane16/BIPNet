## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680

import os
import cv2
import torch
import argparse
import numpy as np
from pytorch_lightning import seed_everything
seed_everything(13)

######################################## Model and Dataset ########################################################
from Network import BIPNet
from datasets.synthetic_burst_val_set import SyntheticBurstVal

##################################################################################################################

parser = argparse.ArgumentParser(description='Synthetic burst super-resolution using BIPNet')

parser.add_argument('--input_dir', default='./syn_burst_val', type=str, help='Directory of NTIRE21 BurstSR validation images')
parser.add_argument('--result_dir', default='./Results/Synthetic/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Trained_models/Synthetic/BIPNet.ckpt', type=str, help='Path to weights')

args = parser.parse_args()

######################################### Load BIPNet ####################################################

model = BIPNet()
model = BIPNet.load_from_checkpoint(args.weights)
model.cuda()
model.summarize()

######################################### NTIRE21 BurstSR Validation ####################################################

dataset = SyntheticBurstVal(args.input_dir)

result_dir = args.result_dir + 'Developement Phase'
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True) 


for idx in range(len(dataset)):

    burst, burst_name = dataset[idx]            

    print("Processing Burst:::: ", burst_name)

    burst = burst.cuda()
    burst = burst.unsqueeze(0)
    with torch.no_grad():
        net_pred = model(burst)

    # Normalize to 0  2^14 range and convert to numpy array
    net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

    # Save predictions as png
    cv2.imwrite('{}/{}.png'.format(result_dir, burst_name), net_pred_np)

for idx in range(len(dataset)):

    burst, burst_name = dataset[idx]            

    print("Processing Burst:::: ", burst_name)

    burst = burst.cuda()
    burst = burst.unsqueeze(0)
    with torch.no_grad():
        net_pred = model(burst)

    # Normalize to 0  2^14 range and convert to numpy array
    net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

    # Save predictions as png
    cv2.imwrite('{}/{}.png'.format(result_dir, burst_name), net_pred_np)

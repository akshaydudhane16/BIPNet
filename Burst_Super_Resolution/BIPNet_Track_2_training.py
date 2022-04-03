## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680



######################################## Pytorch lightning ########################################################

import os
import time
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision import transforms as transforms

######################################## Model and Dataset ########################################################

from Network import BIPNet
from datasets.burstsr_dataset import BurstSRDataset

from utils.data_format_utils import torch_to_numpy, numpy_to_torch
from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import BurstSRPostProcess

from utils.metrics import AlignedL1, AlignedL1_loss, AlignedL2_loss, AlignedSSIM_loss, AlignedPSNR, AlignedSSIM, AlignedLPIPS, AlignedLPIPS_loss

from pwcnet.pwcnet import PWCNet
from utils.warp import warp

import data_processing.camera_pipeline as rgb2raw
from data_processing.camera_pipeline import *

from collections import OrderedDict

##################################################################################################################

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(13)

# In[5]:


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#####################################################################################

log_dir = './logs/Track_2/'

class Args:
    def __init__(self):
        self.image_dir = "./burstsr_dataset"
        self.model_dir = log_dir + "saved_model"
        self.result_dir = log_dir + "results"
        self.batch_size = 1
        self.num_epochs = 25
        self.lr = 1e-5
        self.burst_size = 14
        self.NUM_WORKERS = 6
        
        
args = Args()



######################################### Training #########################################################

class BurstSR_Network():

    def __init__(self, args):
        super().__init__()
        
        PWCNet_weight_PATH = './pwcnet/pwcnet-network-default.pth'        
        alignment_net = PWCNet(load_pretrained=True, weights_path=PWCNet_weight_PATH)
        alignment_net = alignment_net.cuda()
        
        self.aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)
        self.aligned_L1_loss = AlignedL1(alignment_net=alignment_net)
        self.AlignedLPIPS_loss = AlignedLPIPS_loss(alignment_net=alignment_net)
        
    def train(self):               
        
        ################# Load BIPNet ########################   

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir, exist_ok=True)

        if os.path.exists(args.model_dir + './BIPNet.ckpt'):
            print("Loading pre-trained BIPNet on synthetic burst dataset")
            model = BIPNet().load_from_checkpoint(args.model_dir + './BIPNet.ckpt').cuda()            
        else:
            print("Training from scratch:")
            model = BIPNet().cuda()
        
        model.summarize()       

        optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-6)
                    
        ################## DATA Loaders ########################################
            
        train_dataset = BurstSRDataset(root=args.image_dir,  split='train', burst_size=14, crop_sz=24, random_flip=True)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        test_dataset = BurstSRDataset(root=args.image_dir,  split='val', burst_size=14, crop_sz=80, random_flip=False)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        ########################################################################
        
        best_PSNR = 0
        best_epoch = 0
        best_iter = 0       
        
        epochs = args.num_epochs
        
        ################# Model Training ####################################
        
        for epoch in range(epochs):
            
            epoch_start_time = time.time()
            epoch_losses = AverageMeter()
            epoch_loss = 0
            
            with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as _tqdm:
                
                _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epochs))
                
                for i, data in enumerate(train_data_loader):

                    optimizer.zero_grad()
                    burst, labels, meta_info_burst, meta_info_gt, burst_name = data

                    burst = burst.cuda()
                    labels = labels.cuda()
                    
                    preds = model(burst)
                    preds = preds.clamp(0.0, 1.0)
                    
                    loss = self.aligned_L1_loss(preds, labels, burst) + 0.0095*self.AlignedLPIPS_loss(preds, labels, burst)
                    epoch_loss += loss.item()
                    
                    epoch_losses.update(loss.item(), len(burst))
                    _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    _tqdm.update(len(burst))                    
                    
                    if i%(len(train_data_loader)//4)==0 and i>0:
                        PSNR = self.validation(model, test_data_loader)
                        if PSNR > best_PSNR:
                            best_PSNR = PSNR
                            best_epoch = epoch
                            best_iter = i
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict()},
                                       os.path.join(args.model_dir, "model_best.pth"))

                        print("[epoch %d it %d PSNR: %.6f --- best_epoch %d best_iter %d Best_PSNR %.6f]" % (epoch, i, PSNR, best_epoch, best_iter, best_PSNR))                    
                    
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
                    optimizer.step()
                    
                scheduler.step() 
            
                           
            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time, epoch_loss/len(train_dataset)))
            print("------------------------------------------------------------------")                        
            
            
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(args.model_dir, f"model_epoch_{epoch}.pth"))
            
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(args.model_dir, "model_latest.pth"))
            
    
    
    #################### Validation #################################    
    def validation(self, model, test_data_loader):
        PSNR = []
        
        for i, data in enumerate(test_data_loader):
            burst, labels, flow_vectors, meta_info, burst_name = data
            burst = burst.cuda()
            labels = labels.cuda()
            
            with torch.no_grad():
                output = model(burst)
                output = output.clamp(0.0, 1.0)            
            
            PSNR.append(self.aligned_psnr_fn(output, labels, burst).cpu().numpy())
            
        mean_psnr = sum(PSNR) / len(PSNR)        
        return mean_psnr
    

BurstSR_Network(args).train()

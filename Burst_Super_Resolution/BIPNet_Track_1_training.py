## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680

import os

######################################## Pytorch lightning ########################################################

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
seed_everything(13)

######################################## Model and Dataset ########################################################

from Network import BIPNet
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader

##################################################################################################################

log_dir = './logs/Track_1/'

class Args:
    def __init__(self):
        self.image_dir = "./Zurich-RAW-to-DSLR-Dataset"
        self.model_dir = log_dir + "saved_model"
        self.result_dir = log_dir + "results"
        self.batch_size = 1
        self.num_epochs = 100
        self.lr = 1e-4
        self.burst_size = 14
        self.NUM_WORKERS = 6
        
args = Args()

######################################### Data loader ######################################################

def load_data(image_dir, burst_size):

    train_zurich_raw2rgb = ZurichRAW2RGB(root=image_dir,  split='train')
    train_dataset = SyntheticBurst(train_zurich_raw2rgb, burst_size=burst_size, crop_sz=384)    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=args.NUM_WORKERS, pin_memory=True)

    test_zurich_raw2rgb = ZurichRAW2RGB(root=image_dir,  split='test')
    test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=burst_size, crop_sz=384)    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader

######################################### Load BIPNet ####################################################

model = BIPNet()
model.cuda()
model.summarize()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir, exist_ok=True) 

######################################### Training #######################################################

train_loader, test_loader = load_data(args.image_dir, args.burst_size)

checkpoint_callback = ModelCheckpoint(
    monitor='val_psnr',
    dirpath=args.model_dir,
    filename='{epoch:02d}-{val_psnr:.2f}',
    save_top_k=3,
    save_last=True,
    mode='max',
)

trainer = Trainer(gpus=-1,
                    auto_select_gpus=True,
                    accelerator='ddp',
                    max_epochs=300,
                    precision=16,
                    gradient_clip_val=0.01,
                    callbacks=[checkpoint_callback],
                    benchmark=True,
                    deterministic=True,
                    val_check_interval=0.25,
                    progress_bar_refresh_rate=100,
                    profiler="advanced")#,
                    #resume_from_checkpoint = args.model_dir + './BIPNet.ckpt')

trainer.fit(model, train_loader, test_loader)

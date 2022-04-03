## NTIRE 2021 Burst Super-Resolution Challenge - Track 1 Synthetic
### Training
- Download [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
```
python BIPNet_Track_1_training.py
```
### Developement Phase:
- Download [syn_burst_val](https://data.vision.ee.ethz.ch/bhatg/syn_burst_val.zip) and extract it in root directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EYlxq0X49fRGiFD3kMxnM6IB7VNtwhd3atNr4oc1b1psbA?e=pLN14I) and place it in './Trained_models/Synthetic/BIPNet.ckpt'.
        
```
python Track_1_evaluation.py
```
- Results: stored in './Results/Synthetic/Developement Phase'


## NTIRE 2021 Burst Super-Resolution Challenge - Track 2 Real-world
### Training
- Download [BurstSR train and validation set](https://github.com/goutamgmb/NTIRE21_BURSTSR/blob/master/burstsr_links.md).
- Download [Pretrained BIPNet on synthetic burst SR dataset](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EYlxq0X49fRGiFD3kMxnM6IB7VNtwhd3atNr4oc1b1psbA?e=rv8iFx) and place it in './logs/Track_2/saved_model/BIPNet.ckpt'.
```
python BIPNet_Track_2_training.py
```
### Developement Phase:
- Download [burstsr_dataset](https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/val.zip) and extract it in root directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EX4h9sC8zvtPkoHQkvTY8VABxF2C4agXqL9HENW1_7Td9Q?e=XIXchy) and place it in './Trained_models/Real/BIPNet.pth'.

```
python Track_2_evaluation.py
```
- Results: stored in './Results/Real/Developement Phase'

## Grayscale Burst De-noising

### Training
- Download Open Images dataset from [here](https://storage.googleapis.com/openimages/web/download.html).
```
python grayscale_denoising_training.py
```
### Testing:
- Download [Grayscale Burst Denoising test set](https://drive.google.com/file/d/1UptBXV4f56wMDpS365ydhZkej6ABTFq1/view) and put it in ./input directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/akshay_dudhane_mbzuai_ac_ae/EiwCbKwSLThGkrupaTFIj8EBOl47lNIsodjfJvv2hEtIeg?e=acHhkN) and place it in './Trained_models/grayscale_denoising/BIPNet.ckpt'.
        
```
python Grayscale_denoising_testing.py
```
- Results: stored in './Results/grayscale_denoised_output/'


## Color Burst De-noising

### Training
- Download Open Images dataset from [here](https://storage.googleapis.com/openimages/web/download.html).
```
python color_denoising_training.py
```
### Testing:
- Download [Color Burst Denoising test set](https://drive.google.com/file/d/1rXmauXa_AW8ZrNiD2QPrbmxcIOfsiONE/view) and unpack the zip file to './input/color_testset' directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EQo14XRVKslHkNVvd-yWRRMBRufDhOjsx3LB_uECWDcMnA?e=OpdVGQ) and place it in './Trained_models/color_denoising/BIPNet.ckpt'.
        
```
python Color_denoising_testing.py
```
- Results: stored in './Results/color_denoised_output/'

## Grayscale Burst De-noising

### Training
- Download Open Images dataset from [here](https://storage.googleapis.com/openimages/web/download.html).
```
python grayscale_denoising_training.py
```
### Testing:
- Download [Grayscale Burst Denoising test set](https://drive.google.com/file/d/1UptBXV4f56wMDpS365ydhZkej6ABTFq1/view) and put it in ./input directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/Eb_GplW22rNAkgVssAF_qEYBo_BDN5uBmvmfFjPeHNYSGg?e=t7XThQ) and place it in './Trained_models/grayscale_denoising/BIPNet.ckpt'.
        
```
python Grayscale_denoising_testing.py
```
- Results: stored in './Results/grayscale_denoised_output/'

**For fair comparison between proposed BIPNet and other existing methods for gray-scale denoising, cosider PSNR values of the proposed BIPNet "with post-processing".

|          Model         |  Gain 1 | Gain 2 | Gain 4 | Gain 8 |                                            Links                                            | Notes                    |
|:----------------------:|:-----:|:-------:|:-----:|:-----:|:-------------------------------------------------------------------------------------------:|--------------------------|
| BIPNet without post-processing | 41.26 | 38.74   | 35.91 | 31.35 |[model](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/akshay_dudhane_mbzuai_ac_ae/EiwCbKwSLThGkrupaTFIj8EBOl47lNIsodjfJvv2hEtIeg?e=acHhkN) | CVPR 2021 |
| BIPNet with post-processing (sRGB) | 38.53 | 35.94   | 33.074 | 29.89 |[model](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/akshay_dudhane_mbzuai_ac_ae/EiwCbKwSLThGkrupaTFIj8EBOl47lNIsodjfJvv2hEtIeg?e=acHhkN) | Official retrained model |

## Color Burst De-noising

### Training
- Download Open Images dataset from [here](https://storage.googleapis.com/openimages/web/download.html).
```
python color_denoising_training.py
```
### Testing:
- Download [Color Burst Denoising test set](https://drive.google.com/file/d/1rXmauXa_AW8ZrNiD2QPrbmxcIOfsiONE/view) and unpack the zip file to './input/color_testset' directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EcdoWz61ptpInH59dQNc7nsB6OcIMioZkHSDGUZBwgcpjw?e=1RJwIp) and place it in './Trained_models/color_denoising/BIPNet.ckpt'.
        
```
python Color_denoising_testing.py
```
- Results: stored in './Results/color_denoised_output/'


**For fair comparison between proposed BIPNet and other existing methods for color denoising, cosider PSNR values of the proposed BIPNet "with post-processing".

|          Model         |  Gain 1 | Gain 2 | Gain 4 | Gain 8 |                                            Links                                            | Notes                    |
|:----------------------:|:-----:|:-------:|:-----:|:-----:|:-------------------------------------------------------------------------------------------:|--------------------------|
| without post-processing | 42.28 | 40.20   | 37.85 | 34.64 |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EQo14XRVKslHkNVvd-yWRRMBRufDhOjsx3LB_uECWDcMnA?e=OpdVGQ) | CVPR 2021 |
| with post-processing | 40.58 | 38.13   | 35.30 | 32.87 |[model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EQo14XRVKslHkNVvd-yWRRMBRufDhOjsx3LB_uECWDcMnA?e=OpdVGQ) | Official retrained model |

# MSGNet-main
MSGNet: Multi-Source Guidance Network for Fish Segmentation in Underwater Videos  
The paper can be found here [Paper](https://www.frontiersin.org/articles/10.3389/fmars.2023.1256594/abstract)
## Usage
### 1. Requirements  
Pytorch >= 1.11.0 , Note that torchvision must >= **0.12**
### 2. Prepare the data
Downloading dataset and moving it into ./datasets/, which can be found as follow:  
✧ [DeepFish](https://alzayats.github.io/DeepFish/)  
✧ [Seagrass](https://doi.pangaea.de/10.1594/PANGAEA.926930)  
✧ [MoCA-Mask](https://xueliancheng.github.io/SLT-Net-project/)  

We also provide processed datasets, which can be found as follow:   
✧ [Google Drive](https://drive.google.com/file/d/1vcxuW0Erxhk2X5K9HdKrl3ap-GUMqPho/view?usp=sharing)  
✧ [百度网盘](https://pan.baidu.com/s/1pAOLFxF1OL3KJhDd8QkYiA)  ，提取码：k0jg
### 3. Flow extraction
First, we need to generate an optical flow for the video data.
  
cd `./flow_prepare/png2jpg_warp.py` to generate wrap images.  

Then, cd `./flow_prepare/final_RAFT.py` to extract optical flow.  

Refer to `./flow_prepare/example/7623` for more details.  
![image](https://github.com/zp19990818/MSGNet-main/assets/53686038/15edfa5e-c240-49f4-b0aa-38e714a72df0)

### 4. Train 
In order to speed up data reading, the data needs to be packaged in advance to generate a pkl file,   

First, run `./datasets/dataloader_list/adaptor_dataset_.py` to generate `train.pkl` and `val.pkl`  

Then, just run `train.py`.

### 5. Test 
Just run `test.py`， and enjoy it!  

Test results can be evaluated by running `./measures/measure_all.py`

Pre-training weights can be found here:  

✧ [Google Drive](https://drive.google.com/file/d/1nkKitUxrFdJjklX-7fYcLxp6Z5iKcmls/view?usp=sharing)  
✧ [百度网盘](https://pan.baidu.com/s/1FMFMTfBaFlCUceCVRlC-aw)  ，提取码：yaev  

### 6. Result
![image](https://github.com/zp19990818/MSGNet-main/assets/53686038/f7e59064-e5fe-4371-8b1e-4c505df54051)
 
## FAQ
The relevant citation code can be found at the following link:  
  [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)   
  [SUIM-Net](https://github.com/xahidbuffon/SUIM)  
  [WaterSNet](https://github.com/ruizhechen/WaterSNet)  
  [FSNet](https://github.com/GewelsJI/FSNet)  
  [AMC-Net](https://github.com/isyangshu/AMC-Net)
  
  

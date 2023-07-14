import torch
from torchvision import transforms

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.utils import make_dir

# from model.DQW_RA_TMO import VOS
from model.MSGNet import MSGNet

inputRes = (384, 384)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_transforms = transforms.Compose([to_tensor, normalize])
tensor_transforms = transforms.Compose([to_tensor])
model_name = 'MSGNet'  # specify the model name
snapshot = 'D:\\zhangpeng\\git_projects\\MSGNet-main\\ckpt\\official_DeepFish\\MSGNet_100epoch_emptyflow\\MSGNet\\best.pth'  # Replace with your own absolute path

result_dir = './result/MSGNet/empty_flow_official_DeepFish'  # Replace with your absolute path
make_dir(result_dir)
model = MSGNet()
model.load_state_dict(torch.load(snapshot))
torch.cuda.set_device(device=0)
model.cuda()

model.train(False)

val_set = "D:\\zhangpeng\\git_projects\\MSGNet-main\\datasets\\DeepFish_official\\ImageSets\\DeepFish_Seagrass\\test.txt"
with open(val_set) as f:
    seqs = f.readlines()
    seqs = [seq.strip() for seq in seqs]

for video in tqdm(seqs):
    frame_dir = "D:\\zhangpeng\\git_projects\\MSGNet-main\\datasets\\DeepFish_official\\JPEGImages\\1080p"
    flow_dir = "D:\\zhangpeng\\git_projects\\MSGNet-main\\datasets\\DeepFish_official\\Flow\\1080p"

    image_dir = os.path.join(frame_dir, video)
    flow_dir = os.path.join(flow_dir, video)

    imagefiles = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    flowfiles = sorted(glob.glob(os.path.join(flow_dir, '*.jpg')))
    image = Image.open(imagefiles[0]).convert('RGB')
    width, height = image.size
    count = 0
    mask_preds = np.zeros((len(imagefiles), height, width)) - 1
    with torch.no_grad():
        for imagefile, flowfile in zip(imagefiles, flowfiles):
            image = Image.open(imagefile).convert('RGB')
            flow = Image.open(flowfile).convert('RGB')
            width, height = image.size

            image = np.array(image.resize(inputRes))
            flow = np.array(flow.resize(inputRes))

            image = image_transforms(image)
            flow = image_transforms(flow)

            image = image.unsqueeze(0)
            flow = flow.unsqueeze(0)

            image, flow = image.cuda(), flow.cuda()
            mask_pred, mask_pred_4, mask_pred_3, mask_pred_2, mask_pred_1, mask_pred_0 = model(image, flow)
            # mask_pred = model(image, flow)

            mask_pred = mask_pred[0, 0, :, :]
            mask_pred[mask_pred >= 0.5] = 1
            mask_pred[mask_pred < 0.5] = 0

            mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')

            save_folder = '{}/{}/result/{}'.format(result_dir, model_name, video)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_file = os.path.join(save_folder,
                                     os.path.basename(imagefile)[:-4] + '.png')

            mask_pred = mask_pred.resize((width, height))

            mask_preds[count, :, :] = np.array(mask_pred)
            count = count + 1

            mask_pred.save(save_file)

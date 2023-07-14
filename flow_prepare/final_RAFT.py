# 调用torchvision库的RAFT网络，0.12版本以上支持，torchvision>=0.12.0
# 2022-11-03
import os
import cv2
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.utils import flow_to_image
import numpy as np
import torch
import torchvision.transforms as T


def preprocess(img1_path):
    img1 = cv2.imread(img1_path)
    # img2 = cv2.imread(img2_path)

    img1 = torch.tensor(img1)
    # img2 = torch.tensor(img2)

    img1_4D = torch.unsqueeze(img1, dim=0)
    # img_batch = torch.stack([img1, img2], dim=0)  # 2个3D Tensor拼接升维成一个4D Tensor
    img_batch = img1_4D.permute(0, 3, 1, 2)

    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    img_batch = transforms(img_batch)
    # print(f"shape = {img_batch.shape}, dtype = {img_batch.dtype}")
    return img_batch


def calculate(img1_path, img2_path):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(pretrained=True, progress=False).to(device)
    # model = raft_small(pretrained=True, progress=False).to(device)

    model = model.eval()

    list_of_flows = model(img1.to(device), img2.to(device))

    predicted_flows = list_of_flows[-1]

    flow_imgs = flow_to_image(predicted_flows)
    flow_imgs = flow_imgs.permute(0, 2, 3, 1)

    a = flow_imgs.squeeze(dim=0)
    b = a.cpu().numpy()

    return b


path = "../flow_prepare/example/"
list = os.listdir(path)

for i in range(len(list)):
    image_path = path + list[i] + "\\wrap\\"
    save_path = path + list[i] + "\\wrap_flow\\"
    list2 = os.listdir(image_path)

    if not os.path.exists(save_path):  # os模块判断并创建
        os.mkdir(save_path)
        print("make flow dir：" + save_path)
        for j in range(len(list2)):
            if j == 0:
                print("first image do not have a flow!")
            else:
                img1_path = image_path + list2[j-1]
                img2_path = image_path + list2[j]

                print(img1_path)
                # flow_img = calculate()
                test_img = cv2.imread(img1_path)
                rows = test_img.shape[0] # 1080
                cols = test_img.shape[1] # 1920

                flow_img = calculate(img1_path, img2_path)
                final_flow = cv2.resize(flow_img, (cols, rows))

                if not os.path.exists(save_path):  # os模块判断并创建
                    os.mkdir(save_path)
                    cv2.imwrite(save_path + list2[j - 1], final_flow)
                else:
                    cv2.imwrite(save_path + list2[j-1], final_flow)

    else:
        print("flow dir already exist!")
        for j in range(len(list2)):
            if j == 0:
                print("first image do not have a flow!")
            else:
                img1_path = image_path + list2[j-1]
                img2_path = image_path + list2[j]

                print(img1_path)
                # flow_img = calculate()
                test_img = cv2.imread(img1_path)
                rows = test_img.shape[0] # 1080
                cols = test_img.shape[1] # 1920

                flow_img = calculate(img1_path, img2_path)
                final_flow = cv2.resize(flow_img, (cols, rows))

                if not os.path.exists(save_path):  # os模块判断并创建
                    os.mkdir(save_path)
                    cv2.imwrite(save_path + list2[j - 1], final_flow)
                else:
                    cv2.imwrite(save_path + list2[j-1], final_flow)

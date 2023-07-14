# -*- coding: utf-8 -*
import os
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
from datasets.dataloader_list.transform import Compose, JointResize, RandomHorizontallyFlip, RandomRotate
from torch.utils.data import DataLoader, Dataset
from datasets.dataloader_list.DeepFish import DeepFishDataset, DeepFishDatasetVal


class DeepFish_Dataset(Dataset):
    def __init__(
            self, split='train', datasets=["DeepFish_official"], subsets=["train"], inputRes=384):
        self.datasets = datasets
        self.inputRes = inputRes
        self.split = split
        self.modules = []
        self.data_item = []
        if self.split == 'train':
            if "DeepFish_official" in self.datasets:
                module = DeepFishDataset(subsets)
                self.modules.append(module)
            for i in range(len(self.modules)):
                self.data_item = self.data_item + self.modules[i].data_items
        else:
            module = DeepFishDatasetVal()
            self.data_item = self.data_item + module.data_items

        self.transform = Compose([JointResize(inputRes)])
        self.img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )
        self.flow_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )
        self.mask_transform = transforms.ToTensor()
        self.depth_transform = transforms.ToTensor()
        if split == 'train':
            self.transform = Compose([JointResize(inputRes), RandomHorizontallyFlip(), RandomRotate(10)])
            self.img_transform = transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )
            self.flow_transform = transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )

    def __getitem__(self, item):
        data_item = self.data_item[item]
        image_path = data_item['image'][0]
        mask_path = data_item['annos'][0]
        # depth_path = data_item['depth'][0]
        flow_path = data_item['flow'][0]
        image = Image.open(image_path).convert('RGB')
        flow = Image.open(flow_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

        image, mask, flow = self.transform(image, mask, flow)
        image = self.img_transform(image)
        flow = self.flow_transform(flow)
        mask = self.mask_transform(mask)

        return image, flow, mask

    def __len__(self):
        return len(self.data_item)
        

if __name__ == '__main__':
    data = DeepFish_Dataset(split='train')
    print(len(data))

    dataloader = DataLoader(data,  batch_size=1)
    for ii, (image, flow, mask) in enumerate(dataloader):
        print(flow.shape)
        print('---')
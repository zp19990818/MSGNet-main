# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets.dataloader_list.DeepFish import DeepFishDataset, DeepFishDatasetVal
import datasets.dataloader_list.custom_transforms_f as tr
from torchvision import transforms


class DeepFish_Dataset(Dataset):
    def __init__(
            self, split='train', datasets=["DeepFish_Seagrass"], subsets=["train"], inputRes=(384, 384), augment=False,
            transform=None, target_transform=None):
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform
        self.inputRes = inputRes
        self.split = split
        self.modules = []
        self.data_item = []
        if self.split == 'train':
            if "DeepFish_Seagrass" in self.datasets:
                module = DeepFishDataset(subsets)
                self.modules.append(module)

            for i in range(len(self.modules)):
                self.data_item = self.data_item + self.modules[i].data_items
        else:
            module = DeepFishDatasetVal()
            self.data_item = self.data_item + module.data_items

        self.augment_transform = None
        self.augment_transform_image = None
        if augment and split == 'train':
            self.augment_transform = transforms.Compose([
                tr.RandomHorizontalFlip(0.5),
                tr.ScaleNRotate(rots=(-10, 10),
                                scales=(.75, 1.25))])
            self.augment_transform_image = transforms.ColorJitter(0.1, 0.1, 0.1)

    def __getitem__(self, item):
        data_item = self.data_item[item]
        image_path = data_item['image'][0]
        mask_path = data_item['annos'][0]
        flow_path = data_item['flow'][0]

        image = Image.open(image_path).convert('RGB')
        flow = Image.open(flow_path).convert('RGB')

        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)
        if self.inputRes is not None:
            image = np.array(image.resize(self.inputRes))
            flow = np.array(flow.resize(self.inputRes))

            mask = np.array(mask.resize(self.inputRes, resample=0))
        sample = {'image': image, 'mask': mask, 'flow': flow}
        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image, flow, mask = sample['image'], sample['flow'], sample['mask']

        if self.transform is not None:
            image = self.transform(image)
            flow = self.transform(flow)

        if self.target_transform is not None:
            mask = mask[:, :, np.newaxis]
            mask = self.target_transform(mask)

        return image, flow, mask

    def __len__(self):
        return len(self.data_item)


if __name__ == '__main__':
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([to_tensor, normalize])
    target_transforms = transforms.Compose([to_tensor])
    data = DeepFish_Dataset(split='train', datasets=["DeepFish_Seagrass"], augment=True, transform=image_transforms,
                             target_transform=target_transforms)
    print(len(data))
    print('dataset processing...')

    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    for ii, (image, flow, mask) in enumerate(dataloader):
        print('======================')

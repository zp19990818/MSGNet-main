# -*- coding: utf-8 -*-
import os
import os.path as osp
import pickle
from collections import defaultdict

import numpy as np
from loguru import logger
from PIL import Image

from torch.utils.data import Dataset


class DeepFishDataset(Dataset):
    r"""
    COCO dataset helper
    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subsets: list
        dataset split name list [DeepFish, Seagrass]
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    """
    data_items = []
    default_hyper_params = dict(
        dataset_root="../MSGNet-main/datasets/DeepFish_official/",
        subsets=["train"], ratio=1.0, max_diff=10,
    )

    def __init__(self, subsets=["train"]):
        r"""
        Create davis dataset 
        """
        super(DeepFishDataset, self).__init__()
        dataset_root = self.default_hyper_params["dataset_root"]
        self.default_hyper_params["subsets"] = subsets
        self.default_hyper_params["dataset_root"] = osp.realpath(dataset_root)
        if len(DeepFishDataset.data_items) == 0:
            self._ensure_cache()

    def __getitem__(self, item):
        """
        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        record = DeepFishDataset.data_items[item]
        anno = [[anno_file, record['obj_id']] for anno_file in record["annos"]]
        sequence_data = dict(image=record["image_files"], anno=anno, flow=record['flow'])
        return sequence_data

    def __len__(self):
        return len(DeepFishDataset.data_items)

    def _ensure_cache(self):
        dataset_root = self.default_hyper_params["dataset_root"]
        for subset in self.default_hyper_params["subsets"]:
            # year = subset[-4:]
            data_name = "DeepFish_Seagrass"
            image_root = osp.join(dataset_root, "JPEGImages", "1080p")
            flow_root = osp.join(dataset_root, "Flow", "1080p")
            if data_name == "DeepFish_Seagrass":
                anno_root = osp.join(dataset_root, "Annotations", "1080P")
            else:
                anno_root = osp.join(dataset_root, "Annotations", "1080p")
            data_anno_list = []
            cache_file = osp.join(dataset_root, "cache/{}.pkl".format(subset))
            if osp.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    DeepFishDataset.data_items += pickle.load(f)
                logger.info("{}: loaded cache file {}".format(
                    DeepFishDataset.__name__, cache_file))
            else:
                meta_file = osp.join(dataset_root, "ImageSets", data_name,
                                     "train.txt")
                with open(meta_file) as f:
                    video_names = [item.strip() for item in f.readlines()]
                for video_name in video_names:
                    img_dir = os.path.join(image_root, video_name)
                    anno_dir = os.path.join(anno_root, video_name)
                    flow_dir = os.path.join(flow_root, video_name)
                    object_dict = defaultdict(list)
                    for anno_name in os.listdir(anno_dir):
                        anno_file = os.path.join(anno_dir, anno_name)
                        anno_data = np.array(Image.open(anno_file),
                                             dtype=np.uint8)
                        obj_ids = np.unique(anno_data)

                        for obj_id in obj_ids:
                            if obj_id > 0:
                                object_dict[obj_id].append(
                                    anno_name.split(".")[0])  # 按object分开存储对应图像序号
                    for k, v in object_dict.items():
                        for i in range(0, len(v)):
                            record = {}
                            frame = sorted(v)[i]
                            record["video_name"] = video_name
                            record["frame_num"] = sorted(v)[i]
                            record["obj_id"] = k
                            record["image"] = [
                                osp.join(img_dir, frame + '.jpg')
                            ]
                            record["annos"] = [
                                osp.join(anno_dir, frame + '.png')
                            ]
                            record["flow"] = [
                                osp.join(flow_dir, frame + '.jpg')
                            ]
                            data_anno_list.append(record)
                cache_dir = osp.dirname(cache_file)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_anno_list, f)
                logger.info(
                    "DeepFish_Seagrass dataset: cache dumped at: {}".format(cache_file))
                DeepFishDataset.data_items += data_anno_list


class DeepFishDatasetVal(Dataset):
    r"""
    COCO dataset helper
    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subsets: list
        dataset split name list [DeepFish, Seagrass]
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames
    """
    data_items = []
    default_hyper_params = dict(
        dataset_root="../MSGNet-main/datasets/DeepFish_official/", subsets=["val"],
        ratio=1.0, max_diff=10,
    )

    def __init__(self, subsets=["val"]):
        r"""
        Create davis dataset
        """
        super(DeepFishDatasetVal, self).__init__()
        dataset_root = self.default_hyper_params["dataset_root"]
        self.default_hyper_params["subsets"] = subsets
        self.default_hyper_params["dataset_root"] = osp.realpath(dataset_root)
        if len(DeepFishDatasetVal.data_items) == 0:
            self._ensure_cache()

    def __getitem__(self, item):
        """
        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        record = DeepFishDatasetVal.data_items[item]
        anno = [[anno_file, record['obj_id']] for anno_file in record["annos"]]
        sequence_data = dict(image=record["image_files"], anno=anno, flow=record['flow'])
        return sequence_data

    def __len__(self):
        return len(DeepFishDatasetVal.data_items)

    def _ensure_cache(self):
        dataset_root = self.default_hyper_params["dataset_root"]
        for subset in self.default_hyper_params["subsets"]:
            # year = subset[-4:]
            data_name = "DeepFish_Seagrass"
            image_root = osp.join(dataset_root, "JPEGImages", "1080p")
            flow_root = osp.join(dataset_root, "Flow", "1080p")
            if data_name == "DeepFish_Seagrass":
                anno_root = osp.join(dataset_root, "Annotations", "1080P")
            else:
                anno_root = osp.join(dataset_root, "Annotations", "1080p")
            data_anno_list = []
            cache_file = osp.join(dataset_root, "cache/{}.pkl".format(subset))
            if osp.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    DeepFishDatasetVal.data_items += pickle.load(f)
                logger.info("{}: loaded cache file {}".format(
                    DeepFishDatasetVal.__name__, cache_file))
            else:
                meta_file = osp.join(dataset_root, "ImageSets", data_name,
                                     "val.txt")
                with open(meta_file) as f:
                    video_names = [item.strip() for item in f.readlines()]
                for video_name in video_names:
                    img_dir = os.path.join(image_root, video_name)
                    anno_dir = os.path.join(anno_root, video_name)
                    flow_dir = os.path.join(flow_root, video_name)
                    object_dict = defaultdict(list)
                    for anno_name in os.listdir(anno_dir):
                        anno_file = os.path.join(anno_dir, anno_name)
                        anno_data = np.array(Image.open(anno_file),
                                             dtype=np.uint8)
                        obj_ids = np.unique(anno_data)

                        for obj_id in obj_ids:
                            if obj_id > 0:
                                object_dict[obj_id].append(
                                    anno_name.split(".")[0])  # 按object分开存储对应图像序号
                    for k, v in object_dict.items():
                        for i in range(1, len(v) - 1):
                            record = {}
                            frame = sorted(v)[i]
                            record["video_name"] = video_name
                            record["frame_num"] = sorted(v)[i]
                            record["obj_id"] = k
                            record["image"] = [
                                osp.join(img_dir, frame + '.jpg')
                            ]
                            record["annos"] = [
                                osp.join(anno_dir, frame + '.png')
                            ]
                            record["flow"] = [
                                osp.join(flow_dir, frame + '.jpg')
                            ]
                            data_anno_list.append(record)
                cache_dir = osp.dirname(cache_file)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_anno_list, f)
                logger.info(
                    "DeepFish dataset: cache dumped at: {}".format(cache_file))
                DeepFishDatasetVal.data_items += data_anno_list

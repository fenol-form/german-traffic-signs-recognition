import pandas as pd
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms.functional

import os
import json
import typing

import numpy as np
from random import sample


class DatasetGTSRB(torch.utils.data.Dataset):
    """
    :param root_folders: list of data folders
    :param path_to_classes_json:  classes.json path
    """

    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetGTSRB, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)

        # list of pairs (path to img, class idx)
        self.samples = []
        # dict of list of images for each class : classes_to_samples[cls_idx] = [list of postitions of images in self.samples]
        self.classes_to_samples = {}
        for cls, idx in self.class_to_idx.items():
            self.classes_to_samples[idx] = []

        for folder in root_folders:
            lst_classes_path = os.listdir(folder)
            for cls in lst_classes_path:
                path = os.path.join(folder, cls)
                images_filenames = os.listdir(path)
                images_pathes = [os.path.join(path, x) for x in images_filenames]
                self.samples.extend(
                    [(img_path, self.class_to_idx[cls]) for img_path in images_pathes]
                )
                self.classes_to_samples[self.class_to_idx[cls]].extend(
                    list(range(len(self.samples) - len(images_pathes), len(self.samples)))
                )

        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(128, 128),
            # augmentations...
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        """
        return: tensor with image, path to file, class_idx
        """
        path, cls_idx = self.samples[index]
        img = torchvision.io.read_image(path)
        if self.transform:
            img = img.permute(1, 2, 0).numpy()
            img = self.transform(image=img)["image"]
        return img, path, cls_idx

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        get info about classes from classes.json
        """
        with open(path_to_classes_json) as file:
            dct = json.load(file)
        class_to_idx = {cls: item["id"] for cls, item in list(dct.items())}
        classes = list(class_to_idx.keys())
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    :param root: root to data folder
    :param annotations_file: path to targets .csv file (optional)
    """

    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.root = root
        self.samples = [x for x in os.listdir(root)]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(128, 128),
            ToTensorV2(),
        ])

        self.targets = None
        if annotations_file is not None:
            # read annotations
            df = pd.read_csv(annotations_file)
            self.targets = {}  # dict: targets[path to image] = class idx
            for fn in self.samples:
                # get classes and class_to_idx dictionaries
                classes, class_to_idx = DatasetGTSRB.get_classes(path_to_classes_json)

                path_to_file = os.path.join("Test", fn)
                cls = df[df.Path == path_to_file]["ClassId"].iloc[0]
                # fill self.target
                self.targets[fn] = class_to_idx[str(cls)]

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        """
        return: tensor with image, path to file, class idx
        """
        path = self.samples[index]
        img = torchvision.io.read_image(os.path.join(self.root, path))
        if self.transform:
            img = img.permute(1, 2, 0).numpy()
            img = self.transform(image=img)["image"]
        if self.targets:
            cls_idx = self.targets[path]
        else:
            cls_idx = -1

        return img, os.path.join("Test", path), cls_idx


class BatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Class for batch samling with maintaining num of classes and examples of each class in the batch
    :param data_source: CustomDataset
    :param elems_per_class: num of images per each class in the batch
    :param classes_per_batch: num of classes in the batch
    """

    def __init__(self, data_source: DatasetGTSRB, elems_per_class: int, classes_per_batch: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.elem_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.batch_size = self.classes_per_batch * self.elem_per_class
        self.n_batches = len(self.data_source) // self.batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            sample_classes = sample(
                range(len(self.data_source.classes)),
                self.classes_per_batch
            )
            batch = []
            for i, cls_ind in enumerate(sample_classes):
                batch.extend(np.random.choice(
                    self.data_source.classes_to_samples[cls_ind],
                    size=self.elem_per_class
                ))
            yield batch

    def __len__(self):
        return self.n_batches


class SamplerForKNN(torch.utils.data.sampler.Sampler[int]):
    """
    Class for sampling images in index for K-NN
    :param data_source: CustomDataset synt_data
    :param examples_per_class: num of images of each class that should be added to an index
    """

    def __init__(self, data_source: DatasetGTSRB, examples_per_class: int) -> None:
        super().__init__(data_source=data_source)
        self.data_source = data_source
        self.examples_per_class = examples_per_class

    def __iter__(self):
        batch = []
        for i in range(len(self.data_source.classes)):
            batch.extend(
                sample(self.data_source.classes_to_samples[i], self.examples_per_class)
            )
        return iter(batch)

    def __len__(self):
        return self.examples_per_class * len(self.data_source.classes)


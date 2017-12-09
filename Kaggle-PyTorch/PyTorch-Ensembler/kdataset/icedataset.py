import random

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset
from os.path import join

import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import os
# __all__ = ('SeedlingDataset')

IMG_EXTENSIONS = [
    '.jpg',
    'png'
]

def default_loader(input_path):
    input_image = (Image.open(input_path)).convert('RGB')
    return input_image

class IcebergCustomDataSet(Dataset):
    """total datasets."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx, :, :, :], 'labels': np.asarray([self.labels[idx]])}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = image.astype(float) / 255
        return {'image': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).float()
                }


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']

        if random.random() < 0.5:
            image = np.flip(image, 1)

        return {'image': image, 'labels': labels}


class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.3:
            image = np.flip(image, 0)
        return {'image': image, 'labels': labels}


class RandomTranspose(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.7:
            image = np.transpose(image, 0)
        return {'image': image, 'labels': labels}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        img = tensor['image'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img, 'labels': tensor['labels']}

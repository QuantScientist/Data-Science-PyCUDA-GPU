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

to_tensor = transforms.Compose([transforms.ToTensor()])


def default_loader_scale(input_path, size=150):
    input_image = (Image.open(input_path)).convert('RGB')
    if size is not None:
        input_image = input_image.resize((size, size), Image.ANTIALIAS)
    return input_image


def default_loader(input_path):
    input_image = (Image.open(input_path)).convert('RGB')
    return input_image


class SeedDataset(data.Dataset):
    def __init__(self, labels, root_dir, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]  # file name
        fullname = join(self.root_dir, img_name)
        # image = Image.open(fullname).convert('RGB')
        image = default_loader(fullname)
        labels = self.labels.iloc[idx, 2]  # category_id
        #         print (labels)
        if self.transform:
            image = self.transform(image)
        return image, labels


class SeedTestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)
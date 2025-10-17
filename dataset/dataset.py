import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def get_img_file(path):
    num = 0
    imagelist = {}
    file_names = get_img_path(path)
    for file_name, label in file_names:
        dirnames = os.listdir(file_name)
        for dirname in dirnames:
            parent = os.path.join(file_name, dirname)
            filenames = os.listdir(parent)
            for filename in filenames:
                if filename.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    imagelist[num] = (os.path.join(parent, filename), label)
                    num = num + 1
    return imagelist


def get_img_path(path):
    secondFolderList = ["dongdao4_2020", "dongdao9_2020", "dongdao12_2020", "dongdao122_2020", "dongdao144_2020", "jijing816_2020"]
    # thirdFolderList = ["hard"]
    img_path_list = []
    for secondFolder in secondFolderList:
        if secondFolder == 'dongdao4_2020':
            label = 0
        elif secondFolder == 'dongdao9_2020':
            label = 1
        elif secondFolder == 'dongdao12_2020':
            label = 2
        elif secondFolder == 'dongdao122_2020':
            label = 3
        elif secondFolder == 'dongdao144_2020':
            label = 4
        elif secondFolder == 'jijing816_2020':
            label = 5
        folder = os.path.join(path, secondFolder)
        img_path_list.append((folder, label))
    return img_path_list

class Train_Seed_Classification_2021(Dataset):

    def __init__(self, path, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_label = get_img_file(self.root)


    def __len__(self):
        return len(self.images_label)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_path, label_id = self.images_label[index]
        image = cv2.imread(image_path)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label_id = self.target_transform(label_id)
        return image, label_id




class Test_Seed_Classification_2021(Dataset):

    def __init__(self, path, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_label = get_img_file(self.root)

    def __len__(self):
        return len(self.images_label)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_path, label_id = self.images_label[index]
        image = cv2.imread(image_path)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label_id = self.target_transform(label_id)
        return image, label_id




def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it

    Args:
        dataset: instance of CUB_200_2011_Train, CUB_200_2011_Test
    
    Returns:
        return: mean and std of this dataset
    """

    mean_r = 0
    mean_g = 0
    mean_b = 0

    for img, _ in dataset:
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:
        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std

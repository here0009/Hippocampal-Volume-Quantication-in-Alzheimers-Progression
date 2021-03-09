"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset
# from torchvision.transforms import transforms
import random
import numpy as np
# from skimage.util import random_noise
# from skimage.transform import rotate
random.seed(21)

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data
        self.hflip = 0.1
        self.vflip = 0.1
        self.rotate_angle = [-5, 5]
        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used
        # TASK: Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Write code that stores the 2D slice data in the last 2 dimensions of the 3D Tensors.
        # Your tensor needs to be of shape [1, patch_size, patch_size]
        # Don't forget that you need to put a Torch Tensor into your dictionary element's value
        # Hint: your 3D data sits in self.data variable, the id of the 3D volume from data array
        # and the slice number are in the slc variable.
        # Hint2: You can use None notation like so: arr[None, :] to add size-1 
        # dimension to a Numpy array
        # augmentaiont to image and label
        hflip, vflip = random.random(), random.random()
        # r_angle = random.randint(*self.rotate_angle)

        sample_image = self.data[slc[0]]['image'][slc[1]]
        sample_lable = self.data[slc[0]]['seg'][slc[1]]

        # rotation and add_noise show no improvment
        # sample_image = random_noise(sample_image)  # use the default mode ‘gaussian’
        # sample_image = rotate(sample_image, angle=r_angle, preserve_range=True)
        # sample_lable = rotate(sample_lable, angle=r_angle, preserve_range=True)

        if hflip < self.hflip:
            sample_image = np.fliplr(sample_image)
            sample_lable = np.fliplr(sample_lable)

        if vflip < self.vflip:
            sample_image = np.flipud(sample_image)
            sample_lable = np.flipud(sample_lable)

        sample['image'] = torch.from_numpy(sample_image.copy()).type(torch.float).unsqueeze(0)  # or torch.cuda.FloatTensor for gpu? use image.copy(), or it will get an error
        sample['seg'] = torch.from_numpy(sample_lable.copy()).type(torch.long).unsqueeze(0)  # or torch.cuda.IntTensor for gpu?

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)

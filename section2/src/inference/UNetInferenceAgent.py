"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        volume = med_reshape(volume, new_shape=(volume.shape[0], self.patch_size, self.patch_size))
        return self.single_volume_inference(volume)
        

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After
        # that, put all slices into a 3D Numpy array. You can verify if your method is
        # correct by running it on one of the volumes in your training set and comparing
        # with the label in 3D Slicer.
        x = volume.shape[0]
        mask = np.zeros(volume.shape)
        for i in range(x):
            _slice = volume[i, :, :]
            if _slice.min() != _slice.max():
                _slice = (_slice - _slice.min()) / (_slice.max() - _slice.min())  # change the range of _slice to [0, 1]
            _slice_torch = torch.from_numpy(_slice).type(torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
            pred = self.model(_slice_torch)
            pred = np.squeeze(pred.cpu().detach())
            mask[i, :, :] = torch.argmax(pred, dim=0)
        return mask

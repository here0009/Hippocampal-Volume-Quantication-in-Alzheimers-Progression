import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load
from utils.utils import med_reshape

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"G:\MOOC\Udacity\NanoDegree AI for healthcare\Applying AI to 3D Medical Imaging Data\Hippocampal Volume Quanti cation in Alzheimer's Progression\section1\out"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = os.path.join(*['..', 'out'])

image_dir = os.path.join(root_dir, 'images')
label_dir = os.path.join(root_dir, 'labels')
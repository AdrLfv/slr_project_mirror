
from torchvision.transforms import ToTensor, Lambda
from typing import Tuple
#from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn

import cv2
import numpy as np
import os


class Data_augmentation():

    def __init__(self, data_frame, x_shift, y_shift, scale):

        # on récupère chaque frame de chaque sequence de chaque action, on effectue les modifications sur celle-ci,
        # on sauvegarde les données de cette nouvelle frame dans une nouvelle séquence dans la même action,
        # on effectue sur la frame suivante les mêmes modifications et on sauvegarde dans la nouvelle séquence
        # lorsque l'on passe à la séquence suivante on change les paramètres de modification
        self.data_frame = []
        #print("data frame : ", np.array(data_frame).shape)
        for ind, _ in enumerate(data_frame[0:]):
            if (ind % 2 == 0):
                self.data_frame.append(data_frame[ind] * scale + x_shift)
                self.data_frame.append(data_frame[ind+1] * scale + y_shift)

        self.data_frame = self.data_frame
        #print("self.data_frame : ", np.array(self.data_frame).shape)

    def __getitem__(self):
        return self.data_frame

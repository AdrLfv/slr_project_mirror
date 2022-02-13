
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
        self.new_data_frame = []
        x_max = max(data_frame[x] for x in range(0,len(data_frame),2))
        y_max = max(data_frame[y] for y in range(1,len(data_frame),2))
        x_min = min(data_frame[x] for x in range(0,len(data_frame),2))
        y_min = min(data_frame[y] for y in range(1,len(data_frame),2))
        #print("data frame : ", np.array(data_frame).shape)
        for ind, _ in enumerate(data_frame[0:]):
            if (ind % 2 == 0):
                self.new_data_frame.append( (data_frame[ind]-(x_max+x_min)/2) * scale + x_shift + (x_max+x_min)/2)
                self.new_data_frame.append( (data_frame[ind+1]-(y_max+y_min)/2) * scale + y_shift + (y_max+y_min)/2)

        # self.new_data_frame = []
        # for ind, _ in enumerate(data_frame[0:]):
        #     if (ind % 2 == 0):
        #         self.new_data_frame.append(data_frame[ind] * scale + x_shift)
        #         self.new_data_frame.append(data_frame[ind+1] * scale + y_shift)

    def __getitem__(self):
        return self.new_data_frame

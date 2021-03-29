# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:03:49 2021

@author: Daily Milanés Hermosilla
"""

from keras import backend as K
import keras.utils
import numpy as np
import math
import os
import scipy.io as sio

# Load .mat files corresponding to every trial located in a specific directory
# Return four lists, corresponding to left hand, right hand, feet and tongue trials

def load_eeg(dir):
    
    left = []
    right = []
    foot = []
    tongue = []
    contenido = os.listdir(dir)
    for fichero in contenido:
        nombre1 = fichero[:3] + "trial"
        file_load = sio.loadmat(os.path.join(dir, fichero))
        draw = file_load[nombre1]     
        x = np.array(draw, dtype = np.float32)
        if '_Right_'  in fichero:
            right.append(x)
        elif '_Left_' in fichero:
            left.append(x)
        elif '_Foot_' in fichero:
            foot.append(x)
        elif '_Tongue_' in fichero:
            tongue.append(x)
    return left, right, foot, tongue


# This function create crops for each trial especified in data parameter. 
# cropDistance: is stride between 2 crops
# cropsize: size of each crop, in our model is considered 1000 samples.
# Return a list of numpy array with all crops 


def makeCrops(data, cropDistance, cropSize):
    
    z = []
    timeSpam = len(data[:, 0])
    channels = len(data[0, :])
    cropsCount = int(math.ceil((timeSpam-cropSize) / cropDistance))    
    if K.image_data_format() == 'channels_first':
        y = np.zeros((1, cropSize, channels), dtype=np.float32)
        for i in range(0, cropsCount):
             j = i*cropDistance
             y[0,:,:] = data[j:cropSize+j, :]
             z.append(np.copy(y))   
    else:
        y = np.zeros((cropSize, channels, 1), dtype=np.float32)
        for i in range(0, cropsCount):
             j = i*cropDistance
             y[:,:,0] = data[j:cropSize+j, :]
             z.append(np.copy(y))   
    return z



# This function create the numpy arrays to train or to validation from the lists of trials of left hand (leftList),
      # right hand(rightList), feet (footList) and tongue(tongueList)
# indexs is a list that contains the number of trials to train or to validate 	

def makeNumpys(leftList, rightList, footList, tongueList, 
               cropDistance, cropSize, indexs):
     
    data = []
    labels = []    
# Analize if the dataset is 2a or 2b, taking into consideration the lenght of footList  
    nb_class=2
    if (len(footList) > 0): 
        nb_class=4
     
    for i in indexs:
        z = makeCrops(leftList[i], cropDistance, cropSize)
        data = data + z
        ilabel = keras.utils.to_categorical(0, nb_class).astype('float32')
        for k in range(len(z)):
            labels.append(ilabel)
        z = makeCrops(rightList[i], cropDistance, cropSize)
        data = data + z
        ilabel = keras.utils.to_categorical(1, nb_class).astype('float32')
        for k in range(len(z)):
            labels.append(ilabel)
        if (len(footList) > 0):    
            z = makeCrops(footList[i], cropDistance, cropSize)
            data = data + z
            ilabel = keras.utils.to_categorical(2, nb_class).astype('float32')
# All crops created from a same trial taken the same label
            for k in range(len(z)):    
                labels.append(ilabel)
            z = makeCrops(tongueList[i], cropDistance, cropSize)
            data = data + z
            ilabel = keras.utils.to_categorical(3, nb_class).astype('float32')
# All crops created from a same trial taken the same label
            for k in range(len(z)):
                labels.append(ilabel)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

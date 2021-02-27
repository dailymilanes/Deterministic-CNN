# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:09:07 2021

@author: Daily Milanés Hermosilla
"""

# The authors recomend dropoutRate 0.8 to dataset 2a and dropoutRate 0.5 to dataset 2b
# The authors to ensure repeatability of the Experiment #2, recomend to use the followings seeds
# dataset 2a: A01=7, A02=5, A03=11, A04=11, A05=11, A06=2, A07=2, A08=11, A09=2
# dataset 2b: B01=19, B02=5, B03=5, B04=2, B05=19, B06=5, B07=5, B08=7, B09=7

# To ensure repeatability of the experiment #3 and #4, please use seed=5

# To run any experiment, select appropietly subject, seed, dropoutRate, cropDistance=2, cropSize=1000
# Depending on the dataset, nb_classes=4, channel=22, fraction=5/6 to dataset 2a
# nb_classes=2, channel=3, fraction=4/5 to dataset 2b

# Tu run experiment #4, please exclude parameter must be different of 0, and specify subject unknown

import eegTrain
import eegEvaluate

global dataDirectory
eegTrain.dataDirectory = '../Data/'    

global weightsDirectory 
eegTrain.weightsDirectory = '../Weights/'


# Experiment #1
#eegTrain.trainKFold('B01',dropoutRate=0.5, cropDistance = 2, cropSize = 1000)

# Experiment #2
# eegTrain.intraSubjectTrain('B01', 19,
#           dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
#           fraction = 5/6, channels = 3, nb_classes = 2)

# # Experiment #3
# eegTrain.interSubjectTrain(5, dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
#                       fraction = 4/5, channels = 3, nb_classes = 2,
#                       exclude = 0)

# # Experiment 4
# eegTrain.trainUnkownSubject(5,dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
#                       fraction = 4/5, channels = 3, nb_classes = 2,
#                       exclude = 2)
# Evaluate function
# Specified the weightsFileName
#weightsFileName='../Weights/B01_Seed_19_R_1_d_0.50_c_2_x_0_weights.hdf5'
#eegEvaluate.eegEvaluate('B01', cropDistance=2, cropSize=1000, weightsFileName=weightsFileName,
     #                   dropoutRate = 0.5,channels = 3, nb_classes = 2)


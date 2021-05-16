# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:09:07 2021

@author: Daily Milanés Hermosilla
"""

# The authors recomend dropoutRate 0.8 to dataset 2a, dropoutRate 0.5 to dataset 2b and dropout 0.9 to dataset IVa


# To ensure repeatability of the experiment #2, #3 and #4, please use seed=1 up to 16

# To run any experiment, select appropietly subject, seed, dropoutRate, cropDistance=2, cropSize=1000
# Depending on the dataset, nb_classes=4, channel=22, fraction=5/6 to dataset 2a
# nb_classes=2, channel=3, fraction=4/5 to dataset 2b

# Tu run experiment #4, please exclude parameter must be different of 0, and specify unknown subject



import eegTrain
import eegEvaluate

global dataDirectory
eegTrain.dataDirectory = '../Data/'    

global weightsDirectory 
eegTrain.weightsDirectory = '../Weights/'


# Experiment #1
#eegTrain.trainKFold('B01',dropoutRate=0.5, cropDistance = 2, cropSize = 1000)

#Nested KFold

#trainNestedKFold('aa', dropoutRate=0.9, optim='adam', cropDistance = 2, cropSize = 875)

# Experiment #2
# eegTrain.intraSubjectTrain('B01', dropoutRate=0.5, cropDistance = 2, cropSize = 1000)

# # Experiment #3
# eegTrain.interSubjectTrain(dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
#                       nb_classes = 2,exclude = 0)

# # Experiment 4
#eegTrain.trainUnkownSubject(dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
 #                      nb_classes = 2, exclude = 1)
# Evaluate function
# Specified the weightsFileName
#weightsFileName='../Weights/B01_Seed_19_R_1_d_0.50_c_2_x_0_weights.hdf5'
#eegEvaluate.eegEvaluate('B01', cropDistance=2, cropSize=1000, weightsFileName=weightsFileName,
     #                   dropoutRate = 0.5,channels = 3, nb_classes = 2)


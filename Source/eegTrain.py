# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:59:54 2021

@author: Daily Milanés Hermosilla
"""


import keras.utils
import random as random
import keras.metrics
import pandas as pd
import eegUtils
import model
import scipy.io as sio
from sklearn.model_selection import RepeatedKFold


# Global Variable that contains the path where the dataset is preprocessed
dataDirectory = ''

# Global Variable where the pretrained model weights will be saved 
weightsDirectory = ''

# All functions to train the differents experiments.
# repeat= Parameter that indicates the repetition of each experiment, 
         # in order to differentiate the files of each weights and results of each repetition in the training process.
# We use Adam as optimizer, and loss=categorical crossentropy

# This is a funcion to realize the training for experiments #2,#3 and #4
# subject: subject identifier (intra-subject training, Experiment #2), if "All" correspond to Experiment #3 or #4
# seed:Seed to initialize the python random number generator to ensure repeatability of the experiment. 
#      For experiment 2 please, use the seed shows in main.py
# Divide randomly each list left, right, foot, and tongue in two parts, one to train and other to validate the model
# The portion selected to train and to validate depends of the dataset
# If dataset 2a fraction=5/6, if dataset 2b fraction=4/5 
# left, right, foot and tongue: Lists of trial corresponding to left hand, right hand, feet and tongue
#channels: Number of channels of EEG, for dataset 2a are 22,for dataset 2b are 3
#nb_classes:Number of classes, for dataset 2a are 4, for dataset 2b are 2.
#exclude: If experiment 4, exclude different of 0, for the other experiments exclude must be 0.

 
def oneSubjectTrain(left, right, foot, tongue, subject, seed, repeat, 
              exclude = 0, cropDistance = 2, cropSize = 1000, dropoutRate = 0.5, 
              fraction = 5/6, channels = 22, nb_classes = 4):
    
    len1 = len(left)
    len2 = len(foot)
    random.seed(seed)
    
# Shuffle data
    l = random.sample(left, len1)
    r = random.sample(right, len1)
    f = random.sample(foot, len2)
    t = random.sample(tongue, len2)
    
    usedForTraining = int(round(fraction*len1))
    	
# trains_indices: List with number of trial to train according to partition realized
# val_indices: List with number of trial to validate according to partition realized
    
    train_indices = range(usedForTraining)
    val_indices = range(usedForTraining,len1,1)
        
# Create numpy arrays to train and to validate according to the partition realized
    
    train_data, train_labels = eegUtils.makeNumpys(l, r, f, t, cropDistance, cropSize, train_indices)
    val_data, val_labels = eegUtils.makeNumpys(l, r, f, t, cropDistance, cropSize, val_indices)
    
    droputStr = "%0.2f" % dropoutRate 
    
    classifier = model.createModel(Samples = cropSize, dropoutRate=dropoutRate, 
                                   Chans = channels, nb_classes = nb_classes)   
    baseFileName = weightsDirectory + subject + '_Seed_' +str(seed) +'_R_'+ str(repeat)+ '_d_' + droputStr + '_c_'+str(cropDistance)+ '_x_'+str(exclude)

# File name which the weights will be saved
    weightFileName = baseFileName+'_weights.hdf5'
# Establish the callbacks to training  

    callback1 = keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath = weightFileName,
                                                save_best_only=True,
                                                save_weights_only=True)
    callback2 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)    
    callback3 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=5, verbose=1, min_delta=1e-6) 
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = classifier.fit(train_data, train_labels, epochs = 200, verbose = 1, shuffle = True, 
                             validation_data = (val_data, val_labels), 
                             callbacks=[callback1, callback2, callback3])
    hist_df=pd.DataFrame(history.history)
    
# The training history will be saved in the same path that the weights
    file=baseFileName+'.json'
    with open(file,mode='w') as f:
      hist_df.to_json(f)
    f.close()
    
    
# This function prepares a intra subject training for Experiment #2. The number of repetitions is 10 
   
def intraSubjectTrain(subject, seed,
          dropoutRate=0.5, cropDistance = 50, cropSize = 1000):
    
    if subject[0] == 'A':
      channels=22
      fraction=5/6
      nb_classes=4
    else:
      channels=3
      fraction=4/5
      nb_classes=2
    
    trainDirectory = dataDirectory + subject + '/Training/'
    left,right,foot,tongue = eegUtils.load_eeg(trainDirectory)    
    for j in range(1,11):
         oneSubjectTrain(left, right, foot, tongue, subject, seed, j,
                   cropDistance = cropDistance, cropSize = cropSize, 
                   dropoutRate = dropoutRate, fraction = fraction, 
                   channels = channels, nb_classes = nb_classes)


# This function prepares a inter-subject training for Experiments #3 and #4. The number of repetitions is 10 
# If the experiment is #3 exclude=0, and all data training for all subjects are load
# If experiment #4, exclude different of 0 and all data training and evaluating for all subjects except exclude subject are load
  
def interSubjectTrain(seed, dropoutRate=0.5, cropDistance = 50, cropSize = 1000,
                      fraction = 5/6, channels = 22, nb_classes = 4,
                      exclude = 0):
    if nb_classes==4:
      data='A'
      channels=22
      fraction=5/6
    else:
      data='B'
      channels=3
      fraction=4/5
    start = 1  
    if exclude == 1:
        start = 2
    left,right,foot,tongue = eegUtils.load_eeg(dataDirectory + data+'0'+str(start)+'/Training/') 
    if exclude!=0:
        lTmp, rTmp, fTmp, tTmp = eegUtils.load_eeg(dataDirectory + data+ '0'+str(start)+'/Evaluating/')
        left = left + lTmp
        right = right + rTmp
        foot = foot + fTmp
        tongue = tongue + tTmp
        
    print(start)
    for i in range(start + 1, 10):
        if (i == exclude):
            continue
        lTmp, rTmp, fTmp, tTmp = eegUtils.load_eeg(dataDirectory + data+ '0'+str(i)+'/Training/')
        left = left + lTmp
        right = right + rTmp
        foot = foot + fTmp
        tongue = tongue + tTmp
        if exclude!=0:
            lTmp, rTmp, fTmp, tTmp = eegUtils.load_eeg(dataDirectory + data+ '0'+str(i)+'/Evaluating/')
            left = left + lTmp
            right = right + rTmp
            foot = foot + fTmp
            tongue = tongue + tTmp
        print(i)
        print(len(left))
    for j in range(1,11):
        oneSubjectTrain(left, right, foot, tongue, 'All', seed, j,
                   cropDistance = cropDistance, cropSize = cropSize, 
                   dropoutRate = dropoutRate, fraction = fraction, 
                   channels = channels, nb_classes = nb_classes, 
                   exclude = exclude)

# This function implemented a KFold with 10 repetitions, dataset2a is 9x10 and in dataset 2b is 10x10
# The weights are not saved, only the history on training

def eegKFold(left, right, foot, tongue, folds, subject,
             cropDistance = 2, cropSize = 1000, 
             dropoutRate = 0.5, nb_classes = 4):
    result = []
    n=1
    droputStr = "%0.2f" % dropoutRate 
    cv = RepeatedKFold(n_splits = folds, n_repeats = 10)
    for train_indices, test_indices in cv.split(range(len(left))):
        train_data, train_labels = eegUtils.makeNumpys(left, right, foot, tongue, cropDistance, cropSize, train_indices)
        val_data, val_labels = eegUtils.makeNumpys(left, right, foot, tongue, cropDistance, cropSize, test_indices)
        Channels = 22
        if (foot == []):
            Channels = 3
            nb_classes=2
            
        classifier = model.createModel(Samples = cropSize, dropoutRate = dropoutRate,
                                      nb_classes = nb_classes, Chans = Channels)
      
        callback2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12)    
        callback3 = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.33, patience=5, verbose=1, min_delta=1e-6) 
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = classifier.fit(train_data, train_labels, epochs = 100, verbose = 1,
                            validation_data = (val_data, val_labels) , callbacks=[callback2, callback3])
        
        hist_df = pd.DataFrame(history.history) 
        result.append(str(hist_df))
        result.append("======================================")
       
        file = weightsDirectory+subject+'_KFold'+ '_d_' + droputStr + '_c_'+str(cropDistance)+'_exp_'+str(n)+'.json'
        with open(file,mode='w') as f:
           hist_df.to_json(f)
        f.close() 
        n=n+1
    return result    

# This Function prepares the conditions to realize a cross validation (KFlod) for a specific subject
# The function that realizes a KFold is eegKfold and it is calling here
# If dataset 2a number of Folds= 9, if dataset 2b number of Folds=10
  # left, right, foot and tongue are lists that contains the trials corresponding to each label
  

def trainKFold(subject, dropoutRate=0.5, optim='adam', cropDistance = 50, cropSize = 1000):
    left, right, foot, tongue = eegUtils.load_eeg(dataDirectory + subject+'/Training/')
    if (subject[0] == 'A'):
        folds = 9
    else:
        folds = 10        
    return eegKFold(left, right, foot, tongue, folds, cropDistance = cropDistance, subject=subject,
                    cropSize = cropSize, dropoutRate = dropoutRate)        


#This function is to perform the experiment #4
#exclude variable is a number that represents the subject whose data will not be used in the training process.

def trainUnkownSubject(seed, dropoutRate=0.5, cropDistance = 50, cropSize = 1000,
                      fraction = 5/6, channels = 22, nb_classes = 4,
                      exclude = 1):
    interSubjectTrain(seed=seed, dropoutRate=dropoutRate, cropDistance = cropDistance, 
                      cropSize = cropSize,fraction = fraction, channels = channels,
                      nb_classes = nb_classes, exclude = exclude)
    
         
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:03:49 2021

@author: Daily Milanés Hermosilla
"""
import eegUtils
import model
import eegTrain
import math

# This function realizes a evaluation on testing set in a specific subject.
# It is necessary give the path of weights, in weightsFileName variable
# Return the evaluation results.
# This function is used in all experiments, except KFold.
# This function receives a parameter named weightsFileName,
   # this weightsFileName is a file, builted as: 
   # subject +'_d_' + droputStr + '_c_'+str(cropDistance)+  '_Seed' +str(seed) + '_exp_'+ str(repeat)+'_exclude_'+str(exclude)+'_weights.hdf5'    
   # and ubicated in weightsDirectory
   # To load this file, if Experiment #3 or #4, subject = All
   # example: weightsDirectory+'A01_SE_d_0.80_c_2_seed9_exp_9_exclude_0_weights.hdf5'   in Experiment #2 on subject A01
   
def eegEvaluate(subject, cropDistance, cropSize, weightsFileName, dropoutRate = 0.5):
    
    if subject[0] == 'A':
       channels=22
       nb_classes=4
       strLabels=['Left','Right', 'Foot', 'Tongue']
    elif subject[0] == 'B':
       channels=3
       nb_classes=2
       strLabels=['Left','Right']
    dropoutStr = "%0.2f" % dropoutRate 
    seed=1
    testDirectory = eegTrain.dataDirectory + subject + '/Evaluating/'   
    
    # If experiment #4 please add training data to testDirectory
        
    datalist, labelslist = eegUtils.load_eeg(testDirectory, strLabels)
    test_indices = range(len(datalist))
   
    gen3 = eegUtils.Generator(datalist, labelslist, nb_classes, test_indices, channels, cropDistance, cropSize, int(math.ceil(125 / cropDistance)))
    pasosxepocaE= int(math.ceil((len(test_indices)*int(math.ceil((1125-cropSize)/cropDistance)))/32))

    for i in range(1,17):
       classifier = model.createModel(Samples = 1000, dropoutRate = dropoutRate, 
                  Chans = channels, nb_classes = nb_classes)
       classifier.load_weights(weightsFileName)
       classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
       result = classifier.evaluate(gen3, steps=pasosxepocaE, verbose=1) 
       
       seed=seed+1
       
    return result 


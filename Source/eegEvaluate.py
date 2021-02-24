# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:03:49 2021

@author: Daily Milanés Hermosilla
"""
import eegUtils
import model
import eegTrain

# This function evaluate in a specific subject, subject is specified in subject.
# It is necessary give the path of weights, in weightsFileName
# Return the evaluation results.
# This function is used in all experiments, except KFold, in which the weights are not saved.
# This function receives a parameter named weightsFileName,
   # this weightsFileName is a file, builted as: 
   # subject + '_Seed_' +str(seed) +'_R_'+ str(repeat)+ '_d_' + droputStr + '_c_'+str(cropDistance)+ '_x_'+str(exclude)+'_weights.hdf5'    
   # and ubicated in weightsDirectory
   # To load this file, if Experiment #3 or #4, subject = All
   # example: weightsDirectory+'B01_Seed_19_R_1_d_0.50_c_2_x_0_weights.hdf5'
   
def eegEvaluate(subject, cropDistance, cropSize, weightsFileName, dropoutRate = 0.5,
                channels = 22, nb_classes = 4):
   
    testDirectory = eegTrain.dataDirectory + subject + '/Evaluating/'
    leftList,rightList,footList,tongueList = eegUtils.load_eeg(testDirectory)    
    indexs = range(len(leftList))
    test_data, test_labels = eegUtils.makeNumpys(leftList, rightList, footList, 
                        tongueList, cropDistance, cropSize, indexs)
    classifier = model.createModel(Samples = cropSize, dropoutRate = dropoutRate, 
                 Chans = channels, nb_classes = nb_classes)
    
    classifier.load_weights(weightsFileName)
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                       metrics = ['accuracy'])
    
    result = classifier.evaluate(test_data, test_labels, verbose = 1)
    return result



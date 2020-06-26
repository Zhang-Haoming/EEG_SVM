'''
#BCICIV_2b_gdf
#9subjects and 5 sessions per subject
#fs=250hz
ID Training              Evaluation
1 B0101T, B0102T, B0103T B0104E, B0105E
2 B0201T, B0202T, B0203T B0204E, B0205E
3 B0301T, B0302T, B0303T B0304E, B0305E
4 B0401T, B0402T, B0403T B0404E, B0405E
5 B0501T, B0502T, B0503T B0504E, B0505E
6 B0601T, B0602T, B0603T B0604E, B0605E
7 B0701T, B0702T, B0703T B0704E, B0705E
8 B0801T, B0802T, B0803T B0804E, B0805E
9 B0901T, B0902T, B0903T B0904E, B0905E

#5min at the begining each section
    2min eye open(a fixation cross) 1min eye closed 1min eye movement four sections
    (15 seconds artifactswith 5 seconds resting in between)
    (eye blinking, rolling, up-down or left-right movements)
no EOG block is available in session B0102T and B0504E

motor imagery
    left&right
    20trials per run % 120 trials per session
    visual cue presented for 1.25s
    imagine 4s
    short break 1.5s
    1s in break to avoid adaptation
    imagery period:4-7s

'''


import mne
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from EEG_SVM import *
from mne_extract import *
#事件的对应关系
eventDescription = {'276': "eyesOpen", '277': "eyesClosed", '768': "startTrail", '769':"cueLeft", '770':"cueRight", '781':"feedback", '783':"cueUnknown",
            '1023': "rejected", '1077': 'horizonEyeMove', '1078': "verticalEyeMove", '1079':"eyeRotation",'1081':"eyeBlinks",'32766':"startRun"}

#location
path= 'E:\BCICIV_2b_gdf'   ##
filename='\B0103T.gdf'
file_path = path+filename
print(file_path)

#extract event
trainData , trainLabels = extract_event(file_path)

####################################################### SVM method 1 (simple) ################################################################
#your kernel type?
kernel_type='linear'         # 'linear', 'poly', 'rbf', 'sigmoid'
Test_size = 0.8              #  The ratio of test sets (0~1)
scores,class_balance = ShuffleSplit_SVM(trainData, trainLabels,kernel_type,Test_size)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

####################################################### SVM method 2 (complex) ###############################################################

#所调参数  adajust parameter   could be adjust

#your kernel type?
kernel_type='rbf'         # 'linear', 'poly', 'rbf', 'sigmoid'

# C:penalty coefficient
# gamma:
# degree: Highest degree of a polynomial kernel function,for poly
# coef0
if kernel_type=='rbf':
    c_range = np.logspace(-1, 2, 5, base= 3)
    gamma_range = np.logspace(-6, 1, 8, base=3)
    param_grid = [{ 'svc__C': c_range, 'svc__gamma': gamma_range}]

elif kernel_type == 'poly':
    c_range = np.logspace(-5, 5, 11, base=3)
    gamma_range = np.logspace(-3, 3, 7, base=3)
    degree_range = [ 2 ,3 ,4, 5 ,6]
    coef0_range = np.logspace(-3, 3, 7, base=3)
    param_grid = [{'svc__C': c_range,'svc__gamma':gamma_range,'svc__coef0':coef0_range,'svc__degree':degree_range}]

elif kernel_type =='linear':
    c_range = np.logspace(-1, 2, 5, base= 3)
    gamma_range = np.logspace(-6, 1, 8, base=3)
    param_grid = [{ 'svc__C': c_range, 'svc__gamma': gamma_range}]

elif kernel_type =='sigmoid':
    c_range = np.logspace(-5, 5, 11, base=3)
    gamma_range = np.logspace(-3, 3, 7, base=3)
    coef0_range = np.logspace(-3, 3, 7, base=3)
    param_grid = [{ 'svc__C': c_range,'svc__gamma':gamma_range,'svc__coef0':coef0_range}]

else :
    print('wrong kernel type')
    exit()
    
train_grade, test_score, para=GridSearch_SVM(trainData, trainLabels,param_grid )

print('accuracy of train set: %s' % train_grade)
print('accuracy of test set: %s' % test_score)
print('the best para is: %s'% para)



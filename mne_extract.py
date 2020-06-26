import mne
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle


def extract_event(file_path):

    #事件的对应关系
    eventDescription = {'276': "eyesOpen", '277': "eyesClosed", '768': "startTrail", '769':"cueLeft", '770':"cueRight", '781':"feedback", '783':"cueUnknown",
                '1023': "rejected", '1077': 'horizonEyeMove', '1078': "verticalEyeMove", '1079':"eyeRotation",'1081':"eyeBlinks",'32766':"startRun"}

    #input
    rawDataGDF = mne.io.read_raw_gdf(file_path,preload=True,eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
    #print(rawDataGDF.info)
    #print(rawDataGDF.info['ch_names'])
    #print(rawDataGDF.info['sfreq'])


    #手动创建 
    #channel type
    ch_types = ['eeg', 'eeg', 'eeg','eog','eog','eog']
    #ch_names = ['EEG:C3','EEG:Cz',  'EEG:C4','EOG:ch01', 'EOG:ch02', 'EOG:ch03']

    info = mne.create_info(ch_names=rawDataGDF.info.ch_names, sfreq= rawDataGDF.info['sfreq'], ch_types=ch_types)
    # 创建数据结构体
    data = np.squeeze(np.array([rawDataGDF['EEG:Cz'][0], rawDataGDF['EEG:C3'][0], rawDataGDF['EEG:C4'][0], rawDataGDF['EOG:ch01'][0] , rawDataGDF['EOG:ch02'][0] ,rawDataGDF['EOG:ch03'][0]]))

    # 创建RawArray类型的数据
    rawData = mne.io.RawArray(data, info)

    #画图 plot
    #rawData.plot()
    #plt.show()

    #绘制各通道的功率谱密度  plot PSD
    #rawDataGDF.plot_psd()
    #plt.show()

    #get event
    event,eventnum= mne.events_from_annotations(rawDataGDF)
    #print(event)
    #print(eventnum)

    #整理  disposal data
    event_ID={}

    for i in eventnum:
        event_ID[eventDescription[i]]=eventnum[i]
    #print(event_ID)

    epochs=mne.Epochs(rawData,event,event_ID, tmax=3 ,event_repeated = 'merge',preload=True)#merge合并
    #epochs.plot(block=True)
    #plt.show()

    #滤波  filter
    epochs_train = epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    # 提取数据以及标签  
    channels=['EEG:C3','EEG:Cz',  'EEG:C4']

    trainData = epochs_train['cueLeft', 'cueRight'].get_data(channels)
    trainLabels = epochs_train['cueLeft', 'cueRight'].events[:, -1]
    #print(trainData)
    #print(trainLabels) 
    return trainData , trainLabels

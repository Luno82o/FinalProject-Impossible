# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 02:38:32 2022

@author: USER
"""

import librosa
import numpy as np

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed
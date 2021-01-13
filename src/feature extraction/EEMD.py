#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:20:56 2020

@author: andrine
"""

import pandas as pd
import os
import numpy as np
from PyEMD import EEMD
import sys
from scipy.stats import skew 
import random
import time
import librosa 
from scipy.signal import resample 

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/andrine/Desktop/Master/Andrine_Elsetronning_PreProject2020/src')

import utility

target_rate = 44100

def downsample(audio, sr):
    secs = len(audio)/sr # Number of seconds in signal X
    new_sr = 8000
    samps = round(secs*new_sr)     # Number of samples to downsample
    new_audio = resample(audio, samps)
   
    return new_audio, new_sr

def get_n_best_IMFs(audio, t, n_imfs, n_sifts, select_imfs):
    m_trials = 2
    eemd = EEMD(trials=m_trials)
    eemd.spline_kind="slinear"
    eemd.FIXE = n_sifts
    # Execute EEMD on S
    result = []
    
    IMFs = eemd.eemd(audio, t, max_imf = n_imfs)
    print(len(IMFs))
    if (len(IMFs) != 11):
        select_imfs = [2,3,4,5,6]
    for idx in range(len(IMFs)):
        if idx in select_imfs:
            result.append(IMFs[idx])
            
    return result


def get_t(y, sr):
    n = len(y)
    t = np.linspace(0, 1/ sr, n)
    return t

def get_entropy(audio):
    audio_nz = audio[audio != 0]
    return - np.sum(((audio_nz**2)*np.log(audio_nz**2)))
    
def get_energy(audio):  
    N = len(audio)
    return np.sum(np.abs(audio) ** 2) / N
    

def get_features(filename , data, sr , label , n_mfcc = 10):

    ft1 = librosa.feature.mfcc(data, sr = sr, n_mfcc=n_mfcc)
    ft2 = librosa.feature.zero_crossing_rate(data)[0]
    ft3 = librosa.feature.spectral_rolloff(data)[0]
    ft4 = librosa.feature.spectral_centroid(data)[0]
    ft5 = librosa.feature.spectral_contrast(data)[0]
    ft6 = librosa.feature.spectral_bandwidth(data)[0]

    ### Get HOS and simple features 
    ft0_trunc = np.hstack((np.mean(data), np.std(data), skew(data), np.max(data), np.median(data), np.min(data), get_energy(data), get_entropy(data)))
  
    ### MFCC features
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
    
    ### Spectral Features 
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
    return pd.Series(np.hstack((ft0_trunc , ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc, filename, label)))


def get_column_names(select_imfs, n_mfcc = 10):
    stats = ['mean' , 'std' , 'skew', 'max', 'median', 'min']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']
    cols = []
    for n in select_imfs:
        base = f'IMF_{n}_'
        mfcc = []
        for i in range(1 , n_mfcc + 1):
            mfcc.append(f'mfcc_{i}')
        for stat in stats_HOS:
            cols.append(base + stat)    
        for stat in stats:
            for m in mfcc:
                cols.append(base + m + '_' + stat)
           
        for s in others:
            for stat in stats:
                cols.append(base + s + '_' + stat)
               
    cols.append('name')
    cols.append('label')    
    return cols


def get_small_dataset(target_rate, select_imfs, n_samples = 10, n_imfs = 10, n_sifts = 15 , n_mfcc = 10):
    random.seed(20)
    crackle = os.listdir('../../data/Kaggle/processed/audio_slices_crackle/crackle/')
    crackle_audio_idx = random.sample(range(len(crackle)), n_samples)
    no_crackle = os.listdir('../../data/Kaggle/processed/audio_slices_crackle/no-crackle/')
    no_crackle_audio_idx = random.sample(range(len(no_crackle)), n_samples)
    
    df = pd.DataFrame()
    for i in range(0,n_samples):
        sr_c, y_c = utility.read_wav_file('../../data/Kaggle/processed/audio_slices_crackle/crackle/' + crackle[crackle_audio_idx[i]], target_rate)
        sr_nc, y_nc = utility.read_wav_file('../../data/Kaggle/processed/audio_slices_crackle/no-crackle/' + no_crackle[no_crackle_audio_idx[i]], target_rate)
        
        filename_c , filename_nc =  crackle[crackle_audio_idx[i]], no_crackle[no_crackle_audio_idx[i]]
        
        print(f'Processing (EEMD): sample number {i} of {n_samples}')
        y_c = utility.denoise_audio(y_c)
        y_nc = utility.denoise_audio(y_nc)
        
        y_c, sr_c = downsample(y_c, sr_c)
        y_nc, sr_nc = downsample(y_nc, sr_nc)

        t_c = get_t(y_c, sr_c)
        t_nc = get_t(y_nc, sr_nc)

        IMFs_c = get_n_best_IMFs(y_c, t_c, n_imfs, n_sifts, select_imfs)
        IMFs_nc = get_n_best_IMFs(y_nc, t_nc, n_imfs, n_sifts, select_imfs)
        

        row_c = pd.DataFrame()
        row_nc = pd.DataFrame()
        print(f'Processing / feature extract: sample number {i} of {n_samples}')
        for (idx, (imf_c, imf_nc)) in enumerate(zip(IMFs_c , IMFs_nc)):
            if idx == 0:
                row_c = pd.DataFrame()
                row_c['name'] = [filename_c]
                row_c = row_c['name'].apply(get_features, data = imf_c , sr = sr_c, label = 'crackle', n_mfcc = n_mfcc)
                
                row_nc = pd.DataFrame()
                row_nc['name'] = [filename_nc]
                row_nc = row_nc['name'].apply(get_features, data = imf_nc , sr = sr_nc, label = 'no-crackle', n_mfcc = n_mfcc)
             
                continue 
            row_c_new = pd.DataFrame()
            row_c_new['name'] = [filename_c]
            row_c_new = row_c_new['name'].apply(get_features, data = imf_c , sr = sr_c, label = 'crackle', n_mfcc = n_mfcc)
            row_c = pd.merge(row_c, row_c_new ,on=[128,129])
            row_c['129'] = 'crackle'
            row_c['128'] = filename_c
            
            row_nc_new = pd.DataFrame()
            row_nc_new['name'] = [filename_nc]
            row_nc_new = row_nc_new['name'].apply(get_features, data = imf_nc , sr = sr_nc, label = 'no-crackle', n_mfcc = n_mfcc)
            row_nc = pd.merge(row_nc, row_nc_new, on=[128,129])
            row_nc['129'] = 'no-crackle'
            row_nc['128'] = filename_nc
            
            
        df = pd.concat([row_c, df], ignore_index=True)
        df = pd.concat([row_nc, df], ignore_index=True)
   
    
    labels = df.pop(129)
    names = df.pop(128)
    df['name'] = names
    df['label'] = labels
    
    
    del df['129']
    del df['128']
    
    df.columns = get_column_names(select_imfs, n_mfcc = n_mfcc)
    
    return df



# If num mfcc = 15 --> label = [121] , filename = [120]
# If num mfcc = 10 --> label = [91], filename = [90]
# If num mfcc = 30 --> label = [211], filename = [210]
def main():
    start = time.time()
    select_imfs = [4,5,6,7,8] 
    n_mfcc = 15
    n_samples = 2000 # 12108
    df =  get_small_dataset(target_rate,select_imfs, n_samples = n_samples, n_imfs = 10, n_sifts = 15, n_mfcc = n_mfcc)
    print(df.shape)
    df.to_csv(f'features/new_features/EEMD_upper_complete_{n_samples}_samples.csv')
    print(f' Processing finished, total time used = {time.time() - start}')

main()
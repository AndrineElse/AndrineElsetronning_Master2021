#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:29:02 2020

@author: andrine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:39:59 2020

@author: andrine
"""

import pandas as pd
import os
import numpy as np
from PyEMD import EMD
import pywt
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
    emd = EMD()
    emd.spline_kind="slinear"
    emd.FIXE = n_sifts
    # Execute EEMD on S
    IMFs = emd.emd(audio, t, max_imf = n_imfs)

    result = []
    for idx in range(n_imfs):
        if idx in select_imfs:
            result.append(IMFs[idx])
    
    return result

def get_n_best_levels(audio, total_level):
    dwtValues = pywt.wavedec(audio, 'db8', level=total_level)
    len_D = len(dwtValues)
    result = []

    for n, c_D in enumerate(dwtValues):
        if n == 0:
            continue
        result.append(c_D)

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
    

def get_features(filename , data, sr , label):
    ft2 = librosa.feature.zero_crossing_rate(data)[0]
    ft3 = librosa.feature.spectral_rolloff(data)[0]
    ft4 = librosa.feature.spectral_centroid(data)[0]
    ft5 = librosa.feature.spectral_contrast(data)[0]
    ft6 = librosa.feature.spectral_bandwidth(data)[0]

    ### Get HOS and simple features 
    ft0_trunc = np.hstack((np.mean(data), np.std(data), skew(data), np.max(data), np.median(data), np.min(data), get_energy(data), get_entropy(data)))
  
    ### Spectral Features 
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
    return pd.Series(np.hstack((ft0_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc, filename, label)))


def get_column_names(select_imfs, n_levels):
    stats = ['mean' , 'std' , 'skew', 'max', 'median', 'min']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']
    cols = []
    for n in select_imfs:
        base_1 = f'IMF_{n}_'
        for i in range(n_levels):
            base = base_1 + f'level_{i}_'
            for stat in stats_HOS:
                cols.append(base + stat)    
               
            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)
               
    cols.append('name')
    cols.append('label')    
    return cols


def get_small_dataset(target_rate, select_imfs, n_samples = 10, n_imfs = 10, n_sifts = 15 , n_levels = 5):
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
        for (idx_1, (imf_c, imf_nc)) in enumerate(zip(IMFs_c , IMFs_nc)):
            dwt_c = get_n_best_levels(imf_c, n_levels)
            dwt_nc = get_n_best_levels(imf_nc, n_levels)
            for (idx_2, (cD_c, cD_nc)) in enumerate(zip(dwt_c , dwt_nc)):
                if ((idx_1 == 0) and (idx_2 == 0)):
                    row_c = pd.DataFrame()
                    row_c['name'] = [filename_c]
                    row_c = row_c['name'].apply(get_features, data = cD_c , sr = sr_c, label = 'crackle')
                    
                    row_nc = pd.DataFrame()
                    row_nc['name'] = [filename_nc]
                    row_nc = row_nc['name'].apply(get_features, data = cD_nc , sr = sr_nc, label = 'no-crackle')
                 
                    continue 
                
                row_c_new = pd.DataFrame()
                row_c_new['name'] = [filename_c]
                row_c_new = row_c_new['name'].apply(get_features, data = cD_c , sr = sr_c, label = 'crackle')
                row_c = pd.merge(row_c, row_c_new ,on=[38,39])
                row_c['39'] = 'crackle'
                row_c['38'] = filename_c
                
                row_nc_new = pd.DataFrame()
                row_nc_new['name'] = [filename_nc]
                row_nc_new = row_nc_new['name'].apply(get_features, data = cD_nc , sr = sr_nc, label = 'no-crackle')
                row_nc = pd.merge(row_nc, row_nc_new, on=[38,39])
                row_nc['39'] = 'no-crackle'
                row_nc['38'] = filename_nc
         
        df = pd.concat([row_c, df], ignore_index=True)
        df = pd.concat([row_nc, df], ignore_index=True)
   
    
    names = df.pop(38)
    labels = df.pop(39)
    df['name'] = names
    df['label'] = labels
    
    
    del df['38']
    del df['39']
    
    df.columns = get_column_names(select_imfs , n_levels)
    
    return df



# If num mfcc = 15 --> label = [121] , filename = [120]
# If num mfcc = 10 --> label = [91], filename = [90]
# If num mfcc = 30 --> label = [211], filename = [210]
def main():
    start = time.time()
    n_samples = 2000 # 5324
    df =  get_small_dataset(target_rate, [1,2,3,4,5] , n_samples = n_samples, n_imfs = 10, n_sifts = 15)
    print(df.shape)
    df.to_csv(f'features/new_features/EMD_DWT_complete_{n_samples}_samples.csv')
    print(f' Processing finished, total time used = {time.time() - start}')

main()
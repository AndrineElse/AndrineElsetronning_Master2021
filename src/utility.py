#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:55:40 2020

@author: andrine
"""
import pandas as pd

import wave
import numpy as np
import scipy.io.wavfile as wf
import scipy.signal
import pywt
from scipy.signal import resample
import os
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix

from sktime.utils.data_io import load_from_tsfile_to_dataframe,load_from_arff_to_dataframe


module_path = os.path.abspath(os.path.join('..'))

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
font = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 10, weight = 1000)
font_small = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 8, weight = 1000)


from sklearn.decomposition import PCA

def get_col_names_EEMD_EMD_DWT(feature_type):

    n_levels = 5
    select_imfs = [1,2,3,4,5]

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []
    for n in select_imfs:
        base_1 = f'IMF_{n}_'
        for i in range(n_levels):
            base = base_1 + f'level_{i}_'
            if feature_type == 'simple':
                for stat in simple_stats:
                    cols.append(base + stat)
            elif feature_type == 'HOS':
                for stat in stats_HOS:
                    cols.append(base + stat)
                for s in others:
                    for stat in stats:
                        cols.append(base + s + '_' + stat)
    return cols

def get_col_names_DWT(feature_type):
    decomp_levels = 10
    n_mfcc = 10

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []

    for n in range(decomp_levels):
        base = f'level_{n}_'
        if feature_type == 'simple':
            for stat in simple_stats:
                cols.append(base + stat)

        elif feature_type == 'HOS':
            for stat in stats_HOS:
                cols.append(base + stat)
            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)

        elif feature_type == 'MFCC':
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
    return cols


def get_col_names_noDecomp(feature_type):
    n_mfcc = 30

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []
    mfcc = []
    if feature_type == 'simple':
        for stat in simple_stats:
            cols.append(stat)

    elif feature_type == 'HOS':
        for stat in stats_HOS:
            cols.append(stat)

        for s in others:
            for stat in stats:
                cols.append(s + '_' + stat)


    elif feature_type == 'MFCC':
        for i in range(1 , n_mfcc + 1):
            mfcc.append(f'mfcc_{i}')
        for stat in stats_HOS:
            cols.append(stat)
        for stat in stats:
            for m in mfcc:
                cols.append(m + '_' + stat)

        for s in others:
            for stat in stats:
                cols.append(s + '_' + stat)
    return cols



def get_col_names_EEMD_EMD(feature_type):
    select_imfs = [1,2,3,4,5]
    n_mfcc = 15

    simple_stats = ['mean' , 'std', 'median', 'min', 'max']
    stats = simple_stats + ['skew']
    stats_HOS = stats + ['energy', 'entropy']
    others = ['zcr', 'spec_roll_off' , 'spec_centroid', 'spec_contrast', 'spec_bandwidth']

    cols = []
    for n in select_imfs:
        base = f'IMF_{n}_'
        mfcc = []
        for i in range(1 , n_mfcc + 1):
            mfcc.append(f'mfcc_{i}')
        if feature_type == 'simple':
            for stat in simple_stats:
                cols.append(base + stat)
        elif feature_type == 'HOS':
            for stat in stats_HOS:
                cols.append(base + stat)
            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)
        elif feature_type == 'MFCC':
            for stat in stats_HOS:
                cols.append(base + stat)
            for stat in stats:
                for m in mfcc:
                    cols.append(base + m + '_' + stat)

            for s in others:
                for stat in stats:
                    cols.append(base + s + '_' + stat)

    return cols


def downsample(audio, sr, sr_new = 8000):
    secs = len(audio)/sr # Number of seconds in signal X
    new_sr = sr_new
    samps = round(secs*new_sr)     # Number of samples to downsample
    new_audio = resample(audio, samps)

    return new_audio


#Will resample all files to the target sample rate and produce a 32bit float array
def read_wav_file(str_filename, target_rate):
    wav = wave.open(str_filename, mode = 'r')
    (sample_rate, data) = extract2FloatArr(wav,str_filename)

    if (sample_rate != target_rate):
        ( _ , data) = resample_2(sample_rate, data, target_rate)

    wav.close()
    return (target_rate, data.astype(np.float32))

def resample_2(current_rate, data, target_rate):
    x_original = np.linspace(0,100,len(data))
    x_resampled = np.linspace(0,100, int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))

# -> (sample_rate, data)
def extract2FloatArr(lp_wave, str_filename):
    (bps, channels) = bitrate_channels(lp_wave)

    if bps in [1,2,4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1:255, 2:32768}
        if bps in [1,2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor)) #clamp to [0.0,1.0]
        return (rate, data)

    elif bps == 3:
        #24bpp wave
        return read24bitwave(lp_wave)

    else:
        raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))

#Note: This function truncates the 24 bit samples to 16 bits of precision
#Reads a wave object returned by the wave.read() method
#Returns the sample rate, as well as the audio in the form of a 32 bit float numpy array
#(sample_rate:float, audio_data: float[])
def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames,-1)
    short_output = np.empty((nFrames, 2), dtype = np.int8)
    short_output[:,:] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))  #return numpy array to save memory via array slicing


def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels()) #bytes per sample
    return (bps, lp_wave.getnchannels())


def denoise_audio(audio):
    coeff = pywt.wavedec(audio, 'db8')
    sigma = np.std(coeff[-1] )
    n= len( audio )
    uthresh = sigma * np.sqrt(2*np.log(n*np.log2(n)))
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='soft' ) for i in coeff[1:] )
    denoised_audio =  pywt.waverec( coeff, 'db8' )
    return denoised_audio

def get_entropy(timeseries):
    timeseries_nz = timeseries[timeseries != 0]
    return - np.sum(((timeseries_nz**2)*np.log(timeseries_nz**2)))

def get_energy(timeseries):  
    N = len(timeseries)
    return np.sum(np.abs(timeseries) ** 2) / N

def remove_unpure_samples(df):
    path = module_path + '/data/crackleWheeze/'
    crackle = os.listdir(path + 'crackle/')
    wheeze = os.listdir(path + 'wheeze/')
    none = os.listdir(path + 'none/')
    both = os.listdir(path + 'both/')

    for idx, row in df.iterrows():
        filename = row['name']
        if (filename in wheeze) or (filename in both):
            df = df.drop(idx, axis = 0)

    df.reset_index(drop=True)
    return df


def get_X_y(decomp_type, feature_type, pure = True,normal = False,
            fs_filter = False,
            fs_auto_encoder = False,
            fs_pca = False, k = 10, module_path = module_path):
    '''
    Decomp type: noDecomp , EMD, EEMD, DWT, EMD_DWT, EEMD_DWT
    Feature type: simple, HOS or MFCC
    k: NB! k has to be 10 or 30 if fs_auto_encoder is True
    '''
    
    dataset = pd.read_csv(module_path + f'/features/{decomp_type}_2000.csv',  sep=',')
    dataset = dataset.drop('Unnamed: 0', axis = 1)

    #if pure:
     #   dataset = remove_unpure_samples(dataset)
    X, y = dataset.iloc[:, :-2], dataset.iloc[:, -1]


    ##### Only extracting features (simple, HOS or MFCC) #############
    cols = []
    if feature_type == 'all':
        cols = X.columns
    elif decomp_type in ['EMD_DWT', 'EEMD_DWT']:
        if feature_type == 'MFCC':
            print('Action is not valid. EEMD_DWT and EMD_DWT does not have MFCC features')
            cols = X.columns
        else:
            cols = get_col_names_EEMD_EMD_DWT(feature_type)
    elif decomp_type in ['EMD', 'EEMD']:
        cols = get_col_names_EEMD_EMD(feature_type)
    elif decomp_type == 'noDecomp':
        cols = get_col_names_noDecomp(feature_type)
    elif decomp_type == 'DWT':
        cols = get_col_names_DWT(feature_type)


    X, y = dataset[cols] , dataset.iloc[:, -1]

    ##### Normalizing Data #############
    if normal:
        x = X.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        X = pd.DataFrame(x_scaled)

    ##### Performing feature selection #############
    if fs_filter:
        X = SelectKBest(chi2, k=k).fit_transform(X, y)

    if fs_auto_encoder:
        if k == 10:
            cols = [116,  91,  62,  68,  46,  53, 142, 139, 145, 160]
            X, y = X[cols] , dataset.iloc[:, -1]
        elif k == 30:
            cols = [ 27,  79, 210, 136,  82, 203, 184,  87, 141,  60,  95, 103, 198,
            3,  10, 205,  95,  35, 101, 109,   3, 170, 193,  59, 164,  72,
            171,  89, 200,  89]
            X, y = X[cols] , dataset.iloc[:, -1]
        else:
            print('Error: When using Autoencoder for feature selection k has to be either 10 or 30')

    if fs_pca:
        pca = PCA(n_components=k)
        X = pca.fit_transform(X)


    return X, y



def get_t(y, sr):
    n = len(y)
    t = np.linspace(0, 1/ sr, n)
    return t

def convert_arff_to_ts(filepath, filename):
    X, y = load_from_arff_to_dataframe(filepath + '/' + filename)
    new_filename = filename[:-4] + 'ts'
    print(new_filename)
    dataset = filename.split('_')[0]
    print(dataset)
    
    labels = np.unique(y).astype(str)
    label_str = ''
    for label in labels:
        label_str = label_str + label + ' '
    print(label_str)
    w = open(filepath + '/' + new_filename, 'w+')
    
    w.write(f'@problemName {dataset} \n')
    w.write('@timeStamps false \n')
    w.write('@univariate true \n')
    w.write(f'@classLabel true {label_str} \n')
    w.write('@data \n')
    for (idx, row) in X.iterrows():
        new_row = (list(row)[0]).tolist()
        new_row = str(new_row)[1:-1].replace(' ', '') + ':' + y[idx] + '\n'
        w.write(new_row)
        
        
def write_to_ts(filepath, X, y):
    
    w = open(filepath, 'w+')
    
    w.write('@problemName LungSoundsMiniROCKET \n')
    w.write('@timeStamps false \n')
    w.write('@missing false \n')
    w.write('@univariate true \n')
    w.write('@equalLength true \n')
    w.write(f'@seriesLength {str(len(X.columns))} \n')
    w.write('@classLabel true no_crackle crackle\n')
    w.write('@data \n')
    for (idx, row) in X.iterrows():
        new_row = str((list(row)))[1:-1].replace(' ', '') + ':' + y[idx] + '\n'
        w.write(new_row)

def plot_cm(y_true, y_pred, module_path = module_path, color_index = None, class_names = ['no-crackle', 'crackle' ], hex_color_str = None):
    cm = confusion_matrix(y_true, y_pred)
    colors = ['#F94144', '#F3722C', '#F8961E', '#F9844A', '#F9C74F', '#90BE6D', '#43AA8B', '#4D908E', '#577590', '#277DA1']
    #colors = ['#F9414466', '#90BE6D66', '#57759066','#F3722C66', '#F8961E66',
    #         '#F9844A66', '#F9C74F66', '#43AA8B66', '#4D908E66', '#277DA166']
    font = FontProperties(fname = module_path + '/src/visualization/CharterRegular.ttf', size = 10, weight = 1000)
    
    
    if hex_color_str:
        colors_2 = ['#FFFFFF', hex_color_str]
    elif color_index:
        colors_2 = ['#FFFFFF', colors[color_index]]
    else: 
        colors_2 = ['#FFFFFF', colors[0]]
    cmap_name = 'my colormap'
    font_small = FontProperties(fname =  module_path + '/src/visualization/CharterRegular.ttf', size = 6, weight = 1000)

    cm_map = LinearSegmentedColormap.from_list(cmap_name, colors_2)



    f, ax = plt.subplots(1,1) # 1 x 1 array , can also be any other size
    f.set_size_inches(3, 3)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True,
                fmt='.2%', cmap=cm_map, xticklabels=class_names,yticklabels=class_names )
    cbar = ax.collections[0].colorbar
    for label in ax.get_yticklabels() :
        label.set_fontproperties(font_small)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(font_small)
    ax.set_ylabel('True Label', fontproperties = font)
    ax.set_xlabel('Predicted Label', fontproperties = font)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)

    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Text):
            child.set_fontproperties(font)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontproperties(font_small)
        
    return f,ax

#%%
from difflib import Match
import enum
import gc
from genericpath import exists
from operator import le
import os
import ast
import re
from time import time
from unittest import result
import numpy as np
import scipy
from scipy.fftpack import idct
import torch
import torch.fft
from torch import nn
from torch.autograd import Variable
import torchaudio
torchaudio.set_audio_backend("sox_io")
import glob, os
from os.path import join,isfile
from modules.TDNNLayer_original import TDNN as TDNNLayer
from modules.StatisticsPoolingLayers import StatisticsPooling
from extractor import MFCCVadCMVNPadBatch as mfcc_extractor
from classifier import TDNNetwork
from os import listdir, write
# from tqdm import tqdm
from torch.nn import functional as F
import torchaudio.functional as Fa
from torchaudio_local import resample_waveform
import pickle as pkl
import argparse
#import librosa
import math
import torchaudio.transforms as T
from IPython import embed
from joblib import Parallel, delayed
import itertools
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle


#%%

# @Yash: these are 2 of the control parameters; (1) the model and (2) the src id:
# model = "BrainSpeech"
model = "tdnn_state"
src_id = 2

device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# @Yash: I fixed the device to be cpu to run the code on macbook


if model == "tdnn_state":
    model_type = 'tdnn'
    model_class = 'TDNNetwork'
    attentive = True
    input_dim = 30
    output_dim = 250
    node_num = [512, 512, 512, 512, 1500, 3000, 512, 512]
    context = [[-2, 2], [-2, 0, 2], [-3, 0, 3], [0], [0]]
    full_context = [True, False, False, True, True]

    # Model and threshold
    classifier = TDNNetwork(input_dim, output_dim, node_num, context, full_context, device=device)
    classifier = torch.load('pretrained_models/tdnn_state_dict.pth', map_location=device)
    classifier.eval()

elif model == 'BrainSpeech':
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",  run_opts={"device":device})

def get_f0_tube(L, r, correct=True):
    # open_ended tube:
    n = 1
    T = 24  ## Temp in Celsius
    v = 20.05 * math.sqrt(T+273.15) #343   # sound speed in air in m/s
    f0 = n*v/2/(L+0.8*2*r)
    if correct:
        f0 = f0*(1+(1/100))
    return f0

def get_Q_tube(r, f0):
    T = 24  ## Temp in Celsius
    v = 20.05 * math.sqrt(T+273.15)
    A = math.pi * r**2         # radius in meters
    d_rad = (2*math.pi*A*f0**2)/v**2  # 5.34*10**-5*A*f0**2
    d_wall = 5.71*10**-3*(A*f0)**-0.5
    Q = round(1/(d_rad+d_wall))
    return Q


def apply_bandpass_filter(waveform, g, fs, f, Q, const_skirt_gain):
    filtered_wav = g*Fa.bandpass_biquad(waveform, fs, f, Q, const_skirt_gain=const_skirt_gain)
    return filtered_wav 


def apply_band_filter(waveform, g, fs, f, Q):
    filtered_wav = g*Fa.band_biquad(waveform, fs, f, Q, noise=False) 
    return filtered_wav 

def apply_resonant_filter(waveform, fs, f0, r, idx, filter, const_skirt_gain=False, save_audio=True):
    num_harmonics = round(0.8*fs/2/f0)
    g = 1
    Q = get_Q_tube(r, f0)
    f0_list = [f0*i for i in range(1,num_harmonics)]
    if filter == 'bandpass':
        filtered_wavs = Parallel(n_jobs=4, prefer="threads")(delayed(apply_bandpass_filter)(waveform, g, fs, fi, Q, const_skirt_gain=const_skirt_gain) for fi in f0_list)
        z_sum = sum(filtered_wavs)
        del filtered_wavs
    return z_sum


def apply_resonant_filter_Q(waveform, fs, f0, r, idx, filter, const_skirt_gain=False, save_audio=True):
    num_harmonics = round(0.8*fs/2/f0)
    print('Num of harmonics =', num_harmonics)
    g = 1
    Q = get_Q_tube(r, f0)
    f0_list = [f0*i for i in range(1,num_harmonics)]
    Q_list = [round(Q/math.pow(i, 1/4)) for i in range(1,num_harmonics)]

    if filter == 'bandpass':
        filtered_wavs = Parallel(n_jobs=4, prefer='threads')(delayed(apply_bandpass_filter)(waveform, g, fs, fi, Qi, const_skirt_gain=const_skirt_gain) for fi, Qi in zip(f0_list, Q_list))
        z_sum = sum(filtered_wavs)
        del filtered_wavs
    return z_sum


def PredictClass(waveform, classifier, n):
    if model == "BrainSpeech":
        ids_list = []
        batch_size = 100
        for i in range(0, len(waveform), batch_size):
            # print(i, i+100)
            All_probs, best_score, best_index, id_text = classifier.classify_batch(waveform[i:i+batch_size,:])
            probs, ids = All_probs.sort(descending=True)
            ids_list.append(ids[:,:n].to("cpu"))
        ids_tensor = torch.vstack(ids_list)  
        return ids_tensor
    else:
        # forward pass
        waveform = waveform.unsqueeze(dim=1) #.to(device)
        extractor = mfcc_extractor()
        # Feed to the classifier
        # x_mfccs_vad, x_mfccs = extractor(X.transpose(1,2), before_vad=True)
        x_mfccs_vad, x_mfccs = extractor(waveform, before_vad=True)

        clean_logits = classifier.forward(x_mfccs_vad)
        clean_probs  = F.log_softmax(clean_logits,dim=-1).data
        clean_probs_sorted, clean_idx_sorted = clean_probs.sort(1, True)
        clean_class      = clean_idx_sorted[:,0]
        clean_class_prob = clean_probs_sorted[:,0]
        return clean_idx_sorted[:,0:n]


def MatchValues(a, b):
    matches = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            matches += 1
    return matches, len(a) - matches
print(model)


#%%
# @Yash: the third control parameter, the tube length
# Tube parameters:
d_dict = {12: 3.45, 16: 3.45, 24.125:4, 34.25:5.2, 39.125: 3.45, 47.375:5.2, 90.8125: 2, 60.625: 5.2, 38.9375 : 4} #, 16: 4}
L_in = 60.625
L= L_in *2.54/100 # in meter
r = d_dict[L_in]/2/100  # in meter
print('L =', L_in, 'in -- ', L*100, 'cm', ', d =', r*2*100, 'cm')
f0 = get_f0_tube(L, r, correct=True)
Q = get_Q_tube(r, f0)
print("f0 = ", f0, ", Q = ", Q)

# Read the natural recordings:
# @Yash: this file is inside ASI and the userstudy is inside Dataset folder. If you have a different tree, adjust the 2 glob directory read line
wavs = []
for file in glob.glob('../Dataset/Box_UserStudy/natural/{}/*.wav'.format(src_id)):
    waveform, fs = torchaudio.load(file)
    wavs.append(waveform)
waveform=torch.vstack(wavs)
if model == 'tdnn_state':
    waveform = resample_waveform(waveform, fs, 8000, lowpass_filter_width=6, device=waveform.device)
    fs=8000
org_ids = PredictClass(waveform, classifier, 1).to("cpu")

# Apply Filter:
waveform_filter = apply_resonant_filter(waveform, fs, f0, r, 0, 'bandpass', const_skirt_gain=False, save_audio=False)
filter_ids = PredictClass(waveform_filter, classifier, 1).to("cpu")
waveform_filter_Q = apply_resonant_filter_Q(waveform, fs, f0, r, 0, 'bandpass', const_skirt_gain=False, save_audio=False)
filter_ids_Q = PredictClass(waveform_filter_Q, classifier, 1).to("cpu")

# Read tube recording:
wavs = []
for file in glob.glob('../Dataset/Box_UserStudy/{}_{}/{}/*.wav'.format(L_in, d_dict[L_in], src_id)):
    waveform_tube, fs_tube = torchaudio.load(file)
    wavs.append(waveform_tube)
waveform_tube=torch.vstack(wavs)
if model == 'tdnn_state':
    waveform_tube = resample_waveform(waveform_tube, fs_tube, 8000, lowpass_filter_width=6, device=waveform_tube.device)
    fs_tube=8000
tube_ids = PredictClass(waveform_tube, classifier, 1).to("cpu")

#%%

# print('org_ids unique hits = ', len(org_ids.unique()))
# print('Tube errors', MatchValues(org_ids, tube_ids))
# print('filter-tube match', MatchValues(filter_ids, tube_ids))
# print('filterQ-tube match', MatchValues(filter_ids_Q, tube_ids))

# # for i, id in enumerate(org_ids):
# #     print(id, tube_ids[i], filter_ids_Q[i])

def get_false_pred(true, pred):
    false_pred = [id for id in pred.unique().tolist() if id not in true.unique().tolist()]
    return false_pred

false_ids_filter = get_false_pred(org_ids, filter_ids) 
false_ids_filter_Q = get_false_pred(org_ids, filter_ids_Q)
false_ids_tube = get_false_pred(org_ids, tube_ids)


matched_errors_Q = np.intersect1d(false_ids_filter_Q, false_ids_tube) 
matched_ids_Q = np.intersect1d(filter_ids_Q,tube_ids)
# print('Tube errors count:', len(false_ids_tube), '=>', false_ids_tube)
# print('.'*50)
# print('Filter_Q errors count:', len(false_ids_filter_Q), '=>', false_ids_filter_Q)
# print('FilterQ unique predictions:', len(filter_ids_Q.unique()), '=>', filter_ids_Q.unique())
# print('Tube-filterQ match (removing org ids):',  len(matched_errors_Q), '=>', matched_errors_Q)
# print('Tube-filterQ match:', len(matched_ids_Q), '=>' , matched_ids_Q)

# print('.'*50)
matched_errors = np.intersect1d(false_ids_filter, false_ids_tube)
matched_ids = np.intersect1d(filter_ids,tube_ids)
# print('Filter errors count:', len(false_ids_filter), '=>', false_ids_filter)
# print('Filter unique predictions:', len(filter_ids.unique()), '=>', filter_ids.unique())
# print('Tube-filter match (removing org ids):', len(matched_errors), '=>', matched_errors)
# print('Tube-filter match:', len(matched_ids), '=>' , matched_ids)


# %%

def find_mismatch(all_ids, false_ids , match_ids):
    all_ids = np.array(all_ids)
    false_ids = np.array(false_ids)
    match_ids = np.array(match_ids)
    indices = []
    mismatch_ids = [id for id in false_ids if id not in match_ids]
    for i, id in enumerate(all_ids):
        if id in mismatch_ids:
            print(id, all_ids[i])
            indices.append(i)
    return indices, mismatch_ids

indices, mismatch_ids = find_mismatch(filter_ids_Q, false_ids_filter_Q, matched_errors_Q)
# indices, mismatch_ids = find_mismatch(filter_ids_Q, filter_ids_Q.unique(), matched_ids_Q)
print(indices)

# @Yash: 
# 1. "mismatched ids" are the ids we had from the filter but was not predicted by the tube.
# so, we need to repeat the tube recordings and see if we reach these ids by repitition. 
# 2. "indices": are the utterances numbers where the filter resulted in these mismatched ids 
#  so, we should ask the participant to read these utterances using the tube. However, the participant can also read any other utterance to try to reach the mismatched id.

#%%
import soundfile as sf
import sounddevice as sd
from time import sleep

go, fs = sf.read('../Dataset/go.wav')
stop, fs = sf.read('../Dataset/stop.wav')
n_repeat = 10 ## @Yash: select here how many times you want the user to repeat the recording before giving up

def rec_only(idx):
    duration = 4
    fs = 16000
    sd.play(go, samplerate=fs, blocking=True)
    print('!!!!! Starting Sample {} !!!!'.format(idx))
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, blocking=True)
    sd.play(stop, samplerate=fs, blocking=True)
    print("recording Done!")
    print(myrecording.shape)
    myrecording1 = myrecording[:, 0]  #inside_box
    myrecording2 = myrecording[:,1]  #outside box
    
    return myrecording1, myrecording2, fs

# @Yash: I loop over the utterances that the user should say to match the filter model. Instead, you can remove this for loop and do it manually to better control when to start and stop recordings
for utt in indices:      
    print('Utterance number:', utt)
    sleep(5)
    for n in range(n_repeat):
        print('Read utterance number:', utt)
        in_box, out_box, fs = rec_only(utt)
        waveform_tube = torch.tensor(in_box).unsqueeze(0)   
        if model == 'tdnn_state':
            waveform_tube = resample_waveform(waveform_tube, fs, 8000, lowpass_filter_width=6, device=waveform_tube.device)
            fs=8000
        tube_id = PredictClass(waveform_tube, classifier, 1).to("cpu").item()
        print(tube_id)
        if tube_id in mismatch_ids:
            print(f'prediction match is found in the {n} trial')
            out_dir = '../Dataset/Box_UserStudy/{}_{}/{}/'.format(L_in, d_dict[L_in], src_id)
            sf.write(out_dir+ utt+f'_repeat_{n}.wav', in_box, fs)
            # @Yash: I only save the recording if it resulted in a match. You can change this to save all recordings if needed
            mismatch_ids.remove(tube_id)
            break
        else:
            print('No match was found!')
        sleep(2)
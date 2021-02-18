"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 06 - Create Enchanced Audio for Testing and Evaluate
"""

##################################################
# Imports
##################################################
import s00_tools as tls
from pystoi.stoi import stoi
from pysepm import cepstrum_distance, wss
#from pysepm import pesq
#from pesq import pesq
#from pypesq import pesq
import sounddevice as sd
import os
import numpy as np
from texttable import Texttable

##################################################
# Main
##################################################
# Import Spects
spect_NoisySpeech_mag = np.load('Dataset_My_Spects\\Testing_NoisySpeech_spect_arrary.npy')
spect_NoisySpeech_phase = np.load('Dataset_My_Spects\\Testing_NoisySpeech_phase_array.npy')
spect_CleanSpeech_mag = np.load('Dataset_My_Spects\\Testing_CleanSpeech_spect_arrary.npy')
spect_CleanSpeech_phase = np.load('Dataset_My_Spects\\Testing_CleanSpeech_phase_array.npy')
spect_EnhancedSpeech_mag = np.load('Dataset_My_Spects\\Testing_EnhancedSpeech_spect_arrary.npy')

# List Names of Test Files
path_Testing_NoisySpeech = "Dataset_My_Wavs\\Testing_NoisySpeech"
filenames_Testing_NoisySpeech = [os.path.join(path_Testing_NoisySpeech, f) for f in os.listdir(path_Testing_NoisySpeech) if f.endswith(".wav")]
filenames_Testing_NoisySpeech.sort()

# Create Audio
wavs_NoisySpeech = np.zeros((32385, spect_NoisySpeech_mag.shape[0]), np.float32)
wavs_CleanSpeech = np.zeros((32385, spect_NoisySpeech_mag.shape[0]), np.float32)
wavs_EnhancedSpeech = np.zeros((32385, spect_NoisySpeech_mag.shape[0]), np.float32)

for i in range(spect_NoisySpeech_mag.shape[0]):
    wavs_NoisySpeech[:, i] = tls.spect_to_audio(spect_NoisySpeech_mag[i,:,:], spect_NoisySpeech_phase[i,:,:])
    wavs_CleanSpeech[:, i] = tls.spect_to_audio(spect_CleanSpeech_mag[i,:,:], spect_CleanSpeech_phase[i,:,:])
    wavs_EnhancedSpeech[:, i] = tls.spect_to_audio(spect_EnhancedSpeech_mag[i,:,:], spect_NoisySpeech_phase[i,:,:])
    
"""
# Play a sample    
sample_num = 150
sf = 16000

sd.play(wavs_NoisySpeech[:, sample_num], sf)
status = sd.wait()

sd.play(wavs_CleanSpeech[:, sample_num], sf)
status = sd.wait()

sd.play(wavs_EnhancedSpeech[:, sample_num], sf)
status = sd.wait()
"""

 
# Calculate Quality & Intelligibility Measures
# Quality - Cepstrum Distance (CD), Weighted Spectral Slope (WSS)
CD_Overall_prev = []
CD_Overall_aftr = []
CD_SNR_0dB_prev = []
CD_SNR_0dB_aftr = []
CD_SNR_5dB_prev = []
CD_SNR_5dB_aftr = []
CD_SNR_10dB_prev = []
CD_SNR_10dB_aftr = []
CD_SNR_15dB_prev = []
CD_SNR_15dB_aftr = []
CD_SNR_20dB_prev = []
CD_SNR_20dB_aftr = []

WSS_Overall_prev = []
WSS_Overall_aftr = []
WSS_SNR_0dB_prev = []
WSS_SNR_0dB_aftr = []
WSS_SNR_5dB_prev = []
WSS_SNR_5dB_aftr = []
WSS_SNR_10dB_prev = []
WSS_SNR_10dB_aftr = []
WSS_SNR_15dB_prev = []
WSS_SNR_15dB_aftr = []
WSS_SNR_20dB_prev = []
WSS_SNR_20dB_aftr = []

# Intelligibility - Short-time objective intelligibility (STOI)
STOI_Overall_prev = []
STOI_Overall_aftr = []
STOI_SNR_0dB_prev = []
STOI_SNR_0dB_aftr = []
STOI_SNR_5dB_prev = []
STOI_SNR_5dB_aftr = []
STOI_SNR_10dB_prev = []
STOI_SNR_10dB_aftr = []
STOI_SNR_15dB_prev = []
STOI_SNR_15dB_aftr = []
STOI_SNR_20dB_prev = []
STOI_SNR_20dB_aftr = []

for idx in range (len(filenames_Testing_NoisySpeech)): 
    # Calculate CD & STOI for every sample
    CD_prev = cepstrum_distance(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000)
    CD_aftr = cepstrum_distance(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000)
    
    WSS_prev = wss(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000)
    WSS_aftr = wss(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000)
    
    STOI_prev = stoi(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000, extended=True)
    STOI_aftr = stoi(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000, extended=True)
    
    # Add to Overall
    CD_Overall_prev.append(CD_prev)
    CD_Overall_aftr.append(CD_aftr)
    
    WSS_Overall_prev.append(WSS_prev)
    WSS_Overall_aftr.append(WSS_aftr)
    
    STOI_Overall_prev.append(STOI_prev)
    STOI_Overall_aftr.append(STOI_aftr)  
    
    # Add to scesific SNR category
    if ('SNRdb_0.0' in filenames_Testing_NoisySpeech[idx]):
        CD_SNR_0dB_prev.append(CD_prev)
        CD_SNR_0dB_aftr.append(CD_aftr)
        WSS_SNR_0dB_prev.append(WSS_prev)
        WSS_SNR_0dB_aftr.append(WSS_aftr)
        STOI_SNR_0dB_prev.append(STOI_prev)
        STOI_SNR_0dB_aftr.append(STOI_aftr)
    elif ('SNRdb_5.0' in filenames_Testing_NoisySpeech[idx]):
        CD_SNR_5dB_prev.append(CD_prev)
        CD_SNR_5dB_aftr.append(CD_aftr)
        WSS_SNR_5dB_prev.append(WSS_prev)
        WSS_SNR_5dB_aftr.append(WSS_aftr)
        STOI_SNR_5dB_prev.append(STOI_prev)
        STOI_SNR_5dB_aftr.append(STOI_aftr)       
    elif ('SNRdb_10.0' in filenames_Testing_NoisySpeech[idx]):
        CD_SNR_10dB_prev.append(CD_prev)
        CD_SNR_10dB_aftr.append(CD_aftr)
        WSS_SNR_10dB_prev.append(WSS_prev)
        WSS_SNR_10dB_aftr.append(WSS_aftr)
        STOI_SNR_10dB_prev.append(STOI_prev)
        STOI_SNR_10dB_aftr.append(STOI_aftr)
    elif ('SNRdb_15.0' in filenames_Testing_NoisySpeech[idx]):
        CD_SNR_15dB_prev.append(CD_prev)
        CD_SNR_15dB_aftr.append(CD_aftr)
        WSS_SNR_15dB_prev.append(WSS_prev)
        WSS_SNR_15dB_aftr.append(WSS_aftr)
        STOI_SNR_15dB_prev.append(STOI_prev)
        STOI_SNR_15dB_aftr.append(STOI_aftr)
    elif ('SNRdb_20.0' in filenames_Testing_NoisySpeech[idx]):
        CD_SNR_20dB_prev.append(CD_prev)
        CD_SNR_20dB_aftr.append(CD_aftr)
        WSS_SNR_20dB_prev.append(WSS_prev)
        WSS_SNR_20dB_aftr.append(WSS_aftr)
        STOI_SNR_20dB_prev.append(STOI_prev)
        STOI_SNR_20dB_aftr.append(STOI_aftr)
                
        
# Compute & Print Averages
CD_t = Texttable()
CD_t.add_rows([
    ['SNR_Level', 'CD_Noisy_vs_Clean', 'CD_Enchanced_vs_Clean'],
    ['SNR_0dB', sum(CD_SNR_0dB_prev)/len(CD_SNR_0dB_prev), sum(CD_SNR_0dB_aftr)/len(CD_SNR_0dB_aftr)],
    ['SNR_5dB', sum(CD_SNR_5dB_prev)/len(CD_SNR_5dB_prev), sum(CD_SNR_5dB_aftr)/len(CD_SNR_5dB_aftr)],
    ['SNR_10dB', sum(CD_SNR_10dB_prev)/len(CD_SNR_10dB_prev), sum(CD_SNR_10dB_aftr)/len(CD_SNR_10dB_aftr)],
    ['SNR_15dB', sum(CD_SNR_15dB_prev)/len(CD_SNR_15dB_prev), sum(CD_SNR_15dB_aftr)/len(CD_SNR_15dB_aftr)],
    ['SNR_20dB', sum(CD_SNR_20dB_prev)/len(CD_SNR_20dB_prev), sum(CD_SNR_20dB_aftr)/len(CD_SNR_20dB_aftr)],
    ['Overall', sum(CD_Overall_prev)/len(CD_Overall_prev), sum(CD_Overall_aftr)/len(CD_Overall_aftr)],
    ])
print(CD_t.draw())

WSS_t = Texttable()
WSS_t.add_rows([
    ['SNR_Level', 'WSS_Noisy_vs_Clean', 'WSS_Enchanced_vs_Clean'],
    ['SNR_0dB', sum(WSS_SNR_0dB_prev)/len(WSS_SNR_0dB_prev), sum(WSS_SNR_0dB_aftr)/len(WSS_SNR_0dB_aftr)],
    ['SNR_5dB', sum(WSS_SNR_5dB_prev)/len(WSS_SNR_5dB_prev), sum(WSS_SNR_5dB_aftr)/len(WSS_SNR_5dB_aftr)],
    ['SNR_10dB', sum(WSS_SNR_10dB_prev)/len(WSS_SNR_10dB_prev), sum(WSS_SNR_10dB_aftr)/len(WSS_SNR_10dB_aftr)],
    ['SNR_15dB', sum(WSS_SNR_15dB_prev)/len(WSS_SNR_15dB_prev), sum(WSS_SNR_15dB_aftr)/len(WSS_SNR_15dB_aftr)],
    ['SNR_20dB', sum(WSS_SNR_20dB_prev)/len(WSS_SNR_20dB_prev), sum(WSS_SNR_20dB_aftr)/len(WSS_SNR_20dB_aftr)],
    ['Overall', sum(WSS_Overall_prev)/len(WSS_Overall_prev), sum(WSS_Overall_aftr)/len(WSS_Overall_aftr)],
    ])
print(WSS_t.draw())

STOI_t = Texttable()
STOI_t.add_rows([
    ['SNR_Level', 'STOI_Noisy_vs_Clean', 'STOI_Enchanced_vs_Clean'],
    ['SNR_0dB', sum(STOI_SNR_0dB_prev)/len(STOI_SNR_0dB_prev), sum(STOI_SNR_0dB_aftr)/len(STOI_SNR_0dB_aftr)],
    ['SNR_5dB', sum(STOI_SNR_5dB_prev)/len(STOI_SNR_5dB_prev), sum(STOI_SNR_5dB_aftr)/len(STOI_SNR_5dB_aftr)],
    ['SNR_10dB', sum(STOI_SNR_10dB_prev)/len(STOI_SNR_10dB_prev), sum(STOI_SNR_10dB_aftr)/len(STOI_SNR_10dB_aftr)],
    ['SNR_15dB', sum(STOI_SNR_15dB_prev)/len(STOI_SNR_15dB_prev), sum(STOI_SNR_15dB_aftr)/len(STOI_SNR_15dB_aftr)],
    ['SNR_20dB', sum(STOI_SNR_20dB_prev)/len(STOI_SNR_20dB_prev), sum(STOI_SNR_20dB_aftr)/len(STOI_SNR_20dB_aftr)],
    ['Overall', sum(STOI_Overall_prev)/len(STOI_Overall_prev), sum(STOI_Overall_aftr)/len(STOI_Overall_aftr)],
    ])
print(STOI_t.draw())

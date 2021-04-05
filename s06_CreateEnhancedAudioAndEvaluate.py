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
from pysepm import fwSNRseg, SNRseg
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
# Quality - Segmental SNR, Frequency-Weighted Segmental SNR
fwSNRseg_Overall_prev = []
fwSNRseg_Overall_aftr = []
fwSNRseg_SNR_0dB_prev = []
fwSNRseg_SNR_0dB_aftr = []
fwSNRseg_SNR_5dB_prev = []
fwSNRseg_SNR_5dB_aftr = []
fwSNRseg_SNR_10dB_prev = []
fwSNRseg_SNR_10dB_aftr = []
fwSNRseg_SNR_15dB_prev = []
fwSNRseg_SNR_15dB_aftr = []
fwSNRseg_SNR_20dB_prev = []
fwSNRseg_SNR_20dB_aftr = []

SNRseg_Overall_prev = []
SNRseg_Overall_aftr = []
SNRseg_SNR_0dB_prev = []
SNRseg_SNR_0dB_aftr = []
SNRseg_SNR_5dB_prev = []
SNRseg_SNR_5dB_aftr = []
SNRseg_SNR_10dB_prev = []
SNRseg_SNR_10dB_aftr = []
SNRseg_SNR_15dB_prev = []
SNRseg_SNR_15dB_aftr = []
SNRseg_SNR_20dB_prev = []
SNRseg_SNR_20dB_aftr = []

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
    # Calculate fwSNRseg & STOI for every sample
    fwSNRseg_prev = fwSNRseg(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000)
    fwSNRseg_aftr = fwSNRseg(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000)
    
    SNRseg_prev = SNRseg(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000)
    SNRseg_aftr = SNRseg(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000)
    
    STOI_prev = stoi(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000, extended=True)
    STOI_aftr = stoi(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000, extended=True)
    
    # Add to Overall
    fwSNRseg_Overall_prev.append(fwSNRseg_prev)
    fwSNRseg_Overall_aftr.append(fwSNRseg_aftr)
    
    SNRseg_Overall_prev.append(SNRseg_prev)
    SNRseg_Overall_aftr.append(SNRseg_aftr)
    
    STOI_Overall_prev.append(STOI_prev)
    STOI_Overall_aftr.append(STOI_aftr)  
    
    # Add to scesific SNR category
    if ('SNRdb_0.0' in filenames_Testing_NoisySpeech[idx]):
        fwSNRseg_SNR_0dB_prev.append(fwSNRseg_prev)
        fwSNRseg_SNR_0dB_aftr.append(fwSNRseg_aftr)
        SNRseg_SNR_0dB_prev.append(SNRseg_prev)
        SNRseg_SNR_0dB_aftr.append(SNRseg_aftr)
        STOI_SNR_0dB_prev.append(STOI_prev)
        STOI_SNR_0dB_aftr.append(STOI_aftr)
    elif ('SNRdb_5.0' in filenames_Testing_NoisySpeech[idx]):
        fwSNRseg_SNR_5dB_prev.append(fwSNRseg_prev)
        fwSNRseg_SNR_5dB_aftr.append(fwSNRseg_aftr)
        SNRseg_SNR_5dB_prev.append(SNRseg_prev)
        SNRseg_SNR_5dB_aftr.append(SNRseg_aftr)
        STOI_SNR_5dB_prev.append(STOI_prev)
        STOI_SNR_5dB_aftr.append(STOI_aftr)       
    elif ('SNRdb_10.0' in filenames_Testing_NoisySpeech[idx]):
        fwSNRseg_SNR_10dB_prev.append(fwSNRseg_prev)
        fwSNRseg_SNR_10dB_aftr.append(fwSNRseg_aftr)
        SNRseg_SNR_10dB_prev.append(SNRseg_prev)
        SNRseg_SNR_10dB_aftr.append(SNRseg_aftr)
        STOI_SNR_10dB_prev.append(STOI_prev)
        STOI_SNR_10dB_aftr.append(STOI_aftr)
    elif ('SNRdb_15.0' in filenames_Testing_NoisySpeech[idx]):
        fwSNRseg_SNR_15dB_prev.append(fwSNRseg_prev)
        fwSNRseg_SNR_15dB_aftr.append(fwSNRseg_aftr)
        SNRseg_SNR_15dB_prev.append(SNRseg_prev)
        SNRseg_SNR_15dB_aftr.append(SNRseg_aftr)
        STOI_SNR_15dB_prev.append(STOI_prev)
        STOI_SNR_15dB_aftr.append(STOI_aftr)
    elif ('SNRdb_20.0' in filenames_Testing_NoisySpeech[idx]):
        fwSNRseg_SNR_20dB_prev.append(fwSNRseg_prev)
        fwSNRseg_SNR_20dB_aftr.append(fwSNRseg_aftr)
        SNRseg_SNR_20dB_prev.append(SNRseg_prev)
        SNRseg_SNR_20dB_aftr.append(SNRseg_aftr)
        STOI_SNR_20dB_prev.append(STOI_prev)
        STOI_SNR_20dB_aftr.append(STOI_aftr)
                
        
# Compute & Print Averages
fwSNRseg_t = Texttable()
fwSNRseg_t.add_rows([
    ['SNR_Level', 'fwSNRseg_Noisy_vs_Clean', 'fwSNRseg_Enchanced_vs_Clean'],
    ['SNR_0dB', sum(fwSNRseg_SNR_0dB_prev)/len(fwSNRseg_SNR_0dB_prev), sum(fwSNRseg_SNR_0dB_aftr)/len(fwSNRseg_SNR_0dB_aftr)],
    ['SNR_5dB', sum(fwSNRseg_SNR_5dB_prev)/len(fwSNRseg_SNR_5dB_prev), sum(fwSNRseg_SNR_5dB_aftr)/len(fwSNRseg_SNR_5dB_aftr)],
    ['SNR_10dB', sum(fwSNRseg_SNR_10dB_prev)/len(fwSNRseg_SNR_10dB_prev), sum(fwSNRseg_SNR_10dB_aftr)/len(fwSNRseg_SNR_10dB_aftr)],
    ['SNR_15dB', sum(fwSNRseg_SNR_15dB_prev)/len(fwSNRseg_SNR_15dB_prev), sum(fwSNRseg_SNR_15dB_aftr)/len(fwSNRseg_SNR_15dB_aftr)],
    ['SNR_20dB', sum(fwSNRseg_SNR_20dB_prev)/len(fwSNRseg_SNR_20dB_prev), sum(fwSNRseg_SNR_20dB_aftr)/len(fwSNRseg_SNR_20dB_aftr)],
    ['Overall', sum(fwSNRseg_Overall_prev)/len(fwSNRseg_Overall_prev), sum(fwSNRseg_Overall_aftr)/len(fwSNRseg_Overall_aftr)],
    ])
print(fwSNRseg_t.draw())

SNRseg_t = Texttable()
SNRseg_t.add_rows([
    ['SNR_Level', 'SNRseg_Noisy_vs_Clean', 'SNRseg_Enchanced_vs_Clean'],
    ['SNR_0dB', sum(SNRseg_SNR_0dB_prev)/len(SNRseg_SNR_0dB_prev), sum(SNRseg_SNR_0dB_aftr)/len(SNRseg_SNR_0dB_aftr)],
    ['SNR_5dB', sum(SNRseg_SNR_5dB_prev)/len(SNRseg_SNR_5dB_prev), sum(SNRseg_SNR_5dB_aftr)/len(SNRseg_SNR_5dB_aftr)],
    ['SNR_10dB', sum(SNRseg_SNR_10dB_prev)/len(SNRseg_SNR_10dB_prev), sum(SNRseg_SNR_10dB_aftr)/len(SNRseg_SNR_10dB_aftr)],
    ['SNR_15dB', sum(SNRseg_SNR_15dB_prev)/len(SNRseg_SNR_15dB_prev), sum(SNRseg_SNR_15dB_aftr)/len(SNRseg_SNR_15dB_aftr)],
    ['SNR_20dB', sum(SNRseg_SNR_20dB_prev)/len(SNRseg_SNR_20dB_prev), sum(SNRseg_SNR_20dB_aftr)/len(SNRseg_SNR_20dB_aftr)],
    ['Overall', sum(SNRseg_Overall_prev)/len(SNRseg_Overall_prev), sum(SNRseg_Overall_aftr)/len(SNRseg_Overall_aftr)],
    ])
print(SNRseg_t.draw())

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
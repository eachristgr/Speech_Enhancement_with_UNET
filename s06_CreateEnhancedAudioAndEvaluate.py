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
import pysepm
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

 
# Calculate PESQ & STOI  
PESQ_Overall_prev = []
PESQ_Overall_aftr = []
PESQ_SNR_0dB_prev = []
PESQ_SNR_0dB_aftr = []
PESQ_SNR_5dB_prev = []
PESQ_SNR_5dB_aftr = []
PESQ_SNR_10dB_prev = []
PESQ_SNR_10dB_aftr = []
PESQ_SNR_15dB_prev = []
PESQ_SNR_15dB_aftr = []
PESQ_SNR_20dB_prev = []
PESQ_SNR_20dB_aftr = []

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
    
    # Calculate PESQ & STOI for every sample
    #PESQ_prev = pysepm.pesq(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000)
    #PESQ_aftr = pysepm.pesq(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000)
    
    STOI_prev = stoi(wavs_CleanSpeech[:, idx], wavs_NoisySpeech[:, idx], 16000, extended=True)
    STOI_aftr = stoi(wavs_CleanSpeech[:, idx], wavs_EnhancedSpeech[:, idx], 16000, extended=True)
    
    # Add to Overall
    #PESQ_Overall_prev.append(PESQ_prev)
    #PESQ_Overall_aftr.append(PESQ_aftr)
    
    STOI_Overall_prev.append(STOI_prev)
    STOI_Overall_aftr.append(STOI_aftr)  
    
    # Add to scesific SNR category
    if ('SNRdb_0.0' in filenames_Testing_NoisySpeech[idx]):
        #PESQ_SNR_0dB_prev.append(PESQ_prev)
        #PESQ_SNR_0dB_aftr.append(PESQ_aftr)
        STOI_SNR_0dB_prev.append(STOI_prev)
        STOI_SNR_0dB_aftr.append(STOI_aftr)
    elif ('SNRdb_5.0' in filenames_Testing_NoisySpeech[idx]):
        #PESQ_SNR_5dB_prev.append(PESQ_prev)
        #PESQ_SNR_5dB_aftr.append(PESQ_aftr)
        STOI_SNR_5dB_prev.append(STOI_prev)
        STOI_SNR_5dB_aftr.append(STOI_aftr)       
    elif ('SNRdb_10.0' in filenames_Testing_NoisySpeech[idx]):
        #PESQ_SNR_10dB_prev.append(PESQ_prev)
        #PESQ_SNR_10dB_aftr.append(PESQ_aftr)
        STOI_SNR_10dB_prev.append(STOI_prev)
        STOI_SNR_10dB_aftr.append(STOI_aftr)
    elif ('SNRdb_15.0' in filenames_Testing_NoisySpeech[idx]):
        #PESQ_SNR_15dB_prev.append(PESQ_prev)
        #PESQ_SNR_15dB_aftr.append(PESQ_aftr)
        STOI_SNR_15dB_prev.append(STOI_prev)
        STOI_SNR_15dB_aftr.append(STOI_aftr)
    elif ('SNRdb_20.0' in filenames_Testing_NoisySpeech[idx]):
        #PESQ_SNR_20dB_prev.append(PESQ_prev)
        #PESQ_SNR_20dB_aftr.append(PESQ_aftr)
        STOI_SNR_20dB_prev.append(STOI_prev)
        STOI_SNR_20dB_aftr.append(STOI_aftr)
                
# Compute & Print Averages
"""
PESQ_t = Texttable()
PESQ_t.add_rows([
    ['SNR_Level', 'PESQ_Noisy_vs_Clean', 'PESQ_Enchanced_vs_Clean'],
    ['SNR_0dB', sum(PESQ_SNR_0dB_prev)/len(PESQ_SNR_0dB_prev), sum(PESQ_SNR_0dB_aftr)/len(PESQ_SNR_0dB_aftr)],
    ['SNR_5dB', sum(PESQ_SNR_5dB_prev)/len(PESQ_SNR_5dB_prev), sum(PESQ_SNR_5dB_aftr)/len(PESQ_SNR_5dB_aftr)],
    ['SNR_10dB', sum(PESQ_SNR_10dB_prev)/len(PESQ_SNR_10dB_prev), sum(PESQ_SNR_10dB_aftr)/len(PESQ_SNR_10dB_aftr)],
    ['SNR_15dB', sum(PESQ_SNR_15dB_prev)/len(PESQ_SNR_15dB_prev), sum(PESQ_SNR_15dB_aftr)/len(PESQ_SNR_15dB_aftr)],
    ['SNR_20dB', sum(PESQ_SNR_20dB_prev)/len(PESQ_SNR_20dB_prev), sum(PESQ_SNR_20dB_aftr)/len(PESQ_SNR_20dB_aftr)],
    ['Overall', sum(PESQ_Overall_prev)/len(PESQ_Overall_prev), sum(PESQ_Overall_aftr)/len(PESQ_Overall_aftr)],
    ])
print(PESQ_t.draw())
"""

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

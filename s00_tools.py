"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 00 - Useful Functions
"""

##################################################
# Imports
##################################################
import librosa
import numpy as np

##################################################
# Initialize Functions
##################################################
#-------------------------------------------------
# Function to extract magnitude and phase spectrograms
#-------------------------------------------------
def audio_to_spect(wav):
    D = librosa.stft(wav, n_fft=511, hop_length=127, window='hamming')
    mag_spect, phase_spect = librosa.magphase(D, power=1)
    phase_angle_spect = np.angle(phase_spect)
    return mag_spect, phase_angle_spect


#-------------------------------------------------
# Calculate Denoised Siganls
#-------------------------------------------------
def spect_to_audio(mag_spect, phase_angle_spect):
    phase_spect = np.cos(phase_angle_spect) + 1.j * np.sin(phase_angle_spect)
    wav = librosa.istft(mag_spect * phase_spect, hop_length=127, window='hamming')
    return wav

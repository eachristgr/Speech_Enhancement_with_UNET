"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 02 - Create & Save Spectograms
"""

##################################################
# Imports
##################################################
import numpy as np
import librosa
import s00_tools as tls
import os


##################################################
# Initialize Functions
##################################################
#-------------------------------------------------
# Function to extract spectograms for a path
#-------------------------------------------------
def filenames_to_spect(filenames):
    mag_spect_arrary = np.zeros((len(filenames), 256, 256), np.float32)
    phase_angle_spect_array = np.zeros((len(filenames), 256, 256), np.float32)
    for i in range(len(filenames)):
        wav, fs = librosa.load(filenames[i], sr=None)
        mag_spect, phase_angle_spect = tls.audio_to_spect(wav)
        mag_spect_arrary[i,:,:]= mag_spect
        phase_angle_spect_array[i,:,:] = phase_angle_spect
    return mag_spect_arrary, phase_angle_spect_array


##################################################
# Main
##################################################
#-------------------------------------------------
# TRAINING
#-------------------------------------------------
# Create Training NoisySpeach Spectrograms
Training_NoisySpeech_path = "Dataset_My_Wavs\\Training_NoisySpeech"
Training_NoisySpeech_filenames = [os.path.join(Training_NoisySpeech_path, f) for f in os.listdir(Training_NoisySpeech_path) if f.endswith(".wav")]
Training_NoisySpeech_filenames.sort()

Training_NoisySpeech_spect_arrary, Training_NoisySpeech_phase_array = filenames_to_spect(Training_NoisySpeech_filenames)
np.save('Dataset_My_Spects\\Training_NoisySpeech_spect_arrary', np.array(Training_NoisySpeech_spect_arrary))
#np.save('Dataset_My_Spects\\Training_NoisySpeech_phase_array', np.array(Training_NoisySpeech_phase_array))

"""
# Create Training Noise Spectrograms
Training_Noise_path = "Dataset_My_Wavs\\Training_Noise"
Training_Noise_filenames = [os.path.join(Training_Noise_path, f) for f in os.listdir(Training_Noise_path) if f.endswith(".wav")]
Training_Noise_filenames.sort()

Training_Noise_spect_arrary, Training_Noise_phase_array = filenames_to_spect(Training_Noise_filenames)
np.save('Dataset_My_Spects\\Training_Noise_spect_arrary', np.array(Training_Noise_spect_arrary))
np.save('Dataset_My_Spects\\Training_Noise_phase_array', np.array(Training_Noise_phase_array))
del Training_Noise_spect_arrary, Training_Noise_phase_array
"""

# Create Training CleanSpeach Spectrograms
Training_NoisySpeech_path = "Dataset_My_Wavs\\Training_CleanSpeech"
Training_CleanSpeech_filenames = Training_NoisySpeech_filenames
for i in range (len(Training_NoisySpeech_filenames)):
    noisyspeach_filename = Training_NoisySpeech_filenames[i]
    cleanspeach_filename = noisyspeach_filename[len(noisyspeach_filename)-13:len(noisyspeach_filename)]
    Training_CleanSpeech_filenames[i] = Training_NoisySpeech_path + "/" + cleanspeach_filename
    
Training_CleanSpeech_spect_arrary, Training_CleanSpeech_phase_array = filenames_to_spect(Training_CleanSpeech_filenames)
np.save('Dataset_My_Spects\\Training_CleanSpeech_spect_arrary', np.array(Training_CleanSpeech_spect_arrary))
#np.save('Dataset_My_Spects\\Training_CleanSpeech_phase_array', np.array(Training_CleanSpeech_phase_array))


#-------------------------------------------------
# TESTING
#-------------------------------------------------    
# Create Testing NoisySpeach Spectrograms
Testing_NoisySpeech_path = "Dataset_My_Wavs\\Testing_NoisySpeech"
Testing_NoisySpeech_filenames = [os.path.join(Testing_NoisySpeech_path, f) for f in os.listdir(Testing_NoisySpeech_path) if f.endswith(".wav")]
Testing_NoisySpeech_filenames.sort()

Testing_NoisySpeech_spect_arrary, Testing_NoisySpeech_phase_array = filenames_to_spect(Testing_NoisySpeech_filenames)
np.save('Dataset_My_Spects\\Testing_NoisySpeech_spect_arrary', np.array(Testing_NoisySpeech_spect_arrary))
np.save('Dataset_My_Spects\\Testing_NoisySpeech_phase_array', np.array(Testing_NoisySpeech_phase_array))


"""
# Create Testing Noise Spectrograms
Testing_Noise_path = "Dataset_My_Wavs\\Testing_Noise"
Testing_Noise_filenames = [os.path.join(Testing_Noise_path, f) for f in os.listdir(Testing_Noise_path) if f.endswith(".wav")]
Testing_Noise_filenames.sort()

Testing_Noise_spect_arrary, Testing_Noise_phase_array = filenames_to_spect(Testing_Noise_filenames)
np.save('Dataset_My_Spects\\Testing_Noise_spect_arrary', np.array(Testing_Noise_spect_arrary))
np.save('Dataset_My_Spects\\Testing_Noise_phase_array', np.array(Testing_Noise_phase_array))
del Testing_Noise_spect_arrary, Testing_Noise_phase_array
"""


# Create Testing CleanSpeach Spectrograms
Testing_NoisySpeech_path = "Dataset_My_Wavs\\Testing_CleanSpeech"
Testing_CleanSpeech_filenames = Testing_NoisySpeech_filenames
for i in range (len(Testing_NoisySpeech_filenames)):
    noisyspeach_filename = Testing_NoisySpeech_filenames[i]
    cleanspeach_filename = noisyspeach_filename[len(noisyspeach_filename)-13:len(noisyspeach_filename)]
    Testing_CleanSpeech_filenames[i] = Testing_NoisySpeech_path + "/" + cleanspeach_filename
    
Testing_CleanSpeech_spect_arrary, Testing_CleanSpeech_phase_array = filenames_to_spect(Testing_CleanSpeech_filenames)
np.save('Dataset_My_Spects\\Testing_CleanSpeech_spect_arrary', np.array(Testing_CleanSpeech_spect_arrary))
np.save('Dataset_My_Spects\\Testing_CleanSpeech_phase_array', np.array(Testing_CleanSpeech_phase_array))


##################################################
# ...
##################################################
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
id = 200
wav, fs = librosa.load(Training_NoisySpeech_filenames[id], None)
D = librosa.stft(wav, n_fft=511, hop_length=127, window='hamming')
m_spect, phase = librosa.magphase(D, power=1) # power=2
#lp_spect = librosa.amplitude_to_db(m_spect)
#lp_spect = 10*np.log10(m_spect + 1e-6)
#lp_spect = lp_spect + np.absolute(lp_spect.min())
phase_angle = np.angle(phase)

librosa.display.specshow(m_spect, sr=16000, hop_length=127)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()


plt.figure()
plt.imshow(m_spect, origin='lower')
plt.show()
"""
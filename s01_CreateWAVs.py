"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 01 - Explore & Create wav file
"""

##################################################
# Imports
##################################################
import os
import numpy as np
import soundfile as sf
import glob


##################################################
# Initialize Functions
##################################################
#-------------------------------------------------
# Function to read audio
#-------------------------------------------------
def audioread(path, norm = True, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0)/x.shape[0]
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr


#-------------------------------------------------
# Funtion to write audio  
#-------------------------------------------------  
def audiowrite(data, fs, destpath, norm=False):
    if norm:
        rms = (data ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 10) / (rms+eps)
        data = data * scalar
        if max(abs(data))>=1:
            data = data/max(abs(data), eps)
    
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, data, fs)
    return


#-------------------------------------------------
# Function to mix clean speech and noise at various SNR levels
#-------------------------------------------------
def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5

    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech


#-------------------------------------------------
# Function to synthesize CleanSpeach and Noise
#-------------------------------------------------
def synthesizer(raw_clean_dir, raw_noise_dir, F_NoisySpeech_dir, F_CleanSpeech_dir, F_Noise_dir, 
                sampling_rate, total_hours, audioclip_min_length,
                snr_lower, snr_upper, total_snr_levels,
                audio_format, file_samples, 
                silence_length):
    
    # Use directories for clean speach and noise
    clean_read_dir = os.path.join(os.path.abspath(''), raw_clean_dir)
    noise_read_dir = os.path.join(os.path.abspath(''), raw_noise_dir)
    
    # Create folders for new directories for noisy-speech, noise, clean-speach
    noisyspeech_write_dir = os.path.join(os.path.abspath(''), F_NoisySpeech_dir)
    if not os.path.exists(noisyspeech_write_dir): #if it doesn't exist, make it
        os.makedirs(noisyspeech_write_dir)
        
    clean_write_dir = os.path.join(os.path.abspath(''), F_CleanSpeech_dir)
    if not os.path.exists(clean_write_dir): #if it doesn't exist, make it
        os.makedirs(clean_write_dir)
        
    noise_write_dir = os.path.join(os.path.abspath(''), F_Noise_dir)
    if not os.path.exists(noise_write_dir): #if it doesn't exist, make it
        os.makedirs(noise_write_dir)
        
    # Create the appropriate variables for creating audio files
    fs = sampling_rate
    total_secs = int(total_hours * 60 * 60)
    total_samples = int(total_secs * fs)
    audio_length = int(audioclip_min_length * fs)
    SNR = np.linspace(snr_lower, snr_upper, total_snr_levels)
    
    # Create lists which hold the respected filenames
    clean_read_filenames = glob.glob(os.path.join(clean_read_dir, audio_format))
    noise_read_filenames = glob.glob(os.path.join(noise_read_dir, audio_format))
    
    # Synthesizer
    filecounter = 0 # number of files created
    num_samples = 0 # number of samples per file
    
    while num_samples < total_samples:
        
        # sample index is an integer which is going to be used for all the 
        # different clean files in order to choose one random file
        sample_index = np.random.randint(0, np.size(clean_read_filenames))
        clean, fs = audioread(clean_read_filenames[sample_index])
        
        if len(clean) > audio_length: #reached goal for minimum file length
            clean = clean
        else:
            while len(clean) <= audio_length:
                sample_index += 1 #next file usually continues the previous one
                if sample_index > np.size(clean_read_filenames): #out of border index
                    sample_index = np.random.randint(0, np.size(clean_read_filenames))
                newclean, fs = audioread(clean_read_filenames[sample_index])
                # use a moment of silence between appending the two files
                cleanconcat = np.append(clean, np.zeros(int(fs*silence_length)))
                clean = np.append(cleanconcat, newclean)
            #END WHILE
        #END IF
        # same procedure for noise
        noise_index = np.random.randint(0, np.size(noise_read_filenames))
        noise, fs = audioread(noise_read_filenames[noise_index])
        
        while len(noise) < len(clean):
            noise_index += 1 #next file usually continues the previous one
            if noise_index > np.size(noise_read_filenames): #out of border index
                noise_index = np.random.randint(0, np.size(noise_read_filenames))
            newnoise, fs = audioread(noise_read_filenames[noise_index])
            # use a moment of silence between appending the two files
            noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
            noise = np.append(noiseconcat, noise)
        #END WHILE
   
        quotient = len(clean) // file_samples #quotient 7 // 3 = 2, remainder 7 % 3 = 1
        clean = np.append(clean, np.zeros(int((quotient+1)*file_samples - len(clean))))
        if len(noise) > len(clean):
            noise = noise[:len(clean)]
        else:
            noise = np.append(noise, np.zeros(int((quotient+1)*file_samples - len(noise))))
        
        filecounter += 1
        cnt1=0
        
        while cnt1*file_samples <= len(clean):
            curr_clean = clean[(cnt1-1)*file_samples:(cnt1*file_samples)-1]
            curr_noise = noise[(cnt1-1)*file_samples:(cnt1*file_samples)-1]
            
            # These two lines just convert a simple number to a 4 digit number
            # with prepended zeros. i.e 34 = 0034, 7 = 0007, 139 = 0139
            str_filecounter = str(filecounter)
            zero_filecounter = str(str_filecounter).zfill(4)
        
            for i in range(np.size(SNR)):
                clean_snr, noise_snr, noisyspeech_snr = snr_mixer(clean=curr_clean, noise=curr_noise, snr=SNR[i])
                # create the filenames
                noisyspeechfilename = 'noise' + str(zero_filecounter) + '_SNRdb_' + str(SNR[i]) + '_clean' + str(zero_filecounter) + '.wav'
                cleanfilename = 'clean' + str(zero_filecounter) + '.wav'
                noisefilename = 'noise' + str(zero_filecounter) + '_SNRdb_' + str(SNR[i]) + '.wav'
                # create the paths for each filename
                noisyspeechpath = os.path.join(noisyspeech_write_dir, noisyspeechfilename)
                cleanpath = os.path.join(clean_write_dir, cleanfilename)
                noisepath = os.path.join(noise_write_dir, noisefilename)
                # create the new files
                audiowrite(noisyspeech_snr, fs, noisyspeechpath, norm=False)
                audiowrite(clean_snr, fs, cleanpath, norm=False)
                audiowrite(noise_snr, fs, noisepath, norm=False)
                num_samples = num_samples + len(noisyspeech_snr)
                #END FOR
            
            cnt1+=1
            filecounter += 1
        #END WHILE
        #LAST PART OF FILE NEEDS TO BE PREPARED
    return noisyspeech_write_dir, clean_write_dir, noise_write_dir
    #END FOR
    
    
##################################################
# Main
##################################################
#-------------------------------------------------
# General Parameters
#-------------------------------------------------
sampling_rate = 16000
audioclip_min_length = 4
snr_lower = 0
snr_upper = 20
total_snr_levels = 5 
audio_format = '*.wav'
file_samples = 32512
silence_length = 0.2


#-------------------------------------------------
# Create Train Data
#-------------------------------------------------
total_hours = 4
raw_clean_dir = 'Dataset_MS_SNSD\\train_clean'
raw_noise_dir = 'Dataset_MS_SNSD\\train_noise'
NoisySpeech_dir = 'Dataset_My_Wavs\\Training_NoisySpeech'
CleanSpeech_dir = 'Dataset_My_Wavs\\Training_CleanSpeech'
Noise_dir = 'Dataset_My_Wavs\\Training_Noise'

synthesizer(raw_clean_dir, raw_noise_dir, NoisySpeech_dir, CleanSpeech_dir, Noise_dir,
            sampling_rate, total_hours, audioclip_min_length,
            snr_lower, snr_upper, total_snr_levels,
            audio_format, file_samples, 
            silence_length)


#-------------------------------------------------
# Create Test Data
#-------------------------------------------------
total_hours = 0.5
raw_clean_dir = 'Dataset_MS_SNSD\\test_clean'
raw_noise_dir = 'Dataset_MS_SNSD\\test_noise'
NoisySpeech_dir = 'Dataset_My_Wavs\\Testing_NoisySpeech'
CleanSpeech_dir = 'Dataset_My_Wavs\\Testing_CleanSpeech'
Noise_dir = 'Dataset_My_Wavs\\Testing_Noise'

synthesizer(raw_clean_dir, raw_noise_dir, NoisySpeech_dir, CleanSpeech_dir, Noise_dir, 
            sampling_rate, total_hours, audioclip_min_length,
            snr_lower, snr_upper, total_snr_levels,
            audio_format, file_samples, 
            silence_length)


##################################################
# ...
##################################################
"""
Types_Of_Noise = ['AirConditioner', 'CopyMachine', 'Munching', 'ShuttingDoor', 'Typing', 'VacuumCleaner']

#-------------------------------------------------
# Explore Train Clean
#-------------------------------------------------
train_wav_clean_path = 'Dataset_MS_SNSD\\train_clean'

train_wav_clean_files = [pos_wav for pos_wav in os.listdir(train_wav_clean_path) if pos_wav.endswith('.wav')]

users = []
for wav_file in train_wav_clean_files:
    user = wav_file.split('_')[0]
    if user not in users:
        users.append(user)
        
SpeakersDF = pd.DataFrame(columns= ['Speaker', 'Number_Of_Samples'])
for user in users:
    user_wav_files = [pos_wav for pos_wav in os.listdir(train_wav_clean_path) if (pos_wav.startswith(user) & pos_wav.endswith('.wav'))]
    userF = {'Speaker': user, 'Number_Of_Samples': len(user_wav_files)}
    SpeakersDF = SpeakersDF.append(userF, ignore_index=True)

del users, wav_file, user, user_wav_files, userF
    
#-------------------------------------------------
# Explore Train Noise
#-------------------------------------------------    
train_wav_noise_path = 'Dataset_MS_SNSD\\train_noise'

train_wav_noise_files = [pos_wav for pos_wav in os.listdir(train_wav_noise_path) if pos_wav.endswith('.wav')]

types = []
for wav_file in train_wav_noise_files:
    type = wav_file.split('_')[0]
    if type not in types:
        types.append(type)
        
NoisesDF = pd.DataFrame(columns= ['Type_of_Noise', 'Number_Of_Samples'])
for type in types:
    type_wav_files = [pos_wav for pos_wav in os.listdir(train_wav_noise_path) if (pos_wav.startswith(type) & pos_wav.endswith('.wav'))]
    typeF = {'Type_of_Noise': type, 'Number_Of_Samples': len(type_wav_files)}
    NoisesDF = NoisesDF.append(typeF, ignore_index=True)
    
del types, wav_file, type, type_wav_files, typeF

"""
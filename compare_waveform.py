import scipy
import numpy as np 
import librosa 
from sklearn.metrics import pairwise 

#This code is a possibility for what we could do to compare two audio files.
#May not work for our use case because it doesn't use the FFTs of the waveforms

def extract_features(file_path): 
    # Load the audio file 
    y, sr = librosa.load(file_path, sr=None) 
     
    # Extract MFCCs 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) 
     
    # Take the mean of the MFCCs across time 
    mfccs_mean = np.mean(mfccs.T, axis=0) 
     
    return mfccs_mean 
 
def compare_audio(file1, file2): 
    # Extract features from both audio files 
    features1 = extract_features(file1) 
    features2 = extract_features(file2) 
     
    # Compute the Euclidean distance between the feature vectors 
    distance = np.linalg.norm(features1 - features2) 
     
    return distance 
 
# Example usage 
file1 = 'path/to/first_audio_file.wav' 
file2 = 'path/to/second_audio_file.wav' 
 
distance = compare_audio(file1, file2) 
print(f"Distance between the two audio files: {distance}") 
 
# Set a threshold for comparison 
threshold = 30  # You may need to adjust this based on your data 
if distance < threshold: 
    print("The same person is likely speaking in both audio files.") 
else: 
    print("The speakers are likely different.") 
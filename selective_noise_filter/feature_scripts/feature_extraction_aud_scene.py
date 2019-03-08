import os
import numpy as np
import librosa

class FeaturePrep:
    
    def __init__(self, feature_type, window_size, window_shift, sr, duration, num_features=None, directory = None):
        '''
        Directory == where the extracted features will be saved.
        '''
        self.feature_type = feature_type
        if feature_type == "stft":
            self.num_features = window_size * 8 + 1 #I don't know why the extra 1
        else:
            self.num_features = num_features
        self.window_size = window_size
        self.window_shift = window_shift
        self.sr = sr
        self.duration = duration
        self.directory = directory
    
    def get_feats(self,wav):
        #set window and shift sizes
        y,sr = librosa.load(wav,sr=self.sr,duration=self.duration)
        n_fft = int(self.window_size*0.001*sr)
        hop_length = int(self.window_shift*0.001*sr)
        if self.feature_type == "stft":
            feats = np.abs(librosa.stft(y,n_fft = n_fft, hop_length = hop_length))
            feats = np.transpose(feats)
            feats -= (np.mean(feats,axis=0)+1e-8)
        #if len(feats) < int(1/((self.window_shift * 0.001)/float(self.duration)))+1:
            
        
        return feats
    
    def create_empty_matrix(self,waves_list):
        self.total_samples = len(waves_list)
        '''
        total number of rows, same for all samples:
        total window (i.e. duration, e.g. 2 seconds)
        feature window (i.e. window_size, e.g. 50 ms) 
        feature window overlap (i.e. window_shift, e.g. 25 ms)
        '''
        self.feat_sets_per_wave = int(1/((self.window_shift * 0.001)/float(self.duration)))+1#to be honest, I don't know why I need to add 1.. might not generalize to other data. Works for 25 ms window shift.
        empty_matrix = np.zeros((self.total_samples*self.feat_sets_per_wave,self.num_features+1))
        #+1 for the label
        return empty_matrix
    
    def fill_matrix(self,matrix, wav,label_int,matrix_index):
        feats = self.get_feats(wav)
        #zero-padding if wave file shorter than desired length
        if len(feats) < self.feat_sets_per_wave:
            diff = self.feat_sets_per_wave - len(feats)
            feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))))
        for j, samp in enumerate(feats):
            row = np.append(samp,label_int)
            matrix[matrix_index+j] = row
        matrix_index += len(feats)
        
        return matrix, matrix_index
    
    def save_feats(self,features_matrix,timestamp):
        print(self.directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        np.save("{}/{}{}_feats_{}.npy".format(self.directory,self.feature_type,self.num_features,timestamp),features_matrix)
        return True
    

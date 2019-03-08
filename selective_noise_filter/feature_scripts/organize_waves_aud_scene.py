import os
import pandas as pd
import numpy as np
from random import shuffle
import csv

class OrgData:
    def __init__(self,label_type,labels,timestamp):
        self.label_type = self.create_feature_class_dir(label_type)
        self.feature_dir = "./{}/".format(label_type)
        self.labels = labels
        self.timestamp = timestamp
    
    def create_feature_class_dir(self,label_type):
        if not os.path.exists(label_type):
            os.makedirs(label_type)
        return label_type
    
    def csv2pandas(self,csv_path):
        label_type = self.label_type
        label_list = self.labels
        df = pd.read_csv(csv_path)
        df[self.label_type] = df["label"].apply(lambda x: True if x in label_list else False)
        df = df[df[self.label_type]==True]
        return df
    
    def waves_labels_dict(self,df):
        wl_dict = df.groupby('label')['fname'].apply(lambda x: x.values.tolist()).to_dict()
        return wl_dict
    
    def save_encoded_labels(self,dict_labels):
        with open("{}encoded_labels_{}_{}.csv".format(self.feature_dir,self.label_type,self.timestamp),'w') as f:
            w = csv.writer(f)
            w.writerows(dict_labels.items())
        return None
    
    def encode_labels(self):
        labels_list = self.labels.copy()
        labels_list = sorted(labels_list)
        labels2int = {}
        int2labels = {}
        for i, item in enumerate(labels_list):
            labels2int[item] = i
            int2labels[i] = item
        self.save_encoded_labels(int2labels)
        self.labels_encoded = labels2int
        return None
    
    def waves_labels2path_ints(self,data_path,dict_waves_labels,split=False):
        #if split, should be an integer between 0 and 100
        #split data into subsections, e.g. train and validation set, or train and test set 
        if split:
            wave_label_list1 = []
            wave_label_list2 = []
            legible_list1 = []
            legible_list2 = []
            for key, value in dict_waves_labels.items():
                #encode the label as the assigned integer
                label_int = self.labels_encoded[key]
                if len(value) <= 1:
                    print(value)
                waves_shuffled = value.copy()
                shuffle(waves_shuffled)
                threshold = int(len(waves_shuffled) * (split * 0.01))
                for i, wave in enumerate(waves_shuffled):
                    if i < threshold:
                        wave_label_list1.append((data_path+wave,label_int))
                        legible_list1.append((wave,key))
                    else:
                        wave_label_list2.append((data_path+wave,label_int))
                        legible_list2.append((wave,key))
        #otherwise, maintain dataset
        else:
            wave_label_list1 = []
            wave_label_list2 = None #no subsection
            for key, value in dict_waves_labels.items():
                #encode the label as the assigned integer
                label_int = self.labels_encoded[key]
                if len(value) <= 1:
                    print(value)
                for wave in value:
                    wave_label_list1.append((data_path+wave,label_int))
        shuffle(wave_label_list1)
        if wave_label_list2:
            shuffle(wave_label_list2)
            self.log_split_dataset(legible_list1,legible_list2)
        return wave_label_list1, wave_label_list2
    
    def log_split_dataset(self,dataset1, dataset2):
        with open("{}subset1_{}_{}.csv".format(self.feature_dir,self.label_type,self.timestamp),'w') as f:
            w = csv.writer(f)
            w.writerows(dataset1)
        with open("{}subset2_{}_{}.csv".format(self.feature_dir,self.label_type,self.timestamp),'w') as f:
            w = csv.writer(f)
            w.writerows(dataset2)
        return None

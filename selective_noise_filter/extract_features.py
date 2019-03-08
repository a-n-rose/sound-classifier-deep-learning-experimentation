'''
Extract features for acoustic scene analysis, according to research

Things to consider:

1) sound vs speech (difference in feature type)
- will extract 2 seconds, with 50 ms windows, with 25 ms overlap
- features = STFT (lowest comp cost, as no additional calculations made ontop (i.e. mel scale application, descrete cosine transform, etc)

2) accuracy (confusion matrix)
3) measure and compare computation cost and speed --> computational restrictions of raspberry pi / mobile device 
4) security: when / how to extract features (respect privacy)
5) eventual scalability (cloud)


Available labels (and their occurrences in training data):
('Acoustic_guitar', 300)
('Applause', 300)
('Bark', 239)
('Bass_drum', 300)
('Burping_or_eructation', 210)
('Bus', 109)
('Cello', 300)
('Chime', 115)
('Clarinet', 300)
('Computer_keyboard', 119)
('Cough', 243)
('Cowbell', 191)
('Double_bass', 300)
('Drawer_open_or_close', 158)
('Electric_piano', 150)
('Fart', 300)
('Finger_snapping', 117)
('Fireworks', 300)
('Flute', 300)
('Glockenspiel', 94)
('Gong', 292)
('Gunshot_or_gunfire', 147)
('Harmonica', 165)
('Hi-hat', 300)
('Keys_jangling', 139)
('Knock', 279)
('Laughter', 300)
('Meow', 155)
('Microwave_oven', 146)
('Oboe', 299)
('Saxophone', 300)
('Scissors', 95)
('Shatter', 300)
('Snare_drum', 300)
('Squeak', 300)
('Tambourine', 221)
('Tearing', 300)
('Telephone', 120)
('Trumpet', 300)
('Violin_or_fiddle', 300)
('Writing', 270)

'''
import sys
import datetime
from tqdm import tqdm



from feature_scripts.feature_extraction_aud_scene import FeaturePrep
from feature_scripts.organize_waves_aud_scene import OrgData


def get_date():
    '''
    This creates a string of the day, hour, minute and second
    I use this to make folder names unique
    
    For the files themselves, I generate genuinely unique names (i.e. name001.csv, name002.csv, etc.)
    '''
    time = datetime.datetime.now()
    time_str = "{}y{}m{}d{}h{}m{}s".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)

if __name__ == "__main__":
    #set up variables for feature extraction
    timestamp = get_date()
    
    feature_type = "stft"
    num_features = None 
    window_size = 50
    window_shift = window_size // 2
    sr = 16000
    duration = 2
    
    data_path_train = "/home/airos/downloaded_speech_data/kaggle_noise_classification/FSDKaggle2018.audio_train/"
    data_path_test = "/home/airos/downloaded_speech_data/kaggle_noise_classification/FSDKaggle2018.audio_test/"
    
    #noise type
    labels_type = "out_and_about" 
    
    #desired label sets:
    labels_test = ["Glockenspiel","Scissors"]
    labels_office = ["Computer_keyboard","Cough","Drawer_open_or_close","Keys_jangling","Laughter","Microwave_oven","Scissors","Squeak","Tearing","Telephone","Writing"]
    labels_out_and_about = ["Applause","Bus","Cough","Flute","Keys_jangling","Laughter","Meow","Squeak","Telephone","Writing"]

    waves_env = OrgData(labels_type,labels_out_and_about,timestamp)        
    
    #encode and save labels
    waves_env.encode_labels()

    train_df = waves_env.csv2pandas("/home/airos/downloaded_speech_data/kaggle_noise_classification/FSDKaggle2018.meta/train_post_competition.csv")
    print(train_df.head())
    
    test_df = waves_env.csv2pandas("/home/airos/downloaded_speech_data/kaggle_noise_classification/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv")
    print(test_df.head())
    
    wl_dict_train = waves_env.waves_labels_dict(train_df)
    wl_dict_test = waves_env.waves_labels_dict(test_df)

    #attach full path to waves
    #connect path to encoded label
    #randomize before extracting
    paths_encodedlabels_rand_train, paths_encodedlabels_rand_val = waves_env.waves_labels2path_ints(data_path_train,wl_dict_train,split=85)
    paths_encodedlabels_rand_test, __ = waves_env.waves_labels2path_ints(data_path_test,wl_dict_test)

    #extract features
    directory_train = "{}train_data".format(waves_env.feature_dir)
    directory_val = "{}val_data".format(waves_env.feature_dir)
    directory_test = "{}test_data".format(waves_env.feature_dir)
    
    feats_train = FeaturePrep(feature_type, window_size, window_shift, sr, duration, num_features = None, directory = directory_train)
    feats_val = FeaturePrep(feature_type, window_size, window_shift, sr, duration, num_features = None, directory = directory_val)
    feats_test = FeaturePrep(feature_type, window_size, window_shift, sr, duration, num_features = None, directory = directory_test)
 
    matrix_train = feats_train.create_empty_matrix(paths_encodedlabels_rand_train)
    matrix_val = feats_val.create_empty_matrix(paths_encodedlabels_rand_val)
    matrix_test = feats_test.create_empty_matrix(paths_encodedlabels_rand_test)
    
    print("Extracting train features:")
    matrix_train_index = 0
    for wavefile, label_int in tqdm(paths_encodedlabels_rand_train):
        matrix_train, matrix_train_index = feats_train.fill_matrix(matrix_train, wavefile,label_int,matrix_train_index)
    saved = feats_train.save_feats(matrix_train,timestamp)
    if saved:
        print("Saved!")
        matrix_train = None
    else:
        print("Error occurred. Features not extracted.")
        sys.exit()
        
    print("Extracting validation features:")
    matrix_val_index = 0
    for wavefile, label_int in tqdm(paths_encodedlabels_rand_val):
        matrix_val, matrix_val_index = feats_val.fill_matrix(matrix_val,wavefile,label_int,matrix_val_index)
    saved = feats_val.save_feats(matrix_val,timestamp)
    if saved:
        print("Saved!")
        matrix_val = None
    else:
        print("Error occurred. Features not extracted.")
        sys.exit()
        
    print("Extracting test features:")
    matrix_test_index = 0 
    for wavefile, label_int in tqdm(paths_encodedlabels_rand_test):
        matrix_test, matrix_test_index = feats_test.fill_matrix(matrix_test,wavefile,label_int,matrix_test_index)
    saved = feats_test.save_feats(matrix_test,timestamp)
    if saved:
        print("Saved!")
        matrix_test = None
    else:
        print("Error occurred. Features not extracted.")
        sys.exit()

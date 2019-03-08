'''
Model architectures are inspired from the (conference) paper:

Kim, Myungjong & Cao, Beiming & An, Kwanghoon & Wang, Jun. (2018). Dysarthric Speech Recognition Using Convolutional LSTM Neural Network. 10.21437/interspeech.2018-2250. 

'''

import os
import time
import datetime
from pathlib import Path

import numpy as np

#saving and visualizing data
import matplotlib.pyplot as plt
import csv

# for building and training models
import keras

# with 'callbacks', I can save the best version of a model as it trains
# I can also stop the training if it doesn't improve much
# As well as other features..
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint

# modules with functions I wrote to build the model and feed data to the model
from model_scripts.generator_aud_scene_data import Generator
import model_scripts.build_model as build


#to keep saved files unique
def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


    


def main(model_type,epochs,optimizer,sparse_targets,patience,duration,window_shift,window_size,total_feat_samps,timestep,context_size,overlap):

    
    #####################################################################
    ######################## HOUSE KEEPING ##############################
    
    start = time.time()
    
    #figure out the step size 
    #frame_width = 31
    #samps = 81
    #timestep = 10

    #i.e: overlap == 5:
    #0-31
    #4-36
    #9-41
    #14-46
    #19-51
    #24-56
    #29-61
    #34-66
    #39-71
    #44-76
    #49-81
    
    
    head_folder_curr_project = "./office"
    
    #create folders to store information, i.e. graphs, logs, models
    #create folders to store models, logs, and graphs
    graphs_folder = head_folder_curr_project+"/graphs"
    models_folder =  head_folder_curr_project+"/models"
    model_log_folder = head_folder_curr_project+"/model_logs"
    
    for folder in [graphs_folder,models_folder,model_log_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    #create name for model to be saved/associated with:
    #make model name unique with timestamp
    modelname = "{}_aud_scene".format(model_type.upper())
    #update model name if the name is already taken:
    path = unique_path(Path(models_folder), modelname+"{:03d}.hd")
    modelname = Path(path).stem
    
    ##collect variables stored during feature extraction
    #load_feature_settings_file = head_folder_curr_project+"/features_log.csv".format(project_head_folder)
    #with open(load_feature_settings_file, mode='r') as infile:
        #reader = csv.reader(infile)            
        #feats_dict = {rows[0]:rows[1] for rows in reader}
    
    #num_labels = int(feats_dict['num classes'])
    #num_features = int(feats_dict['num total features'])
    #timesteps = int(feats_dict['timesteps'])
    #context_window = int(feats_dict['context window'])
    
    #options:
    #3 sets of 27 sample-frames --> 82 samples in series (no overlap)
    #6 sets of 13 sample-frames
    #7 sets of 11 sample-frames
    context_window = 5
    frame_width = context_window * 2 + 1
    num_labels = 11
    num_features = 401
    timesteps = 7
    if overlap:
        step_size = (total_feat_samps - frame_width) // (timesteps - 1)
        #overlap = step1 // (timestep-1)
        #step_size = 5
        print("stepsize: ",step_size)
    else:
        step_size = 0
    
    
    #'communicate' with the user information about the model and features used to train it:
    print("\n\nTraining the {} model\n".format(model_type).upper())
    #print("\nInfo about training data:\n".upper())
    #for key, item in feats_dict.items():
        #print(key," : ",item)
    
    # specify some additional variables:
    frame_width = context_window*2+1
    #is for convolutional neural networks. They need to know which colors they're working with. 1 --> grayscale, 3 --> rgb or red-green-blue, 4 --> rgba or red-green-blue-alpha (whatever that last one is)
    color_scale = 1 
    #####################################################################
    ######################### BUILD MODEL  ##############################
    

    loss_type, activation_output = build.assign_model_settings(num_labels,sparse_targets)


    #build the model architecture:
    #read up on what they do and feel free to adjust!
    #For the LSTM:
    lstm_cells = num_features #what I've noticed people implementing..
    #For the CNN:
    feature_map_filters = num_features
    kernel_size = (8,4)
    #maxpooling
    pool_size = (3,3)
    #hidden dense layer
    dense_hidden_units = 60
    
    #feel free to adjust model architecture within the script 'build_model.py' in the folder 'model_scripts'
    model = build.buildmodel(model_type,num_labels,frame_width,timesteps,step_size,num_features,color_scale,lstm_cells,feature_map_filters,kernel_size,pool_size,dense_hidden_units,activation_output)
    
    #see what the model architecture looks like:
    print(model.summary())
    model.compile(optimizer=optimizer,loss=loss_type,metrics=['accuracy'])

    
    ######################################################################
    ###################### MORE HOUSEKEEPING!  ###########################
    
    #set up "callbacks" which help you keep track of what goes on during training
    #also saves the best version of the model and stops training if learning doesn't improve 
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience)
    csv_logging = CSVLogger(filename='{}/log_{}.csv'.format(model_log_folder,modelname))
    checkpoint_callback = ModelCheckpoint('{}/checkpoint_'.format(models_folder)+modelname+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    #####################################################################
    ################ LOAD TRAINING AND VALIDATION DATA  #################
    
    #load the .npy files containing the data
    filename_train = "{}/data_train/stft401_feats_2019y3m8d1h9m32s.npy".format(head_folder_curr_project)
    train_data = np.load(filename_train)
    print("train_shape",train_data.shape)
    
    filename_val = "{}/data_val/stft401_feats_2019y3m8d1h9m32s.npy".format(head_folder_curr_project)
    val_data = np.load(filename_val)

    #load these into their generators
    train_generator = Generator(model_type,train_data,timesteps,step_size,frame_width)
    val_generator = Generator(model_type,val_data,timesteps,step_size,frame_width)


    #####################################################################
    #################### TRAIN AND TEST THE MODEL #######################
    
    start_training = time.time()
    #train the model and keep the accuracy and loss stored in the variable 'history'
    #helpful in logging/plotting how training and validation goes
    if step_size == 0:
        interval = 1
    else:
        interval = step_size
    history = model.fit_generator(
            train_generator.generator(),
            steps_per_epoch = train_data.shape[0]/((timesteps*frame_width)//interval),
            epochs = epochs,
            callbacks=[early_stopping_callback, checkpoint_callback],
            validation_data = val_generator.generator(), 
            validation_steps = val_data.shape[0]/((timesteps*frame_width)//interval))
    end_training = time.time()
    #Note, please examine the generator class in the script "generator_speech_CNN_LSTM.py" in the folder "model_scripts"
    
    
    print("\nNow testing the model..")
    #now to test the model on brandnew data!
    filename_test = "{}/data_test/stft401_feats_2019y3m8d1h9m32s.npy".format(head_folder_curr_project)
    test_data = np.load(filename_test)
    test_generator = Generator(model_type,test_data,timesteps,step_size,frame_width)
    score = model.evaluate_generator(test_generator.generator(), test_data.shape[0]/((timesteps*frame_width)//interval))
    loss = round(score[0],2)
    acc = round(score[1]*100,3)

    msg="Model Accuracy on test data: {}%\nModel Loss on test data: {}".format(acc,loss)
    print(msg)
    
    print("Saving model..")
    model.save('{}/'.format(models_folder)+modelname+'.h5')
    print('Done!')
    print("\nModel saved as:\n{}.h5".format(models_folder+"/"+modelname))
    
    #####################################################################
    ####### TRY AND SEE WHAT THE HECK WAS GOING ON WHILE TRAINING #######

    print("Now saving history and plots")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("train vs validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")
    plt.savefig("{}/{}_LOSS.png".format(graphs_folder,modelname))

    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("train vs validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")
    plt.savefig("{}/{}_ACCURACY.png".format(graphs_folder,modelname))
    
    end = time.time()
    duration_total = round((end-start)/60,3)
    duration_training = round((end_training-start_training)/60,3)
    print("\nTotal duration = {}\nTraining duration = {}\n".format(duration_total,duration_training))
    
    
    #####################################################################
    ########## KEEP TRACK OF SETTINGS AND THE LOSS/ACCURACY #############
    

    #document settings used in model
    parameters = {}
    parameters["model type"] = model_type.lower()
    parameters["epochs"] = epochs
    if "lstm" in model_type.lower():
        parameters["num lstm cells"] = lstm_cells
    if "cnn" in model_type.lower():
        parameters["cnn feature maps"] = feature_map_filters
        parameters["cnn kernel size"] = kernel_size
        parameters["cnn maxpooling pool size"] = pool_size
        if "cnn" == model_type.lower():
            parameters["cnn dense hidden units"] = dense_hidden_units
    parameters["optimizer"] = optimizer
    parameters["num training data"] = len(train_data)

    #just to keep track and know for sure how many classes got presented during training and validation
    for key, value in train_generator.dict_classes_encountered.items():
        parameters["label "+str(key)+" representation in training"] = value
    for key, value in val_generator.dict_classes_encountered.items():
        parameters["label "+str(key)+" representation in validation"] = value
    parameters["loss type"] = loss_type
    parameters["activation output"] = activation_output
    parameters["test acc"] = acc
    parameters["test loss"] = loss
    parameters["duration in minutes"] = duration_training
    #save in csv file w unique name
    log_path = unique_path(Path(model_log_folder),modelname+'.csv')
    with open(log_path,'w',newline='') as f:
        w = csv.writer(f)
        w.writerows(parameters.items())
    
    print("\n\nIf you want to implement this model, the model's name is:\n\n{}\n\n".format(modelname))
    
    return True



if __name__ == "__main__":
    
    
    model_type = "cnnlstm" # cnn, lstm, cnnlstm
    epochs = 10
    optimizer = 'sgd' # 'adam' 'sgd'
    sparse_targets = True 
    patience = 5 
    #total time: 2 seconds of each kind of sound
    duration = 2
    window_shift = 25
    window_size = 50
    total_feat_samps = int(1/((window_shift * 0.001)/float(duration)))+1
    timestep = 10
    context_size = 15 # 5, 9 = speech recognition; 
    overlap = False

    
    
    try:
        main(model_type,epochs,optimizer,sparse_targets,patience,duration,window_shift,window_size,total_feat_samps,timestep,context_size,overlap)
    except UnboundLocalError as e:
        print("\n\nERROR: {}".format(e))
        print("\nCheck for typos in your input\n")

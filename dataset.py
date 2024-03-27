# from datasets.kinetics import Kinetics
# from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
# from datasets.hmdb51 import HMDB51
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import json
import numpy as np
import pandas as pd
from os import listdir
from keras_preprocessing.sequence import pad_sequences

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    if opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data

def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    if opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data

class ActionsDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        sequence = sequence.reshape(sequence.shape[1])

        return torch.tensor(sequence,dtype=torch.float), torch.tensor(label,dtype=torch.float)
        # return dict(
        #     # sequence=torch.Tensor(sequence.to_numpy()),
        #     sequence=torch.Tensor(sequence),
        #     label=torch.tensor(label).long()
        # )
     
def get_non_image_set():
    # read features json in order from train list and make a sequence
    train_non_image_data = []
    test_non_image_data = []

    hands_data_folder = '/Users/vladislav/Documents/JupyterNotebook/ActionDetection/CNN_LSTM/My_CNN_LSTM/data/hands_data/'
    train_file = pd.read_csv('/Users/vladislav/Documents/JupyterNotebook/ActionDetection/CNN_LSTM/My_CNN_LSTM/data/annotation/trainlist01.txt', sep=" ", header=None)
    train_file.columns = ['filename', 'classname']

    test_file = pd.read_csv('/Users/vladislav/Documents/JupyterNotebook/ActionDetection/CNN_LSTM/My_CNN_LSTM/data/annotation/testlist01.txt', sep=" ", header=None)
    test_file.columns = ['filename']

    classInd = pd.read_csv('/Users/vladislav/Documents/JupyterNotebook/ActionDetection/CNN_LSTM/My_CNN_LSTM/data/annotation/classInd.txt', sep=" ", header=None)
    classInd.columns = ['encoded_class', 'classname']
    
    len_train_features = []
    # find max length
    for i in range(0, len(train_file)):
        classname = train_file['filename'][i].split('/')[0]
        videoname = train_file['filename'][i].split('/')[1]
        # print(videoname)
        for filename in listdir(hands_data_folder+classname):
            if filename.startswith(videoname[:-4]):
                # print(filename)                     
                with open(hands_data_folder+classname+'/'+filename, 'r') as file:
                    data = json.load(file)
                    len_train_features.append(len(np.array(data[0])))

    len_test_features = []
    # find max length
    for i in range(0, len(test_file)):
        classname = test_file['filename'][i].split('/')[0]
        videoname = test_file['filename'][i].split('/')[1]
        # print(videoname)
        for filename in listdir(hands_data_folder+classname):
            if filename.startswith(videoname[:-4]):
                # print(filename)                     
                with open(hands_data_folder+classname+'/'+filename, 'r') as file:
                    data = json.load(file)
                    len_test_features.append(len(np.array(data[0])))

    if (np.array(len_train_features).max() >= np.array(len_test_features).max()):
        pad_length = np.array(len_train_features).max()
    else:
        pad_length = np.array(len_test_features).max()

    len_non_image_data = pad_length

    for i in range(0, len(train_file)):
        classname = train_file['filename'][i].split('/')[0]
        videoname = train_file['filename'][i].split('/')[1]
        #encode classname
        encoded_classname = classInd[classInd['classname'] == classname]['encoded_class'].to_numpy()
        # print(videoname)
        for filename in listdir(hands_data_folder+classname):
            if filename.startswith(videoname[:-4]):
                # print(filename)                     
                with open(hands_data_folder+classname+'/'+filename, 'r') as file:
                    data = json.load(file)
                    train_features = np.array(data)
                # print(train_features.shape)
                # pad to length
                padded_train_features = pad_sequences(train_features, maxlen=pad_length, truncating='post')
                # add class to array
                train_non_image_data.append((padded_train_features, encoded_classname))

    for i in range(0, len(test_file)):
        classname = test_file['filename'][i].split('/')[0]
        videoname = test_file['filename'][i].split('/')[1]
        #encode classname
        encoded_classname = classInd[classInd['classname'] == classname]['encoded_class'].to_numpy()
        # print(videoname)
        for filename in listdir(hands_data_folder+classname):
            if filename.startswith(videoname[:-4]):
                # print(filename)                     
                with open(hands_data_folder+classname+'/'+filename, 'r') as file:
                    data = json.load(file)
                    test_features = np.array(data)
                # print(train_features.shape)
                # pad to length
                padded_train_features = pad_sequences(test_features, maxlen=pad_length, truncating='post')
                # add class to array
                test_non_image_data.append((padded_train_features, encoded_classname))

    print("NON IMG DATA LEN: ", len_non_image_data)
    return ActionsDataset(train_non_image_data), ActionsDataset(test_non_image_data)

# def get_test_non_image_set():
#     # read features json in order from test list and make a sequence
#     test_non_image_data = []

#     hands_data_folder = '/Users/vladislav/Documents/JupyterNotebook/ActionDetection/CNN_LSTM/My_CNN_LSTM/data/hands_data/'
#     test_file = pd.read_csv('/Users/vladislav/Documents/JupyterNotebook/ActionDetection/CNN_LSTM/My_CNN_LSTM/data/annotation/testlist01.txt', sep=" ", header=None)
#     test_file.columns = ['filename']
    
#     len_test_features = []
#     # find max length
#     for i in range(0, len(test_file)):
#         classname = test_file['filename'][i].split('/')[0]
#         videoname = test_file['filename'][i].split('/')[1]
#         # print(videoname)
#         for filename in listdir(hands_data_folder+classname):
#             if filename.startswith(videoname[:-4]):
#                 # print(filename)                     
#                 with open(hands_data_folder+classname+'/'+filename, 'r') as file:
#                     data = json.load(file)
#                     len_test_features.append(len(np.array(data[0])))

#     pad_length = len_test_features.max()
#     for i in range(0, len(test_file)):
#         classname = test_file['filename'][i].split('/')[0]
#         videoname = test_file['filename'][i].split('/')[1]
#         # print(videoname)
#         for filename in listdir(hands_data_folder+classname):
#             if filename.startswith(videoname[:-4]):
#                 # print(filename)                     
#                 with open(hands_data_folder+classname+'/'+filename, 'r') as file:
#                     data = json.load(file)
#                     test_features = np.array(data[0])
#                 # print(train_features.shape)
#                 # pad to length
#                 padded_train_features = pad_sequences(test_features, maxlen=pad_length, truncating='post')
#                 # add class to array
#                 test_non_image_data.append((padded_train_features, classname))

#     return ActionsDataset(test_non_image_data)


# def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
#     assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
#     assert opt.test_subset in ['val', 'test']

#     if opt.test_subset == 'val':
#         subset = 'validation'
#     elif opt.test_subset == 'test':
#         subset = 'testing'
#     if opt.dataset == 'kinetics':
#         test_data = Kinetics(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#     elif opt.dataset == 'activitynet':
#         test_data = ActivityNet(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             True,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#     elif opt.dataset == 'ucf101':
#         test_data = UCF101(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#     elif opt.dataset == 'hmdb51':
#         test_data = HMDB51(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)

#     return test_data

'''Configureation of Generic Object Decoding'''
# reference: https://github.com/KamitaniLab/GenericObjectDecoding

import os


analysis_name = 'GenericObjectDecoding'

# Data settings

# Select only convolutional layers (ignore RELU, BN, etc.)
selected_layers = [1,23,52,75,101,124,148,175]
units_per_layer = 1000

subjects = {'Subject1' : ['data/Subject1.mat'],
            'Subject2' : ['data/Subject2.mat'],
            'Subject3' : ['data/Subject3.mat'],
            'Subject4' : ['data/Subject4.mat'],
            'Subject5' : ['data/Subject5.mat']}

rois = {'VC' : 'ROI_VC = 1',
        'LVC' : 'ROI_LVC = 1',
        'HVC' : 'ROI_HVC = 1',
        'V1' : 'ROI_V1 = 1',
        'V2' : 'ROI_V2 = 1',
        'V3' : 'ROI_V3 = 1',
        'V4' : 'ROI_V4 = 1',
        'LOC' : 'ROI_LOC = 1',
        'FFA' : 'ROI_FFA = 1',
        'PPA' : 'ROI_PPA = 1'}

num_voxel = {'VC' : 1000,
             'LVC' : 1000,
             'HVC' : 1000,
             'V1' : 500,
             'V2' : 500,
             'V3' : 500,
             'V4' : 500,
             'LOC' : 500,
             'FFA' : 500,
             'PPA' : 500}

image_feature_file = 'data/ImageFeatures.h5'
# features = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8']
# features = ['resnet1', 'resnet2', 'resnet3', 'resnet4', 'resnet5', 'resnet6', 'resnet7', 'resnet8']
features = ['resnet' + str(i_layer) for i_layer in selected_layers]

# Results settings
results_dir = os.path.join('results', analysis_name)
results_file = os.path.join('results', analysis_name + '.pkl')

# Figure settings
roi_labels = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC']

####################################################################################
# Execute: 
#   $ python collect_activations.py
# 
# Computes activation units from selected layers of neural network across train and
# test images and saves data by layer to files.
####################################################################################

# reference: https://github.com/philipperemy/keras-visualize-activations
# reference: https://github.com/KamitaniLab/GenericObjectDecoding
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import keras.backend as K
from dep.imagenet_utils import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from os import walk
import sys
import random
import cPickle as pickle
import time
import re

import god_config_copy as config

# -------------------------------------------------------------------------------- #
# Helper functions
# -------------------------------------------------------------------------------- #

def get_activation_units(model, model_inputs, random_sample=None, layer_name=None):

    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    # Select ResNet50 layers to capture activaiton units
    outputs = [outputs[i] for i in config.selected_layers]

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        
        if random_sample is None:
            activations.append(layer_activations.flatten())
        else:
            # Choose random_sample random units in each layer (consistent across images)
            random.seed(72)
            activations.append(random.sample(layer_activations.flatten(), random_sample))
    
    # Return list of units    
    return [unit for layer_activations in activations for unit in layer_activations]

# -------------------------------------------------------------------------------- #
# Collect activations of 1200 training images in segments
# -------------------------------------------------------------------------------- #

# Allow printing of large data structures
np.set_printoptions(threshold=sys.maxint)

# Get the model
model = ResNet50(include_top=True, weights='imagenet',
                 input_tensor=None, input_shape=None,
                 pooling=None, classes=1000)

# Training and testing image directories
training_images_dir = 'images/training'
testing_images_dir = 'images/test'
images_dirs = [training_images_dir, testing_images_dir]

file_names = []
category_names = []
count = 0

activation_units = np.zeros((1200+50-3, len(config.selected_layers) * config.units_per_layer))

#################################################################################
#                           COLLECT ACTIVATIONS (~2500s)                        #
#################################################################################

# Collect activations over train and test images
start_time = time.time()
for images_dir in images_dirs:

    for root, dirs, files in walk(images_dir):
        
        if root == images_dir: continue
        print("Collecting activation units in %s" % root)

        for file_name in files:
            if file_name != ".DS_Store":
                print(file_name)

                # Load image
                img_dir = root
                img_name = file_name
                img_path = join(img_dir, img_name)
                img = image.load_img(img_path, target_size=(224, 224))

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # Compute activation units
                activation_units[count] = get_activation_units(model, x, config.units_per_layer)
                
                # Store data in file_names list and activation_units matrix
                parts = re.split("_|\.", file_name)
                file_name_float = float(parts[0][1:]) + float(parts[1])/1000000
                file_names.append([file_name_float])

                count += 1

print("Computed activation units in %.4f seconds" % (time.time() - start_time))


# Save units to file (~10s)
start_time = time.time()
units_file = "activation_units_TT.p"
pickle.dump(activation_units, open(units_file, "wb"))
print("Saved units to file %s in %.4f seconds" % (units_file, (time.time() - start_time)))


# Save filename list
pickle.dump(file_names, open("activation_file_names_TT.p", "wb"))
print("Saved file names.")


# Load matrix of activation units for all images (~30 sec)
start_time = time.time()
all_activation_units_file = "activation_units_TT.p"
units = pickle.load(open(all_activation_units_file, "rb"))
print("Loaded units in %.4f seconds" % (time.time() - start_time))


# 8 ResNet layers recorded
start_time = time.time()
for i in range(len(config.selected_layers)):
    # units_file = "resnet" + str(i+1) + ".p"
    units_file = "resnet" + str(config.selected_layers[i]) + ".p"
    pickle.dump(units[:, i * 1000:(i+1)*1000], open(units_file, "wb"))
print("Saved units in %.4f seconds" % (time.time() - start_time))

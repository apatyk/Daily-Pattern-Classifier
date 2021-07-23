# Adam Patyk
# Clemson University
# MS Thesis: Daily Pattern Classifier
# Summer 2021

# GenerateSamples.py
# Purpose: Generates daily samples for data augmentation
# Usage: python GenerateSamples.py <window_length_minutes>

import sys
import os
import tensorflow as tf # updated for TensorFlow 2.2.0
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm

sys.path.append('../') # for .py files in ../common/
import common.loadfile as loadfile
import common.training as training
import common.testing as testing

if len(sys.argv) != 2:
    sys.exit("Usage: python GenerateSamples.py <window_length_in_min>")  

# prepare TensorFlow for GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

epochs = 30
win_min = int(sys.argv[1]) #6
train_stride_sec = 15
test_stride_sec = 100

win_len = int(win_min * 60 * 15)
train_step = int(train_stride_sec * 15)
test_step = int(test_stride_sec * 15)
start_time = datetime.now()

save_dir = '/scratch2/apatyk/'

arr = ["echo -n 'PBS: node is '; cat $PBS_NODEFILE",\
      "echo PBS: job identifier is $PBS_JOBID",\
      "echo PBS: job name is $PBS_JOBNAME"]

[os.system(cmd) for cmd in arr]

print("*****************************************************************", flush=True)
print("Execution Started at " + start_time.strftime("%m/%d/%Y, %H:%M:%S"), flush=True)
print("Window Length: {:.2f} min ({:d} data)\tTraining Slide: {:d} sec ({:d} data)\tTesting Slide: {:d} sec ({:d} data)\tEpochs: {:d}".format(win_min, win_len, train_stride_sec, train_step, test_stride_sec, test_step, epochs), flush=True)

# load the dataset for training wiht majority vote GT labeling for windows 
num_files, all_training_data, training_samples_array, training_labels_array = loadfile.loadAllData3(win_len,
                                                                                                    train_step,
                                                                                                    removerest=0,
                                                                                                    removewalk=0,
                                                                                                    removebias=1)

# load the dataset for testing with a different stride and GT labeling (center point)
all_testing_data, testing_samples_array, testing_labels_array = loadfile.loadAllDataTesting('../common/batch-unix.txt', 
                                                                                              win_len, 
                                                                                              test_step, 
                                                                                              removebias=1)

print("Data loaded.", flush=True)

# normalize the datasets
shimmer_global_mean = [-0.012359981,-0.0051663737,0.011612018,
                        0.05796114,0.1477952,-0.034395125 ]

shimmer_global_stddev = [0.05756385,0.040893298,0.043825723, 
                        17.199743,15.311142,21.229317 ]

shimmer_trended_mean = [-0.000002,-0.000002,-0.000000,
                0.058144,0.147621,-0.033260 ]

shimmer_trended_stddev = [0.037592,0.034135,0.032263,
                17.209038,15.321441,21.242532 ]

all_zero_means = [0,0,0,0,0,0]

mean_vals = all_zero_means
std_vals = shimmer_trended_stddev

all_training_normalized = loadfile.globalZscoreNormalize(all_training_data, mean_vals, std_vals)
all_testing_normalized = loadfile.globalZscoreNormalize(all_testing_data, mean_vals, std_vals)

print("Data normalized.")

# generate training samples from trained model
num_samples = 200000
subjects = [*range(num_files)]
num_subjects = len(subjects)
num_iterations = math.ceil(num_samples / num_subjects)

print(f'Generating training samples ({num_subjects} subjects)', flush=True)

for i in tqdm(range(num_iterations)):
    start_time = datetime.now()
    
    # train model on all training data
    H, model = training.trainModel(training_samples_array, training_labels_array, all_training_normalized, win_len, epochs, save_dir + f'tmp_{win_min}min.h5')
    
    # output P(E) and GT to text file for each recording using the trained model
    for s in subjects:
        subject_bool = np.isin(testing_samples_array[:,0], s)
        s_samples = testing_samples_array[subject_bool]
        s_labels = testing_labels_array[subject_bool]
        raw_samples, gt_labels = testing.get_raw_data(s_samples, s_labels, all_testing_normalized)
        if raw_samples.size != 0:
            probs = model.predict(raw_samples, batch_size=1024)
            result = np.hstack((np.reshape(gt_labels,(1,-1)).T, probs))
            np.savetxt(save_dir + f'training-samples/W{win_min}_P{s:03.0f}_I{i:03.0f}.txt', result)
    
    tf.keras.backend.clear_session()
    del model
    
    end_time = datetime.now()
    print(f'Iteration Duration: {end_time - start_time}', flush=True)

print(f'{num_iterations * num_subjects} testing samples saved.')
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import sklearn
from sklearn.model_selection import KFold
from datetime import datetime

sys.path.append('../') # for .py files in ../common/
import common.testing as testing

# prepare for GPU workflow
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

len_threshold = 850
k = 5
epochs = int(sys.argv[3]) #30
batch_size = int(sys.argv[1]) #64
num_units = int(sys.argv[2]) #16
model_path = f'models/daily-pattern-gru-b{batch_size}-u{num_units}-e{epochs}.h5'
num_subjects = 354
n_timesteps = len_threshold
save_data = True

# load numpy arrays from binary .npy files (created from .txt samples in LoadFiles script)
raw_samples = np.load('200k-samples-W6/daily-samples.npy', allow_pickle=True)
raw_labels = np.load('200k-samples-W6/daily-labels.npy', allow_pickle=True)
all_filenames = np.load('200k-samples-W6/daily-filenames.npy').astype(int)
original_sample_lengths = np.array([len(sample) for sample in raw_samples])

# pad or truncate data sequences accordingly
all_samples = pad_sequences(raw_samples, len_threshold, dtype='float64', padding='post', truncating='post', value=-1)
all_labels = pad_sequences(raw_labels, len_threshold, dtype='int32', padding='post', truncating='post', value=-1)
print('Data ready.')

# prepare k-fold cross validation
kfold = KFold(k, shuffle=True, random_state=seed)
# randomly shuffle array of indices
x = range(num_subjects)
subjects = np.array(random.sample(x, num_subjects), copy=False)

total_TPR, total_TNR, total_F1, total_Prec, total_WAcc = [], [], [], [], []
total_ep_TPR, total_ep_F1, total_ep_FP_TP = [], [], []

print(f'Training with batch_size = {batch_size}, units = {num_units}', flush=True)
for i, (training_subjects, testing_subjects) in enumerate(kfold.split(subjects)):
    ### TRAINING
    print(f'FOLD {i+1}') 
    # retrieve only samples/labels corresponding to training fold
    print('Training...')
    training_bool = np.isin(all_filenames, training_subjects)
    training_samples = tf.convert_to_tensor(all_samples[training_bool], np.float32)
    training_labels = tf.convert_to_tensor(all_labels[training_bool], np.int8)
    
    training_samples = tf.reshape(training_samples, (-1, n_timesteps, 1))
    training_labels = tf.reshape(training_labels, (-1, n_timesteps, 1))
    
    tf.keras.backend.clear_session()
    mcp_save = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='accuracy')

    # define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Masking(mask_value=-1,
                                input_shape=(n_timesteps, 1)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=num_units, 
                                 return_sequences=True,
                                 kernel_initializer='glorot_normal', # Xavier normal initialization
                                 bias_initializer='zeros'),
            merge_mode='sum'
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=training_samples, y=training_labels,
                        epochs=epochs, batch_size=batch_size, verbose=2,
                        callbacks=[mcp_save])
    
    ### TESTING
    print('Testing...')
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
    total_ep_TP, total_ep_FP, total_ep_FN = 0, 0, 0

    # retrieve only samples/labels corresponding to testing fold
    testing_bool = np.isin(all_filenames, testing_subjects)
    testing_samples = tf.convert_to_tensor(all_samples[testing_bool], np.float32)
    testing_labels = tf.convert_to_tensor(all_labels[testing_bool], np.int8)
    testing_sample_lengths = original_sample_lengths[testing_bool]
    
    testing_samples = tf.reshape(testing_samples, (-1, n_timesteps, 1))
    testing_labels = tf.reshape(testing_labels, (-1, n_timesteps, 1))
    
    # inference for all testing data using best model from training
    model = tf.keras.models.load_model(model_path)
    testing_probs = model.predict(testing_samples, batch_size=4096)
    
    # save data for testing post-hoc (varying thresholds)
    if save_data: 
        np.save(f'testing/testing_lengths_{epochs}epochs_fold{i+1}.npy', testing_sample_lengths)
        np.save(f'testing/testing_probs_{epochs}epochs_fold{i+1}.npy', testing_probs)
        np.save(f'testing/testing_samples_{epochs}epochs_fold{i+1}.npy', tf.squeeze(testing_samples).numpy())
        np.save(f'testing/testing_labels_{epochs}epochs_fold{i+1}.npy', tf.squeeze(testing_labels).numpy())

    # get episode metrics on testing dataset
    for i in range(len(testing_labels)):
        probs = testing_probs[i,:testing_sample_lengths[i]]
        gt_labels = testing_labels[i,:testing_sample_lengths[i]]
        # thresholding segmentation
        results = testing.single_threshold(probs, gt_labels, winmin=6, stepsec=100, threshold=0.1)
        # time-based metrics
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(gt_labels, results['predictions'][0], labels=[0,1]).ravel()
        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN
        # episode-based metrics
        ep_TP, ep_FP, ep_FN = testing.calc_episode_metrics(results, gt_labels)
        total_ep_TP += ep_TP
        total_ep_FP += ep_FP
        total_ep_FN += ep_FN

    # calculate and report overall metrics
    TPR = testing.true_positive_rate(total_TP, total_FN)
    TNR = testing.true_negative_rate(total_TN, total_FP)
    F1 = testing.f1_score(total_TP, total_FP, total_FN)
    Prec = testing.precision(total_TP, total_FP)
    WAcc = testing.weighted_accuracy(total_TP, total_FP, total_TN, total_FN)

    print('--- Time Metrics ---')
    print(f'WAcc: {WAcc:.3f}\tTPR: {TPR:.3f}\tTNR: {TNR:.3f}\tF1: {F1:.3f}\tPrecision: {Prec:.3f}')

    ep_TPR = testing.true_positive_rate(total_ep_TP, total_ep_FN)
    ep_F1 = testing.f1_score(total_ep_TP, total_ep_FP, total_ep_FN)
    ep_FP_TP = total_ep_FP / total_ep_TP

    print('--- Episode Metrics ---')
    print(f'TPR: {ep_TPR:.3f}\tF1: {ep_F1:.3f}\tFP/TP: {ep_FP_TP:.3f}')
    
    total_TPR.append(TPR)
    total_TNR.append(TNR)
    total_F1.append(F1)
    total_Prec.append(Prec)
    total_WAcc.append(WAcc)
    total_ep_TPR.append(ep_TPR)
    total_ep_F1.append(ep_F1)
    total_ep_FP_TP.append(ep_FP_TP)
    
    del model
    print("*****************************************************************", flush=True)

print('AVERAGE:')
print('--- Time Metrics ---')
print(f'WAcc: {np.mean(total_WAcc):.3f}\tTPR: {np.mean(total_TPR):.3f}\tTNR: {np.mean(total_TNR):.3f}\tF1: {np.mean(total_F1):.3f}\tPrecision: {np.mean(total_Prec):.3f}')

print('--- Episode Metrics ---')
print(f'TPR: {np.mean(total_ep_TPR):.3f}\tF1: {np.mean(total_ep_F1):.3f}\tFP/TP: {np.mean(total_ep_FP_TP):.3f}')

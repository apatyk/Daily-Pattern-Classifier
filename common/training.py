import random
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, GlobalAveragePooling1D

import tensorflow.keras.backend as kb

### undersamples and trains model from samples and labels provided
def trainModel(training_samples, training_labels, data, winlength, epochs, modelpath):
    # undersample the training dataset
    eating_indices = [i for i, e in enumerate(training_labels) if e >= 0.5]
    noneating_indices = [i for i, e in enumerate(training_labels) if e < 0.5]
    undersampled_noneating_indices = random.sample(noneating_indices, len(eating_indices))
    undersampled_balanced_indices = eating_indices + undersampled_noneating_indices
    shuffled_undersampled_balanced_indices = undersampled_balanced_indices.copy()
    random.shuffle(shuffled_undersampled_balanced_indices)

    axdata = []
    aydata = []

    for i in shuffled_undersampled_balanced_indices:
        f = training_samples[i,0]
        t1 = training_samples[i,1]
        t2 = training_samples[i,2]
        sample = data[f][t1:t2]
        label = training_labels[i]
        axdata.append(sample)
        aydata.append(label)

    balanced_data = np.array(axdata, copy=True)
    balanced_labels = np.array(aydata, copy=True)
    del axdata
    del aydata
    
    print("Training on {:d} samples of length {:d}".format(len(shuffled_undersampled_balanced_indices), len(balanced_data[0])))

    tf.keras.backend.clear_session()

    # use multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        mcp_save = keras.callbacks.ModelCheckpoint(modelpath, save_best_only=True, monitor='accuracy')

        model = Sequential()
        model.add(Conv1D(10, 44, strides=2,activation='relu', input_shape=(winlength, 6), name='input_layer'))
        model.add(Conv1D(10, 20, strides=2, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01)))
        model.add(Conv1D(10, 4, strides=2, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01)))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(1, activation='sigmoid', name='output_layer'))

        model.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

        H = model.fit(x=balanced_data, y=balanced_labels,
                    epochs=epochs, batch_size=256, verbose=0,
                    callbacks=[mcp_save])
    
    del balanced_data
    del balanced_labels
    
    return H, model

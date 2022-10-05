#---------------------------------------------------------------------------------------------------#
# File name: train.py                                                                               #
# Autor: Chrissi2802                                                                                #
# Created on: 08.09.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Google Brain - Ventilator Pressure Prediction (VPP)
# Exact description in the functions.
# This file provides functions for training and testing.


import dataset, models, helpers

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np


def plot_loss_and_error(train_losses, train_error, test_losses = [], test_error = [], fold = ""):
    """This function plots the loss and error for training and, if available, for validation."""
    # Input:
    # train_losses; list, Loss during training for each epoch
    # train_error; list, Error during training for each epoch
    # test_losses; list default [], Loss during validation for each epoch
    # test_error; list default [], Error during validation for each epoch
    # fold; string default "", Fold number

    fig, ax1 = plt.subplots()
    epochs = len(train_losses)
    xaxis = list(range(1, epochs + 1))

    # Training
    # Loss
    trl = ax1.plot(xaxis, train_losses, label = "Training Loss", color = "red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    # Error
    ax2 = ax1.twinx()
    tra = ax2.plot(xaxis, train_error, label = "Training Error", color = "fuchsia")
    ax2.set_ylabel("Mean absolute percentage error in %")
    ax2.set_ylim(0.0, 200.0)
    lns = trl + tra # Labels

    # Test
    if ((test_losses != []) and (test_error != [])):
        # Loss
        tel = ax1.plot(xaxis, test_losses, label = "Validation Loss", color = "lime")

        # Error
        tea = ax2.plot(xaxis, test_error, label = "Validation Error", color = "blue")

        lns = trl + tel + tra + tea    # Labels

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    plt.title("Loss and Error Fold " + fold)
    fig.savefig("Loss_and_Error_Fold_" + fold + ".png")
    plt.show()


def train_vpp():
    """This function performs the training and testing for the Ventilator Pressure Prediction (VPP) dataset."""

    # Hyperparameter
    epochs = 2 #500    # For testing 2
    batch_size = 1024
    verbose = 1
    
    # Hardware config
    strategy = helpers.hardware_config("GPU")

    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    with strategy.scope():

        # Dataset
        CVPP_Datasets = dataset.VPP_Datasets(create_new = True, many_features = True)
        train, target, test = CVPP_Datasets.get_datasets()

        path = CVPP_Datasets.get_path()
        path_models = path.rstrip("Dataset/") + "/Models/"

        # Crossvalidation
        k_fold = KFold(n_splits = 5, shuffle = True)    # For testing 2
        
        # Numpy array for the predictions
        test_predictions = np.empty([test.shape[0] * test.shape[1], k_fold.n_splits])

        # Perform the crossvalidation
        for fold, (train_index, test_index) in enumerate(k_fold.split(train, target)):

            print("Fold:", fold)

            # Data for this fold
            x_train, x_valid = train[train_index], train[test_index]
            y_train, y_valid = target[train_index], target[test_index]

            # Wrap data in Dataset objects
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).with_options(options)
            valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).with_options(options)
            test_data = tf.data.Dataset.from_tensor_slices((test)).batch(batch_size).with_options(options)

            # Model, choose one
            #model = models.gru_net(train)
            model = models.gru_net_big(train)
            #model = models.lstm_net_big(train)

            print(model.summary())

            model.compile(optimizer = "adam", loss = "mae", metrics = [tf.keras.metrics.MeanAbsolutePercentageError()])

            learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10, verbose = verbose)
            early_stopping = EarlyStopping(monitor = "val_loss", patience = 60, verbose = verbose, mode = "min", 
                                           restore_best_weights = True)
            model_checkpoint = ModelCheckpoint(path_models + model.name + str(fold) + ".hdf5", monitor = "val_loss", verbose = verbose, save_best_only = True, 
                                               mode = "auto", save_freq = "epoch")

            # Training
            history = model.fit(train_data, 
                                validation_data = valid_data, 
                                epochs = epochs,
                                verbose = 2,    # for debugging verbose
                                batch_size = batch_size, 
                                callbacks = [learning_rate, early_stopping, model_checkpoint])

            # Plot training and testing curves
            plot_loss_and_error(history.history["loss"], history.history["mean_absolute_percentage_error"], 
                                history.history["val_loss"], history.history["val_mean_absolute_percentage_error"], str(fold))

            # Save predictions 
            test_predictions[:, fold] = model.predict(test_data, batch_size = batch_size).squeeze().reshape(-1, 1).squeeze()
            print()

        # Save submissions
        CVPP_Datasets.write_submissions_mean(test_predictions)
        print("Training, validation and testing completed!")


if (__name__ == "__main__"):

    Pr = helpers.Program_runtime()

    train_vpp()
    
    Pr.finish(print = True)

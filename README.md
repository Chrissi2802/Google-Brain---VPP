# Google Brain - Ventilator Pressure Prediction (VPP)

This repository contains several models for a regression of the Google Brain - Ventilator Pressure Prediction (VPP) dataset. The website for this Kaggle Competition can be found [here](https://www.kaggle.com/competitions/ventilator-pressure-prediction "Kaggle Competition VPP website"). <br> 
Neural networks are used for feature extraction and regression. These were implemented in Python using the [TensorFlow](https://www.tensorflow.org/ "TensorFlow website") library. <br>
Furthermore, not only are various neural networks available, but also training procedures and scripts for data preprocessing and storage. <br> 

## Data
The aim of this competition is to predict the airway pressure in the breathing circuit of a breath based on numerous time series of breaths. For this purpose, various control inputs and lung property are available, which serve as a basis for prediction. The competition website also lists further details on the data. <br> 
The data cannot be made available here. However, they can be downloaded from the Kaggle website (link above). <br> 

## Models (neural networks)
The following models were trained and tested:
- Small Gated Recurrent Units (GRU), this is a Recurrent Neural Network (RNN) model <br> 
- Big GRU model <br>
- Big Long Short-Term Memory (LSTM) model <br> 

## Overview of the folder structure and files
| Files               | Description                                                         |
| ------------------- | ------------------------------------------------------------------- |
| Dataset/            | contains the data and the submissions                               |
| Models/             | contains the trained models                                         |
| Plots/              | contains all plots from the training and testing                    |
| dataset.py          | provides the dataset and prepares the data                          |
| helpers.py          | provides auxiliary classes and functions for neural networks        |
| Job.sh              | provides a script to carry out the training on a computer cluster   |
| models.py           | provides the models                                                 |
| train.py            | provides functions for training and testing                         |
| vpp_jupyter.ipynb   | contains all existing python files in a jupyter notebook            |

## Achieved results
The scores were calculated by Kaggle. The metric is the mean absolute error (MAE). <br> 
| Models          | Private leaderboard score   | Training time (hh:mm:ss)    |
| --------------- | --------------------------- | --------------------------- |
| GRU             | 0.3943                      | 02:27:43                    |
| Big GRU         | 0.2917                      | 07:38:54                    |
| Big LSTM        | 0.3021                      | 10:57:58                    |

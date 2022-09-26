#---------------------------------------------------------------------------------------------------#
# File name: dataset.py                                                                             #
# Autor: Chrissi2802                                                                                #
# Created on: 08.09.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Google Brain - Ventilator Pressure Prediction (VPP)
# Exact description in the functions.
# This file provides the datasets and prepares the data.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from datetime import datetime


class VPP_Datasets():
    """Class to design the Ventilator Pressure Prediction (VPP) Datasets."""

    def __init__(self, create_new = True, many_features = True):
        """Initialisation of the class (constructor). It prepares the data to be used for training and testing."""
        # Input:
        # create_new; boolean default True, create new data or load old data
        # many_features; boolean default True, many features should be added

        print("Prepare the dataset for training and testing ...")

        self.path = "./Dataset/"    
        self.submission = pd.read_csv(self.path + "sample_submission.csv")
        self.create_new = create_new
        self.many_features = many_features

        if (self.create_new == True):
            self.dataset_train = pd.read_csv(self.path + "train.csv")
            self.dataset_test = pd.read_csv(self.path + "test.csv")
            self.__feature_engineering()

            if (self.many_features == False):
                self.__visualisation()

            self.__clean_and_split()
            self.__normalize()
            self.__reshape()
        else:
            self.load_dataset_numpy()

        print("Preparation of the data completed!")

    def __feature_engineering(self):
        """This method extends and changes the features."""

        if (self.many_features == True):
            # many features from the internet
            self.drop_columns = ["id", "breath_id", "one", "count", "breath_id_lag", "breath_id_lag2", 
                                 "breath_id_lagsame", "breath_id_lag2same", "pressure"]
            self.dataset_train = self.__add_many_features(self.dataset_train)
            self.dataset_test = self.__add_many_features(self.dataset_test)
        else:
            # few features from the internet
            self.drop_columns = ["id", "breath_id", "pressure"]
            self.__add_features(self.dataset_train)
            self.__add_features(self.dataset_test)

    def __add_features(self, dataset):
        """This method adds a few features from the internet to a DataFrame."""
        # Code from: https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/273974
        # Input:
        # dataset; DataFrame

        # add a new feature which is the cumulative sum of the u_in feature
        dataset["u_in_cumsum"] = (dataset["u_in"]).groupby(dataset["breath_id"]).cumsum()

    def __visualisation(self):
        """This method visualises the data."""

        self.dataset_train.head(1000).plot(subplots = True, sharex = True, title = "Example data", figsize = (16, 9), layout = (5, 2))
        plt.savefig("Example_data.png")
        plt.show()

    def __clean_and_split(self):
        """This method splits the data from the labels and deletes unusable features."""

        self.targets = self.dataset_train["pressure"].to_numpy()
        self.dataset_train = self.dataset_train.drop(self.drop_columns, axis = 1)

        self.drop_columns.pop() # Delete last element, targets
        self.dataset_test = self.dataset_test.drop(self.drop_columns, axis = 1)

    def __normalize(self):
        """This method scales / normalises the features."""

        scaler = RobustScaler()

        self.dataset_train = scaler.fit_transform(self.dataset_train)
        self.dataset_test = scaler.transform(self.dataset_test)

    def __reshape(self):
        """This method reshapes the data as it has a time dependency."""

        # After 80 steps in the timestamp, it starts again at 0
        self.dataset_train = self.dataset_train.reshape(-1, 80, self.dataset_train.shape[-1])
        self.dataset_test = self.dataset_test.reshape(-1, 80, self.dataset_test.shape[-1])
        self.targets = self.targets.reshape(-1, 80)

    def get_datasets(self):
        """This method returns the training data, labels and test data."""
        # Output:
        # self.dataset_train, self.targets, self.dataset_test; numpy arrays

        return self.dataset_train, self.targets, self.dataset_test

    def get_path(self):
        """This method returns the path."""
        # Output:
        # self.path; string

        return self.path

    def save_datasets_numpy(self):
        """This method saves the data arrays to a binary file in NumPy .npy format."""
        
        np.save(self.path + "train.npy", self.dataset_train)
        np.save(self.path + "target.npy", self.targets)
        np.save(self.path + "test.npy", self.dataset_test)
        print("Data saved as NumPy files!")

    def load_dataset_numpy(self):
        """This method loads arrays from .npy files."""

        self.dataset_train = np.load(self.path + "train.npy")
        self.targets = np.load(self.path + "target.npy")
        self.dataset_test = np.load(self.path + "test.npy")
        print("Data loaded from NumPy files!")

    def write_submissions_mean(self, test_predictions):
        """This method writes the predictions from the cross-validation (the average of all predictions) into a csv file."""
        # Input:
        # test_predictions; numpy array

        # Every single fold is used. The average value is calculated and saved.
        self.submission["pressure"] = test_predictions.mean(axis = 1)    # Mean of row
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.submission.to_csv(self.path + now + "_mean_submission.csv", index = False)

    def __add_many_features(self, df):
        """This method adds many features from the internet to a DataFrame."""
        # Code from: https://www.kaggle.com/code/mohitsahal/lstm-plus-gru
        # Input:
        # df; DataFrame
        # Output:
        # df; DataFrame

        # Step 1
        df['cross']= df['u_in'] * df['u_out']
        df['cross2']= df['time_step'] * df['u_out']
        df['area'] = df['time_step'] * df['u_in']
        df['area'] = df.groupby('breath_id')['area'].cumsum()
        df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
        df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
        #print("Step-1...Completed")
        
        # Step 2
        df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
        df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
        df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
        df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
        df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
        df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
        df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
        df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
        df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
        df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
        df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
        df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
        df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
        df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
        df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
        df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
        df = df.fillna(0)
        #print("Step-2...Completed")
        
        # Step 3
        df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
        df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
        df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
        df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
        #print("Step-3...Completed")
        
        # Step 4
        df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
        df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
        df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
        df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
        df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
        df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
        df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
        df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
        #print("Step-4...Completed")
        
        # Step 5
        df['one'] = 1
        df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
        df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
        df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
        df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
        df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
        df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
        df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
        df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
        df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
        df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']
        #print("Step-5...Completed")
        
        # Step 6
        df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
        # This feature leads to errors with the Pandas version used. Therefore, it is not used.
        #df['ewm_u_in_mean'] = (df\
        #                    .groupby('breath_id')['u_in']\
        #                    .ewm(halflife=9)\
        #                    .mean()\
        #                    .reset_index(level=0,drop=True))
        df[["15_in_sum","15_in_min","15_in_max","15_in_mean"]] = (df\
                                                                .groupby('breath_id')['u_in']\
                                                                .rolling(window=15,min_periods=1)\
                                                                .agg({"15_in_sum":"sum",
                                                                        "15_in_min":"min",
                                                                        "15_in_max":"max",
                                                                        "15_in_mean":"mean"})\
                                                                .reset_index(level=0,drop=True))
        #print("Step-6...Completed")
        
        # Step 7
        df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']
        df['u_out_lagback_diff1'] = df['u_out'] - df['u_out_lag_back1']
        df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag_back2']
        df['u_out_lagback_diff2'] = df['u_out'] - df['u_out_lag_back2']
        #print("Step-7...Completed")
        
        # Step 8
        df['R'] = df['R'].astype(str)
        df['C'] = df['C'].astype(str)
        df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
        df = pd.get_dummies(df)
        #print("Step-8...Completed")
        
        return df
        

if (__name__ == "__main__"):
    
    CVPP_Datasets = VPP_Datasets(create_new = True, many_features = False)
    train, target, test = CVPP_Datasets.get_datasets()

    print(train.shape, target.shape, test.shape)
    #CVPP_Datasets.save_datasets_numpy()
    #CVPP_Datasets.write_submissions_mean(np.ones((4024000, 2)))

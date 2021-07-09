import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_csv(path, interval):
    """
    Reads the csv file in the corresponding path and returns a dataframe with all data rearranged in desired intervals 
    """

    df = pd.read_csv(path, usecols=[0,5, 6,7,8,21,24], index_col=0, parse_dates=True)
    df = df.fillna(method='ffill')
    df = df.resample(interval).mean()
    df = df.fillna(method='ffill')

    return df

def feature_and_targets(data):
    """
    It removes from the dataset the features that will be used in the prediction model and the data that must be predicted so that we can validate the model. 
    """

    data['day of the week'] = data.index.dayofweek
    data['day of the year'] = data.index.dayofyear
    data['hour of the day'] = data.index.hour
    data['minute of the hour'] = data.index.minute
    data["Consumption"] = data['T1']+data['T2']+data['T3']+data['T4']
    #data["Consumption"] = data["TGBT"]    

    features = ['day of the week','day of the year','hour of the day','minute of the hour', 'AirTemp','rh']#, 'wd', 'ws','rh', 'rain']
    labels   = ["Consumption"]
    inputs   = features + labels
    data = data[inputs]

    return data, features,labels,len(features), len(labels)

def normalize(data):
    """
    Normalizes the dataset individually for each column between -1 and 1 
    """

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)

    return data_scaled,scaler

def split_data(data, sequence_length, features, labels, test_size=0.25):
    """
    splits data to training and testing parts
    """

    #Cut the dataset into 2 parts: the first will be for training and the second for validation 
    ntest = int(round(len(data) * (1 - test_size)))
    df_train, df_test = data.iloc[:ntest], data.iloc[ntest:]

    #Separates the data between the features that will be used and the results that should be predicted 
    x_train = np.asarray(df_train[features].iloc[:-sequence_length])
    x_test = np.asarray(df_test[features].iloc[:-sequence_length])
    y_test = np.asarray(df_test[labels].iloc[sequence_length:])
    y_train = np.asarray(df_train[labels].iloc[sequence_length:])

    return x_train, x_test, y_test, y_train,df_train, df_test

def batch_generator(batch_size, sequence_length, num_features, num_labels, x, y):
    """
    Generator function for creating random batches of training-data.
    """

    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_features)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_labels)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
   
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            if len(x)<sequence_length:
                print("there will be a problem test too short", len(x))
            idx = np.random.randint(len(x) - 2*sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x[idx:idx+sequence_length]
            y_batch[i] = y[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)
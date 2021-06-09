"""
The script performs the prediction for the energy consumption of the next 24 hours in 15 minute intervals. 
The predictions are saved in the file 'NRLab_conso_fcast.csv' and the data is read from the file 'Last_day.csv' 
"""


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
The function performs the prediction and has the data and size variables as input. 

data - dataframe with stored data (consumption, temperature, humidity, etc.) 
size - dataframe size and consequently the model that will be used. 
        12 => 3 hours model
        24 => 6 hours model
        96 => 24hours model
"""
def predict(data, size):

    tf.compat.v1.reset_default_graph()
    sequence_length = int(size)  

    #Organizing the data and keeping only the columns that will be used in the model
    features = ['day of the week','day of the year','hour of the day','minute of the hour', 'AirTemp','rh']
    labels   = ['TGBT']
    inputs   = features + labels
    data = data[inputs]


    #Normalizing data between 1 and -1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)
    x_test_scaled = np.asarray(data_scaled[features])


    #First let's load meta graph and restore weights
    sess=tf.Session()   
    #####Change file path#####
    saver = tf.train.import_meta_graph('Model'+str(size)+'/my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('Model'+str(size)+'/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    outputs = graph.get_tensor_by_name("op_to_restore:0")

    y_pred = sess.run(outputs, feed_dict={x: [x_test_scaled]})
    y_test = y_pred.reshape(-1, 1)
    scaler.fit(data[labels])

    DF = pd.DataFrame(scaler.inverse_transform(y_test), index=data.iloc[0:sequence_length].index + pd.Timedelta(minutes=15*sequence_length), columns=labels)
        
    return DF

#####Change file path#####
data = pd.read_csv('~/DriveX/EMS_NRLab/Forecast_Cons/Last_day.csv', index_col=6, parse_dates=True)
data = data.resample('15Min').mean()
data = data.fillna(method='ffill')

#Adding new features so that we can use the date and time in the model
data['day of the week'] = data.index.dayofweek
data['day of the year'] = data.index.dayofyear
data['hour of the day'] = data.index.hour
data['minute of the hour'] = data.index.minute

#Prediction
Predict_3h = predict(data.iloc[-12:], 12)
Predict_6h = predict(data.iloc[-24:], 24)
Predict_1d = predict(data.iloc[-96:], 96)
frames = [Predict_3h,Predict_6h.iloc[-12:],Predict_1d.iloc[-72:]]

#####Change file path#####
Forecast = pd.concat(frames)
Forecast.to_csv('~/DriveX/EMS_NRLab/Forecast_Cons/NRLab_conso_fcast.csv', index = True)

import tensorflow as tf
import matplotlib.pyplot as plt
from Data_Processing import *
from tensorflow.keras.utils import Progbar
from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')


##Parameters that are used to create the model
sequence_length = int(60/15*24*10)      #Defines our prediction window, that is, how long will we predict 
batch_size = 5
num_neurons = 200  
learning_rate = 0.001
num_layers = 5
num_iter = 50
keep_prob = 0.2

##
data = read_csv('../Data/DrahiX_Data.csv', '15Min')
data, features,labels, num_features, num_labels = feature_and_targets(data)

data, scaler = normalize(data)
x_train, x_test, y_test, y_train,df_train, df_test= split_data(data, sequence_length, features, labels, test_size=0.25)

##
generator     = batch_generator(batch_size, sequence_length, num_features, num_labels, x_train, y_train)
testgenerator = batch_generator(batch_size, sequence_length, num_features,  num_labels,  x_test,  y_test)



################################################################################
#                                   MODEL                                      #
################################################################################
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, sequence_length, num_features], name='x')
    y = tf.placeholder(tf.float32, [None, sequence_length, num_labels], name="y") 
    
    def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

    with tf.name_scope('lstm'):
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(num_neurons, 1-keep_prob) for _ in range(num_layers)])


    outputs, current_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    stacked_rnn_output = tf.reshape(outputs, [-1, num_neurons])           #change the form into a tensor
    stacked_outputs = tf.layers.dense(stacked_rnn_output, num_labels) 
    outputs = tf.reshape(stacked_outputs, [-1, sequence_length, num_labels],name="op_to_restore")          #shape of results

    loss = tf.losses.mean_squared_error(y, outputs)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        histogram_summary = tf.summary.scalar('MSE', loss)
        merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter('./graphs', sess.graph)
        progbar = Progbar(num_iter)

        for iter in range(num_iter):
            x_batch, y_batch = next(generator)
            _current_state, l = sess.run([current_state, train], feed_dict={x: x_batch, y: y_batch})
            histogram_summary = tf.summary.histogram('My_histogram_summary', loss)
            res_sum = sess.run(merged, feed_dict={x: x_batch, y: y_batch})   
            mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
            #writer.add_summary(res_sum, iter)
            progbar.update(iter, values=[('MSE', mse)])
            
            
        saver.save(sess, 'Model'+str(sequence_length)+'/my_test_model',global_step=1000)
            

################################################################################
#                                   TEST                                       #
################################################################################

        y_pred = sess.run(outputs, feed_dict={x: [x_test[:sequence_length]]})
        y_test = y_pred.reshape(-1, num_labels)
        DF = pd.DataFrame(y_test, index=df_test.iloc[0:sequence_length].index + pd.Timedelta(minutes=15*sequence_length), columns=labels)
    

        for i in range(int(len(x_test)/sequence_length - 2)):
            y_pred = sess.run(outputs, feed_dict={x: [x_test[sequence_length*(i+1):sequence_length*(i+2)]]})
            y_test = y_pred.reshape(-1, num_labels)
    
            result = pd.DataFrame(y_test, index=df_test.iloc[sequence_length*(i+1):sequence_length*(i+2)].index + pd.Timedelta(minutes=15*sequence_length), columns=labels)
            DF = pd.concat([DF,result])

        mse = mean_squared_error(df_test["Consumption"].iloc[:len(DF)], DF["Consumption"])
        rmse = sqrt(mse)
        mae = mean_absolute_error(df_test["Consumption"].iloc[:len(DF)], DF["Consumption"])

        fig = plt.figure(figsize=(20, 5))
        df_test["Consumption"].plot()
        DF["Consumption"].plot( label=' Predicted',alpha=0.8)
        plt.title('MSE : %.3f     MAE : %.3f     RMSE : %.3f' % (mse,mae,rmse))
        plt.legend()
        plt.grid()
        plt.show()

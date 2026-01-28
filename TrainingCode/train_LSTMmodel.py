# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
import read_gauge as Gauge
import BuildModel
import time
import os
import csv

# Set the random seed
np.random.seed(50)
tf.compat.v1.set_random_seed(50)
# Suppress logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Tensorflow backend settings (multiple CPUs)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Define river/stream (use USGS number)
stream_list = ['03574500', '03575100', '02400680', '03586500', '01611500', 
               '01616500', '01666500', '01594526', '01585100', '02031000', 
               '01645000', '01643000', '01639000', '01632900', '01604500', 
               '01648000', '01661500', '01654000', '01656000', '01625000', 
               '01639500', '03432350', '03433500', '03428200', '03427500', 
               '03598000', '03599500', '03604000', '03602500', '03588500', 
               '03597590', '03469251', '03451000', '03524500', '03529500', 
               '03550000', '03497300', '03485500', '03478400', '03544970', 
               '03568933', '03439000']

#Set start and stop time for training (default)
sdate = dt.datetime(2010, 1, 5, 0)
edate = dt.datetime(2019, 4, 30, 23)
# Define validation start time
vdate = dt.datetime(2018, 8, 1, 0)

# Define Gauge correction offset (rare case where gauge is moved)
g_offset = 0.0

# Define local output directory
mod_output_dir = 'Models'

# Use early stopping
earlyStopCond = True

# Place lag in reverse order (LSTM expectation) - back from forecast time of +6hr
lag = [36, 30, 24, 18, 12, 6]

# QPE/QPF interval in hours
qpeInt = 6

# Loop through the streams
for stream in stream_list:
    print('Stream :', stream)
    start_time = time.time()
    filename = f'./GaugeData/data/GaugeData_{stream}_20100105_20230110.txt'

    if not os.path.isfile(filename):
        print('No File')
        print(filename)
        continue

    # Define river basin shapefiles
    shpfile = './USGSData/Basins'

    # Read the gauge data - pull out the hourly data (input is every 30 minutes)
    gTime, gHeight = Gauge.getDataAlt(filename, sdate, edate)
    print('Gauge Time:', gTime[0], gTime[-1])

    # Now get the Precipitation and Relative Soil Moisture Data
    # Define the MRMS csv file
    MRMSfile = f'./MRMS/MRMS_train{stream}.csv'
    # Read the csv file
    MRMSTime = []
    MRMSdata = []
    with open(MRMSfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            hour = int(row[3])
            MRMSTime.append(dt.datetime(year, month, day, hour))
            MRMSdata.append(float(row[4]))

    # Numpy arrays
    MRMSTime = np.array(MRMSTime)
    MRMSdata = np.array(MRMSdata)
    print(MRMSTime.shape, MRMSTime[0], MRMSTime[-1])
    print(MRMSdata.shape)

    # Define LIS csv file
    LISfile = f'./LIS/LIS_train{stream}.csv'
    LISTime = []
    LISdata = []
    with open(LISfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            hour = int(row[3])
            LISTime.append(dt.datetime(year, month, day, hour))
            LISdata.append([float(row[4]), float(row[5]), float(
                row[6]), float(row[7]), float(row[8])])

    # Numpy arrays
    LISTime = np.array(LISTime)
    LISdata = np.array(LISdata)
    print(LISTime.shape, LISTime[0], LISTime[-1])
    print(LISdata.shape)

    # Determine the start time from the lag (need lag data before start)
    sTime_tmp = gTime[0] + dt.timedelta(hours=max(lag))
    # Find that starting time
    ind2 = np.where(gTime == sTime_tmp)[0]
    # Make sure we find a time
    while (len(ind2) < 1):
        sTime_tmp = sTime_tmp + dt.timedelta(hours=1)
        ind2 = np.where(gTime == sTime_tmp)[0]

    # Make the valid time array that we will have clag hours worth of previous data for
    Time = gTime[ind2[0]:]
    # Get the true gauge heights
    truth1 = gHeight[ind2[0]:]

    print("Lag", "Start", "Stop")
    print(max(lag), Time[0], Time[-1])

    # Format all of the data
    # Initialized gauge height and LIS up to 48 hr before
    start_data_format = time.time()
    validData = truth1.shape
    predictors = []
    truth = []
    valpredictors = []
    valtruth = []
    missing_count = 0
    for i in range(validData[0]):
        # Calculate time of forecast
        foreTime = Time[i] + dt.timedelta(hours=max(lag))
        trueInd = np.where(gTime == foreTime)[0]
        if (len(trueInd) < 1):
            if (foreTime > vdate):
                valtruth.append(np.nan)
            else:
                truth.append(np.nan)
        else:
            if (foreTime > vdate):
                valtruth.append(gHeight[trueInd[0]])
            else:
                truth.append(gHeight[trueInd[0]])

        # Create the time stamps for the needed antecedent data
        # Gauge
        time1 = [foreTime - dt.timedelta(hours=j) for j in lag]

        # Gauge
        g_tmp = []
        # LIS SMP
        LIS_tmp = []
        # Rain
        Rain_tmp = []
        # All predictions
        pList = []
        for cTime in time1:
            # Gauge Height
            # Get the time match index for guage height
            diff = foreTime - cTime
            if (diff.seconds / 3600 > 42):
                g_tmp.append(0)
                pList.append(0)
            else:
                gInd = np.where(gTime == cTime)[0]
                if (len(gInd) < 1):
                    g_tmp.append(np.nan)
                    pList.append(np.nan)
                else:
                    g_tmp.append(gHeight[gInd[0]])
                    pList.append(gHeight[gInd[0]])

            # LIS VSM
            # Find the closest negative value
            x = [ii - cTime for ii in LISTime]
            y = [ii.days * 24 + ii.seconds / 3600. for ii in x]
            m = max(ii for ii in y if ii <= 0)
            lisInd = y.index(m)
            # Save the data in a list
            LIS_tmp.append(LISdata[lisInd, :].tolist())
            for ii in range(LISdata.shape[1]):
                pList.append(LISdata[lisInd, ii])

            # MRMS
            # Set MRMS time match start (exclude cTime because MRMS is accumulated)
            newTime1 = cTime + dt.timedelta(hours=1)
            # End MRMS time interval
            newTime2 = cTime + dt.timedelta(hours=qpeInt)
            # Get the time match index
            rainInd1 = np.where(MRMSTime == newTime1)[0]
            rainInd2 = np.where(MRMSTime == newTime2)[0]
            # Account for missing data
            niter = 0
            while (len(rainInd1) < 1):
                if (newTime1 >= newTime2):
                    break
                newTime1 = newTime1 + dt.timedelta(hours=1)
                rainInd1 = np.where(MRMSTime == newTime1)[0]
                niter += 1
            niter = 0
            while (len(rainInd2) < 1):
                if (newTime2 <= newTime1):
                    break
                newTime2 = newTime2 - dt.timedelta(hours=1)
                rainInd2 = np.where(MRMSTime == newTime2)[0]
                niter += 1
            # Create the 6hr QPF
            if (len(rainInd1) < 1 or len(rainInd2) < 1):
                missing_count += 1
                qpf = np.nan
            elif (rainInd1 == rainInd2):
                qpf = MRMSdata[rainInd1[0]]
            else:
                # Plus 1 on second index is to include the end time
                qpf = np.sum(MRMSdata[rainInd1[0]:rainInd2[0] + 1])
            # Save the data in a list
            Rain_tmp.append(qpf)
            pList.append(qpf)

        # Add to the predictor list
        if (foreTime > vdate):
            valpredictors.append(pList)
        else:
            predictors.append(pList)

    # Create numpy array
    predictors = np.array(predictors)
    truth = np.array(truth)
    valpredictors = np.array(valpredictors)
    valtruth = np.array(valtruth)

    # Combine predictors and gauge truth into one array
    # This is done to remove all times (rows) with missing data (NaNs)
    allData = np.hstack((predictors, truth.reshape(truth.shape[0], 1)))
    valallData = np.hstack((valpredictors, valtruth.reshape(valtruth.shape[0], 1)))
    # Remove the rows with missing data (NaNs)
    newData = allData[~np.isnan(allData).any(axis=1)]
    valnewData = valallData[~np.isnan(valallData).any(axis=1)]
    # np.save(f'TrainData_{stream}_snow.npy', newData)
    # np.save(f'ValData_{stream}_snow.npy', valnewData)
    newData2 = allData[np.isnan(allData).any(axis=1)]

    if (len(newData) < 1):
        print('Skipping the dataset....')
        continue

    # Scale the data
    scalerPreds = StandardScaler()
    scalerPreds.fit(newData[:, :newData.shape[1] - 1])
    scale_preds = scalerPreds.transform(newData[:, :newData.shape[1] - 1])
    valscale_preds = scalerPreds.transform(valnewData[:, :valnewData.shape[1] - 1])
    # Scale the truth
    scalerTrue = StandardScaler()
    scalerTrue.fit(newData[:, -1].reshape(-1, 1))
    scale_true = scalerTrue.transform(newData[:, -1].reshape(-1, 1))[:, 0]
    valscale_true = scalerTrue.transform(valnewData[:, -1].reshape(-1, 1))[:, 0]

    # Reshape to time dimension for LSTM
    scale_preds = scale_preds.reshape(scale_preds.shape[0], len(lag), 7)
    valscale_preds = valscale_preds.reshape(valscale_preds.shape[0], len(lag), 7)

    end_data_format = time.time() - start_data_format
    print('Data Formatting time [sec,min]: ', end_data_format, end_data_format / 60.)

    # Split the data into train/test (K-fold validation)
    start_model_train = time.time()
    num_epochs = 500
    batch_size = 512
    lr = 0.001
    optimizer = 'Adam'
    nnLayers = [2]
    nnNeurons = [512]
    activations = ['tanh']
    # Dropout
    Drop = False
    drop_per = [0.0]
    # Regularization
    regular = False
    rCoef = 0.001
    # Early stopping
    patience = 25

    # Do a grid search type interation
    # Layers
    for lay in nnLayers:
        # Neurons
        for neurons in nnNeurons:
            # Output activation functions
            for activate in activations:
                # Recurrent Dropout
                for dPer in drop_per:

                    # Build the model using all training data (input is the number of predictors)
                    model = BuildModel.LSTM(scale_preds.shape[1], scale_preds.shape[2],
                                            lays=lay,
                                            nodes=neurons, optimizer=optimizer, lr=lr,
                                            Drop=Drop, dPer=dPer, regular=regular,
                                            rCoef=rCoef,
                                            activate=activate)
                    # Train the model (silent mode : verbose=0)
                    if (earlyStopCond):
                        earlyStop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                                            verbose=2, baseline=None, restore_best_weights=True, mode='min')
                        callbacks_list = [earlyStop]
                        history = model.fit(scale_preds, scale_true, epochs=num_epochs,
                                            validation_data=(
                                                valscale_preds, valscale_true),
                                            batch_size=batch_size, callbacks=callbacks_list,
                                            verbose=2)
                    else:
                        history = model.fit(scale_preds, scale_true, epochs=num_epochs,
                                            batch_size=batch_size, verbose=2)

                    end_model_train = time.time() - start_model_train

                    print('Finished Training model for :', stream)
                    print('Layers: ', lay, 'Neurons: ', neurons, 'Activation: ', activate,
                          'Recurrent Dropout: ', dPer)

                    print('Model training time [sec,min]: ',
                          end_model_train, end_model_train / 60.)

                    # Save the model weights
                    setting_str = '_L' + \
                        str(lay) + 'N' + str(neurons).zfill(3) + 'A' + \
                        activate + 'D' + str(int(dPer * 100)).zfill(3)
                    print(setting_str)
                    model.save_weights('./' + mod_output_dir +
                                       '/model_weights_' + stream + setting_str + '.h5')
                    # Save the model architecture
                    with open('./' + mod_output_dir + '/model_architecture_' + stream + setting_str + '.json', 'w') as f:
                        f.write(model.to_json())
                    # Dump the scaler
                    joblib.dump(scalerPreds, './' + mod_output_dir +
                                '/scalerPreds_' + stream + setting_str + '.save')
                    joblib.dump(scalerTrue, './' + mod_output_dir +
                                '/scalerTrue_' + stream + setting_str + '.save')

                    end_time = time.time() - start_time
                    print('Total Training time [sec, min]: ', end_time, end_time / 60.)
                    print(' ')

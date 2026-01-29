# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import os
import tensorflow as tf
import numpy as np
import datetime as dt
import read_gauge as Gauge
import hFunc
import time
import csv

start_total = time.time()

# Suppress logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# # Define river/stream (use USGS number)
stream_list = ['01585100', '01594526', '01604500', '01611500', '01616500', '01625000',
               '01632900', '01639000', '01639500', '01643000', '01645000', '01648000',
               '01654000', '01656000', '01661500', '01666500', '02031000', '02400680', 
               '03427500', '03428200', '03432350', '03433500', '03439000', '03451000',
               '03469251', '03478400', '03485500', '03497300', '03524500', '03529500',
               '03544970', '03550000', '03568933', '03574500', '03575100', '03575830',
               '03586500', '03588500', '03597590', '03598000', '03599500', '03602500',
               '03604000']

#Need to start at 0, 6, 12, 18 for easy QPE interval
start_time = dt.datetime(2021, 1, 1, 0)
end_time = dt.datetime(2023, 5, 31, 18)

#Add LIS
addLIS = True

# Define Forecast text file Output directory
for_txt_dir = f'./output'
# Define model input directory
mod_dir = f'./Models'

# Define forecast length in hours
fLength = 168
# Define LSTM lags (hrs)
# Sum interval for MRMS QPE
qpeInt = 6
# Define the number of days
# use 5 or 1.5
lag_days = 1.5
lag = [(i + 1) * qpeInt for i in range(int(lag_days * 24 / qpeInt))][::-1]
testing = True

#Create MRMS QPE segments
time_segments = [start_time - dt.timedelta(hours=max(lag))]
while (time_segments[-1] < (end_time + dt.timedelta(hours=fLength))):
    time_segments.append(time_segments[-1] + dt.timedelta(hours=qpeInt))

# Gauge data directory
gauge_outdir = '/GaugeData'

# Get the data for each stream
data_root = './Data'
# Read the csv file
MRMSTime = []
MRMSdata = []
LISTime = []
LISdata = []
gfiles = []
for s in stream_list:
    # Get data list per stream
    LISTime.append([])
    LISdata.append([])

    # Now get the Precipitation and Relative Soil Moisture Data
    # Define the MRMS csv file
    MRMSTime_tmp = []
    MRMSdata_tmp = []
    MRMSfile = os.path.join(data_root, f'MRMS/MRMS_train{s}.csv')
    with open(MRMSfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            hour = int(row[3])
            MRMSTime_tmp.append(dt.datetime(year, month, day, hour))
            MRMSdata_tmp.append(float(row[4]))
    # Convert MRMS 1hr to 6hr now
    # Start the time on a 0, 6, 12, 18 Z time and everything will fall into place  
    MRMSdata_tmp, MRMSTime_tmp = hFunc.convert1hr_6hr_targets(np.array(MRMSdata_tmp), 
                                                              np.array(MRMSTime_tmp), 
                                                              time_segments, qpeInt)

    MRMSTime.append(MRMSTime_tmp)
    MRMSdata.append(MRMSdata_tmp)

    # Define LIS csv file
    LISfile = os.path.join(data_root, f'LIS/LIS_train{s}.csv')
    with open(LISfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            hour = int(row[3])
            LISTime[-1].append(dt.datetime(year, month, day, hour))
            LISdata[-1].append([float(row[4]), float(row[5]), float(
                row[6]), float(row[7]), float(row[8])])

    #Get filename
    gfiles.append(
        f'./GaugeData/GaugeData_{s}_20190105_20240131.txt')

# Read the gauge data - pull out the hourly data (input is every 30 minutes)
gTime, gHeight, badfile = Gauge.getData(gfiles)

forTime = start_time
while forTime <= end_time:

    print('Forecast Time: ', forTime)

    # List containing the forecast hours (x hr segments)
    forList = [(i + 1) * qpeInt for i in range(int(fLength / qpeInt))]

    # Get the forecast end time
    fTimeEnd = forTime + dt.timedelta(hours=fLength)
    # Get LSTM start time (minus 1 day)
    fTimeStart = forTime - dt.timedelta(hours=max(lag))

    start_prediction = time.time()
    # Make predictions
    # for i, stream in enumerate(LIS_streams):
    for i, stream in enumerate(stream_list):
        print('Stream ID: ', stream)
        # Check for strea data
        if (len(gHeight[i]) < 1):
            print('No data found in gauge file')
            continue

        # Get the average precipitation across the basin
        MRMSavg = np.array(MRMSdata[i])

        # Run the model with MRMS forcing and LIS reanalysis
        MRMSpredictions, MRMSpredTime, segFault = hFunc.make_predictions(mod_dir, stream,
                                                                         forTime, forList,
                                                                         lag, qpeInt,
                                                                         gTime[i], gHeight[i],
                                                                         np.array(
                                                                             LISTime[i]),
                                                                         np.array(
                                                                             LISdata[i]),
                                                                         np.array(
                                                                             MRMSTime[i]),
                                                                         MRMSavg,
                                                                         np.array(
                                                                             MRMSTime[i]),
                                                                         MRMSavg,
                                                                         testing=testing,
                                                                         addLIS=addLIS)

        # Create output
        hFunc.createTXT_MRMS(stream, for_txt_dir, forTime, MRMSpredictions, MRMSpredTime,
                             gHeight[i], gTime[i], np.array(MRMSTime[i]), MRMSavg)

    total_prediction = time.time() - start_prediction
    print('Total Prediction Time [sec, min]: ', total_prediction, total_prediction / 60.)
    total_time = time.time() - start_total
    print('Total Time [sec, min]: ', total_time, total_time / 60.)
    print('Done')

    forTime = forTime + dt.timedelta(hours=6)

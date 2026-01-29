# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import numpy as np
import os
import datetime as dt
from tensorflow.keras import models
import tensorflow as tf
import joblib
import csv
import json


def convert1hr_6hr_targets(data, date, time_targets, qpeInt):

    data_out = []
    date_out = []
    for tar in time_targets:
        # Accumulated over the 6 hr window
        segtime1 = tar + dt.timedelta(hours=1)
        segtime2 = tar + dt.timedelta(hours=qpeInt)
        # Get the time match index
        ind1 = np.where(date == segtime1)[0]
        ind2 = np.where(date == segtime2)[0]
        # Account for missing data
        while (len(ind1) < 1):
            if (segtime1 >= segtime2):
                break
            segtime1 = segtime1 + dt.timedelta(hours=1)
            ind1 = np.where(date == segtime1)[0]
        while (len(ind2) < 1):
            if (segtime2 <= segtime1):
                break
            segtime2 = segtime2 - dt.timedelta(hours=1)
            ind2 = np.where(date == segtime2)[0]
        # Create the 6hr QPF
        if (len(ind1) < 1 or len(ind2) < 1):
            # Missing data
            #data_tmp = np.nan
            data_tmp = 0.0
        elif (ind1 == ind2):
            # Only 1 file in interval
            data_tmp = data[ind1[0]]
        else:
            # Plus 1 on second index is to include the end time
            data_tmp = np.sum(data[ind1[0]:ind2[0] + 1])
        data_out.append(data_tmp)
        # The make_prediction function assumes the QPE time is the
        # end of the accumulation window...so add the QPE interval
        date_out.append(tar + dt.timedelta(hours=qpeInt))

    return np.array(data_out), np.array(date_out)

# This river info
def get_info(river):

    # Let hardcode a load file
    with open('./river_info.json', 'r') as f:
        flood_dict = json.load(f)

    # Get  info
    nws_id = flood_dict[river]['nws_id'].upper()
    wfo_id = flood_dict[river]['wfo_id']

    return nws_id, wfo_id

def make_predictions(mod_dir, stream, forTime, forList, lag, qpeInt, gTime, gHeight,
                     LISTime, LISdata, QPFTime, QPFData, QPETime, QPEData,
                     testing=True, skip_load=False,
                     model=None, scalerPreds=None, scalerTrue=None, addLIS=True,
                     addGauge=True, addSnow=False, manual_scale=False):

    # Load the model architecture
    if (testing and not skip_load):
        json_file = open(os.path.join(mod_dir, 'model_architecture_' +
                                      stream + '_L2N512AtanhD000.json'))
        loaded_model_json = json_file.read()
        json_file.close()
        model = models.model_from_json(loaded_model_json)
        # Load weights
        model.load_weights(os.path.join(mod_dir, 'model_weights_' +
                                        stream + '_L2N512AtanhD000.h5'))
        # Load the scaler
        if (not manual_scale):
            scalerPreds = joblib.load(os.path.join(
                mod_dir, 'scalerPreds_' + stream + '_L2N512AtanhD000.save'))
            scalerTrue = joblib.load(os.path.join(
                mod_dir, 'scalerTrue_' + stream + '_L2N512AtanhD000.save'))
        else:
            with open(os.path.join(
                    mod_dir, 'scalerPreds_' + stream + '.json'), 'r') as f:
                scalerPreds = json.load(f)
            with open(os.path.join(
                    mod_dir, 'scalerTrue_' + stream + '.json'), 'r') as f:
                scalerTrue = json.load(f)
    elif (not skip_load):
        json_file = open(os.path.join(mod_dir, 'model_architecture_' +
                                      stream + '.json'))
        loaded_model_json = json_file.read()
        json_file.close()
        model = models.model_from_json(loaded_model_json)
        # Load weights
        model.load_weights(os.path.join(mod_dir, 'model_weights_' +
                                        stream + '.h5'))
        # Load the scaler
        scalerPreds = joblib.load(os.path.join(
            mod_dir, 'scalerPreds_' + stream + '.save'))
        scalerTrue = joblib.load(os.path.join(
            mod_dir, 'scalerTrue_' + stream + '.save'))

    # Compile the data
    # Loop through the forecast hours (6,12,18,24,..,120)
    predTime = [forTime]
    # Get the starting gauge height - for plotting
    indG = np.where(gTime == forTime)[0]
    if (len(indG) < 1):
        error_time = forTime - dt.timedelta(hours=1)
        for i in range(6):
            indG = np.where(gTime == error_time)[0]
            if (len(indG) > 0):
                predictions = [gHeight[indG[0]]]
                break
            error_time = error_time - dt.timedelta(hours=1)
        if (len(indG) < 1):
            print('Error1...missing gauge data...')
            return [], [], True
    else:
        predictions = [gHeight[indG[0]]]
    missing_count = 0
    for cfor in forList:
        # Current forecast time
        cforTime = forTime + dt.timedelta(hours=cfor)
        # print(cforTime)
        predTime.append(cforTime)
        # Create the time stamps for the needed antecedent data
        time1 = [cforTime - dt.timedelta(hours=j) for j in lag]

        # Gauge
        g_tmp = []
        # LIS
        LIS_tmp = []
        # MRMS
        Rain_tmp = []
        # All predictions
        pList = []
        for cTime in time1:
            # Get the time match index
            if (addGauge):
                # Determine if the gauge value is known, otherwise use the previous predictions
                if (cTime <= forTime):
                    gInd = np.where(gTime == cTime)[0]
                    if (len(gInd) < 1):
                        error_time = cTime - dt.timedelta(hours=1)
                        for i in range(6):
                            gInd = np.where(gTime == error_time)[0]
                            # print(gInd)
                            if (len(gInd) > 0):
                                g_tmp.append(gHeight[gInd[0]])
                                pList.append(gHeight[gInd[0]])
                                break
                            error_time = error_time - dt.timedelta(hours=1)
                        if (len(gInd) < 1):
                            print('Error2...missing gauge data...')
                            return [], [], True
                    else:
                        g_tmp.append(gHeight[gInd[0]])
                        pList.append(gHeight[gInd[0]])
                else:
                    gInd = np.where(np.array(predTime) == cTime)[0]
                    if (len(gInd) < 1):
                        g_tmp.append(np.nan)
                        pList.append(np.nan)
                    else:
                        g_tmp.append(predictions[gInd[0]])
                        pList.append(predictions[gInd[0]])

            # LIS VSM
            # Find the closest negative value
            if (addLIS):
                x = [ii - cTime for ii in LISTime]
                y = [ii.days * 24 + ii.seconds / 3600. for ii in x]
                m = max(ii for ii in y if ii <= 0)
                lisInd = y.index(m)
                # Save the data in a list
                LIS_tmp.append(LISdata[lisInd, :].tolist())
                for ii in range(LISdata.shape[1]):
                    pList.append(LISdata[lisInd, ii])

            # End Precip time interval (precip is accumulated)
            newTime = cTime + dt.timedelta(hours=qpeInt)
            # Get the time match index
            if (cTime < forTime):
                # Use MRMS QPE
                rainInd = np.where(QPETime == newTime)[0]
                # Account if missing than set precip to 0
                if (len(rainInd) < 1):
                    missing_count += 1
                    qpf = 0.0  # np.nan use 0 so the prediction can continue
                else:
                    # use the matched time
                    qpf = QPEData[rainInd[0]]
            else:
                # Use a QPF product
                rainInd = np.where(QPFTime == newTime)[0]
                # Account if missing than set precip to 0
                if (len(rainInd) < 1):
                    missing_count += 1
                    qpf = 0.0  # np.nan use 0 so the prediction can continue
                else:
                    # use the matched time
                    qpf = QPFData[rainInd[0]]

            # Save the data in a list
            Rain_tmp.append(qpf)
            pList.append(qpf)

        # Create the predictor array
        predictors = np.array(pList)
        # Scale the data
        # Reshape as input needs to be a 2D array but this is a single sample
        if (not manual_scale):
            scale_preds = scalerPreds.transform(predictors.reshape(1, -1))
        else:
            pred_means = np.array(scalerPreds['mean'])
            pred_stds = np.array(scalerPreds['scale'])
            scale_preds = (predictors - pred_means) / pred_stds
            # Reshape to fit the reshape below
            scale_preds = scale_preds.reshape(1, -1)
        # Reshape to time dimension for LSTM
        if (addLIS and addGauge and addSnow):
            scale_preds = scale_preds.reshape(scale_preds.shape[0], len(lag), 11)
        elif (addLIS and addGauge):
            scale_preds = scale_preds.reshape(scale_preds.shape[0], len(lag), 7)
        elif (addLIS and not addGauge):
            scale_preds = scale_preds.reshape(scale_preds.shape[0], len(lag), 6)
        else:
            scale_preds = scale_preds.reshape(scale_preds.shape[0], len(lag), 2)

        # Use the forest's predict method on the test data
        ScalePrediction = model.predict(scale_preds, verbose=0)
        # Unscale the prediction
        if (not manual_scale):
            prediction = scalerTrue.inverse_transform(ScalePrediction)
        else:
            true_means = np.array(scalerTrue['mean'])
            true_stds = np.array(scalerTrue['scale'])
            prediction = (ScalePrediction * true_stds) + true_means
        predictions.append(prediction[0][0])

    # Return a numpy array of the predictions
    return np.array(predictions), predTime, False


def createTXT_MRMS(stream, outtxt_dir, forTime, MRMSprediction, MRMSpredTime,
                   GaugeHeight, GaugeTime, MRMSTime, MRMSavg, name_add=None,
                   num_div=None):

    # Get Stream Info
    nws_id, wfo_id = get_info(stream)

    if (name_add is None):
        outputNameTxt = f"{forTime.strftime('%Y%m%d_%H00_')}{wfo_id}_{nws_id}.txt"
    else:
        outputNameTxt = f"{name_add}_{forTime.strftime('%Y%m%d_%H00_')}{wfo_id}_{nws_id}.txt"

    # Get the gauge data at the forecast times
    new_GHeight = []
    for time in MRMSpredTime:
        try:
            #print(GaugeHeight[GaugeTime == time][0])
            new_GHeight.append(GaugeHeight[GaugeTime == time][0])
        except:
            new_GHeight.append(-999.)

    # Get the MRMS at the forecast times
    new_MRMSavg = []
    for time in MRMSpredTime:
        try:
            #print(GaugeHeight[GaugeTime == time][0])
            ind = np.where(MRMSTime == time)[0]
            if (num_div is None):
                new_MRMSavg.append(MRMSavg[ind[0]])
            else:
                tmp = []
                for i in range(num_div):
                    tmp.append(MRMSavg[ind[0], i])
                new_MRMSavg.append(tmp)
        except:
            new_MRMSavg.append(-999.)

    create_tab_output_MRMS(outtxt_dir, outputNameTxt, MRMSpredTime,
                           MRMSprediction, new_GHeight, new_MRMSavg, 
                           num_div=num_div)

            
def create_tab_output_MRMS(outdir, outputName, timestamp, model1, gauge, QPE, 
                           num_div=None):

    with open(os.path.join(outdir, outputName), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        header = ['Date[UTC]       ', '  LSTM (MRMS)', '   Gauge Obs']
        if (num_div is not None):
            for i in range(num_div):
                header.append('   QPE[in]')
            writer.writerow(header)
            for i in range(len(timestamp)):
                row = [timestamp[i].strftime('%Y-%m-%d_%H:%M'),
                                "{:12.6}".format(model1[i]), "{:12.6}".format(gauge[i])]
                for ii in range(num_div):
                    row.append("{:12.6}".format(QPE[i][ii]))
                writer.writerow(row)
        else:
            header.append('   QPE[in]')
            writer.writerow(header)
            for i in range(len(timestamp)):
                writer.writerow([timestamp[i].strftime('%Y-%m-%d_%H:%M'),
                                "{:12.6}".format(model1[i]), "{:12.6}".format(gauge[i]),
                                "{:12.6}".format(QPE[i])])

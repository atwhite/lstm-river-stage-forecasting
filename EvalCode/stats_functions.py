# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import numpy as np
import datetime as dt
import glob
import csv
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# This reads in the rating curve data
def get_USGSRating(stream, root, shift_adj=True):

    # Get filenames
    filenames = glob.glob(os.path.join(root, 'RatingCurve*.txt'))

    # Get stream names
    stream_ids = [f.split('_')[1] for f in filenames]

    # Read the data
    hgt = []
    shift = []
    flow = []
    # Find the correct file
    try:
        ind = stream_ids.index(stream)
        with open(filenames[ind]) as f:
            reader = csv.reader(f, delimiter='\t')
            # Read the rows
            for row in reader:
                # Skip header junk
                try:
                    data1 = float(row[0])
                    data2 = float(row[1])
                    data3 = float(row[2])
                    hgt.append(data1)
                    shift.append(data2)
                    flow.append(data3)
                except:
                    continue
    except:
        print('Rating Curve not found...', stream)

    # return np.array(hgt), np.array(shift), np.array(flow)
    if shift_adj:
        shift_hgt = np.array(hgt) + np.array(shift)
        #Since the shift is not uniform we need to get unique values for interpolation
        shift_hgt, ind_uni = np.unique(shift_hgt, return_index = True)
        flow = np.array(flow)[ind_uni]
        return shift_hgt, flow
    else:
        return np.array(hgt), np.array(flow)


# This function interpolates the data between stage and flow using
# the usgs ratings curves
def rate_interp(indata, hgt, flow, hgt_to_flow=True):

    # Loop through input length
    if not isinstance(indata, list):
        indata = [indata]
    interp = []
    for i in range(len(indata)):
        # Check for non rating curve info
        if (len(hgt) < 1):
            interp.append(np.nan)
            continue
        # Check for NaN input
        if (indata[i] != indata[i]):
            interp.append(np.nan)
            continue

        # Convert height to flow or flow to height
        if (hgt_to_flow):
            # Check that we have valid data
            if (indata[i] >= max(hgt)):
                interp.append(np.nan)
            elif (indata[i] <= min(hgt)):
                interp.append(np.nan)
            else:
                # Take the difference between rating height and shifted input
                diff = indata[i] - hgt
                ind = np.argmin(diff[diff >= 0])
                # Low flow divide by zero protection
                if (flow[ind] <= 0.000001):
                    interp.append(0.0)
                else:
                    interp.append(flow[ind] * np.exp(np.log(flow[ind + 1] / flow[ind]) *
                                                     ((indata[i] - hgt[ind]) / (hgt[ind + 1] - hgt[ind]))))
        else:
            # Check that we have valid data
            if (indata[i] >= max(flow)):
                interp.append(np.nan)
            elif (indata[i] <= min(flow)):
                interp.append(np.nan)
            else:
                # Take the difference between rating height and shifted input
                diff = indata[i] - flow
                ind = np.argmin(diff[diff >= 0])
                interp.append(hgt[ind] + (hgt[ind + 1] - hgt[ind]) *
                              (np.log(indata[i] / flow[ind]) / np.log(flow[ind + 1] / flow[ind])))

    return np.squeeze(interp)


# Function to read all data
def readData(filename, sdate, edate, hgt_rate, flow_rate, 
             filter_pred_times=False, skip_months=None,
             return_stats=False):

    # Use csv reader to read the file
    Time_tmp = []
    disCharge_tmp = []
    gHeight_tmp = []
    alt_read = False
    # Check the first line
    with open(filename, 'r') as f:
        line = f.readline()
        if ('This was generated with Python' in line):
            alt_read = True
    if (alt_read):
        # Use csv reader to read the file
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            # Skip the first line
            next(reader)
            for row in reader:
                date_string = dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                if (date_string >= sdate and date_string <= edate):
                    Time_tmp.append(date_string)
                    gHeight_tmp.append(float(row[1]))
                    dis_tmp = rate_interp(float(row[1]), hgt_rate, flow_rate)
                    disCharge_tmp.append(dis_tmp)
    else:
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            # Skip the first 29 lines
            # Skip the first 29 lines
            check_height = []
            check_discharge = []
            for i in range(50):
                line = next(reader)
                check_height.append(
                    [True if 'Gage height' in ll else False for ll in line])
                check_discharge.append(
                    [True if 'Discharge' in ll else False for ll in line])
                # Break out of the header loop
                if ('#' not in line[0]):
                    break
            # This will set the index for gauge height (some files have discharge in them)
            # Probably should just download the files without discharge
            if (not any(item == True for sublist in check_discharge for item in sublist)):
                gauge_ind = 4
            elif (len([True for sublist in check_discharge for item in sublist if item == True]) > 1):
                # Two sensors for discharge
                gauge_ind = 8
            else:
                gauge_ind = 6
            for row in reader:
                if (row[0] == 'USGS' and row[6]):
                    pass
                else:
                    continue
                # Create datetime object
                tString = row[2]
                year = tString[0:4]
                month = tString[5:7]
                day = tString[8:10]
                hour = tString[11:13]
                minute = tString[14:16]
                # Get a time offset
                tZone = row[3]
                if (tZone == 'CST'):
                    offset = 6
                elif (tZone == 'CDT'):
                    offset = 5
                elif (tZone == 'EST'):
                    offset = 5
                elif (tZone == 'EDT'):
                    offset = 4
                dum = (dt.datetime(int(year), int(month), int(day), int(hour), int(minute))
                       + dt.timedelta(hours=offset))
                if (row[4] != 'Ice' and row[gauge_ind] != 'Ice' and row[4] != 'Eqp' and row[gauge_ind] != 'Eqp'
                        and row[4] != 'Mnt' and row[gauge_ind] != 'Mnt' and row[4] != ''
                        and row[gauge_ind] != ''):
                    if (dum >= sdate and dum <= edate):
                        Time_tmp.append(dum)
                        disCharge_tmp.append(float(row[4]))
                        gHeight_tmp.append(float(row[gauge_ind]))

    # Only use on the hour data
    marray = [i.minute for i in Time_tmp]
    hind = np.where(np.array(marray) == 0)[0]
    # New arrays
    Time = np.array(Time_tmp)[hind]
    disCharge = np.array(disCharge_tmp)[hind]
    gHeight = np.array(gHeight_tmp)[hind]
    # Only get data at the prediction times
    if (filter_pred_times):
        valid = [0, 6, 12, 18]
        marray_tst = [i for i, t in enumerate(Time) if t.hour in valid]
        Time = Time[marray_tst]
        gHeight = gHeight[marray_tst]
        disCharge = disCharge[marray_tst]
    if (skip_months is not None):
        marray_tst = [i for i, t in enumerate(Time) if t.month not in skip_months]
        Time = Time[marray_tst]
        gHeight = gHeight[marray_tst]
        disCharge = disCharge[marray_tst]

    if return_stats:
        # Get some percentiles
        gH_per = np.percentile(
            gHeight, [30, 50, 75, 90, 95, 97.5, 98, 98.5, 99, 99.5])
        gD_per = np.percentile(
            disCharge, [30, 50, 75, 90, 95, 97.5, 98, 98.5, 99, 99.5])
        # Get some mean values
        mean_obs_hgt = np.mean(gHeight)
        mean_obs_flow = np.mean(disCharge)
        return Time, gHeight, disCharge, gH_per, gD_per, mean_obs_hgt, mean_obs_flow
    else:
        return Time, gHeight, disCharge


# Read all available txt forecasts
def read_all_forecasts(fnames, data_types, hgt_rate, flow_rate):

    # Define what data types need to be coverted
    convert = ['FOR', 'MRMS', 'w/o LIS', 'w/o Gauge']
    data_flow = []
    data_height = []
    dates = []
    # Loop through the files
    for a, b in zip(fnames, data_types):
        data_tmp_flow = []
        data_tmp_hgt = []
        dates.append([])
        with open(a, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            # Skip header
            next(reader)
            for row in reader:
                date = dt.datetime.strptime(row[0], '%Y-%m-%d_%H:%M')
                dates[-1].append(date)
                if (b == 'FOR'):
                    data_tmp_hgt.append([float(row[1]), float(row[2]), float(row[3])])
                    data_tmp_flow.append([rate_interp(float(row[1]), hgt_rate, flow_rate), 
                                      rate_interp(float(row[2]), hgt_rate, flow_rate), 
                                      rate_interp(float(row[3]), hgt_rate, flow_rate)])
                elif (b == 'MRMS' or b == 'w/o LIS' or b == 'w/o Gauge'):
                    data_tmp_hgt.append(float(row[1]))
                    data_tmp_flow.append(rate_interp(float(row[1]), hgt_rate, flow_rate))
                elif ('NWM' in b):
                    print(b)
                    data_tmp_flow.append(float(row[1]))
                    data_tmp_hgt.append(float(row[2]))
        if (b == 'FOR'):
            data_tmp_flow = np.array(data_tmp_flow)
            data_tmp_hgt = np.array(data_tmp_hgt)
            for i in range(3):
                data_flow.append(data_tmp_flow[:, i].tolist())
                data_height.append(data_tmp_hgt[:, i].tolist())
        else:
            data_flow.append(data_tmp_flow)
            data_height.append(data_tmp_hgt)

    # Check to make sure all of the dates are the same length
    inlen = len(dates[0])
    for i in range(1, len(dates)):
        if (len(dates[i]) != inlen):
            print('Date lengths are different...')
            print(i, inlen, len(dates[i]))
            exit()

    return np.squeeze(data_height).tolist(), np.squeeze(data_flow).tolist(), dates[0]


# This function creates the plots of interest
def multi_plot(x, values, labels, ylabel, title, num_force, outname, outdir):

    # Lets use seaborn style
    plt.style.use('seaborn')
    colors = ['g', 'r', 'b', 'c', 'm', 'k', 'tab:brown', 'tab:gray']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['s', 'X', 'd', 'h', 'o', '*', '+', '|']
    fig, ax = plt.subplots(nrows=len(values), ncols=1, sharex='col', sharey='row', 
                            figsize=(10, 10))
    #Align y-label
    box = dict(facecolor='white', edgecolor='white', pad=10, alpha=0.2)
    #Loop through plots, then data
    for i, value in enumerate(values):
        for j in range(num_force):
            # Bias
            ax[i].plot(x, value[:, j], color=colors[j], linestyle=linestyles[j], 
                    marker=markers[j], label=labels[j], linewidth=1.2, markersize=5)

        ax[i].set_xlim(x[0], x[-1])
        if (i == len(values)-1):
            ax[i].set_xlabel('Forecast Hour', fontsize=12)
        ax[i].set_ylabel(ylabel[i], fontsize=12 , bbox=box)

        ax[i].set_xticks(x[::2])
        ax[i].set_xticklabels(x[::2])  # , fontsize = 10)

        ax[i].tick_params(axis='both', labelsize=12)

        ax[i].set_title(title[i], fontsize=14)
        ax[i].legend(loc='best', ncol=3, fontsize=12)

    fig.subplots_adjust(left=0.10, right=0.95, top=0.93,
                        bottom=0.07, wspace=0.1)
    plt.savefig(os.path.join(outdir, outname))
    plt.close()

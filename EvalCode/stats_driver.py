# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import os
import glob
import json
import numpy as np
import datetime as dt
import pandas as pd
import stats_functions as func
import hydroeval as he


# Define the streams we want
stream_list = ['03574500', '03575100', '02400680', '03586500', '01611500', 
               '01616500', '01666500', '01594526', '01585100', '02031000', 
               '01645000', '01643000', '01639000', '01632900', '01604500', 
               '01648000', '01661500', '01654000', '01656000', '01625000', 
               '01639500', '03432350', '03433500', '03428200', '03427500', 
               '03598000', '03599500', '03604000', '03602500', '03588500', 
               '03597590', '03469251', '03451000', '03524500', '03529500', 
               '03550000', '03497300', '03485500', '03478400', '03544970', 
               '03568933', '03439000']

#Define output directory for figure
outdir = './stats'

#Start and End dates
sdate = dt.datetime(2021, 3, 1, 0)
edate = dt.datetime(2023, 4, 30, 18) 

#Define months to use
seasonal = False
season = 'all'
pick_months = {'DJF': [12, 1, 2], 'MAM': [3,  4,  5], 
               'JJA': [ 6, 7, 8], 'SON': [9, 10, 11]}

# Define months to skip if need (i.e. summer)
# Blank list equals no skipping of months
skip_months = []
if (seasonal):
    #Skip months
    other_months = {'DJF': [3, 4, 5, 6, 7, 8, 9, 10, 11], 
                    'MAM': [1, 2, 6, 7, 8, 9, 10, 11, 12], 
                    'JJA': [1, 2, 3, 4, 5, 9, 10, 11, 12], 
                    'SON': [1, 2, 3, 4, 5, 6, 7, 8, 12]}
    skip_months = other_months[season]

#Set data list
ftypes = ['MRMS', 'FOR', 'NWM1', 'NWM2', 'NWM3', 
          'NWM4', 'NWM5', 'NWM6', 'NWM7']

#Forecast interval
for_int = 6
# Define the length of forecast (in hours)
forecast_length = 168
# Calculate forecast hours
fh = np.arange((forecast_length) / for_int + 1, dtype=np.int32) * for_int
#Needed/Requested Forecast times
ftreq = pd.date_range(sdate, edate, freq='6h').to_pydatetime() 

#Create the dataframe
df = {'Time': ftreq}
#Create dataframe
df = pd.DataFrame(df)

#Dictionary with stream info
stream_dict = '../RunCode/river_info.json'
with open(stream_dict, 'r') as f:
    river_dict = json.load(f)
#Define obs root dir
obs_dir = './ReanalysisData'
nws_ids = []
obs_files = []
for stream in stream_list:
    nws_ids.append(river_dict[stream]['nws_id'].upper())
    obs_files.append(os.path.join(obs_dir, f'GaugeData_{stream}_20190105_20240131.txt'))

# Rating curve root directory
rate_dir = './RatingCurves'

#Define input directories
indirs = []
num_files = len(ftypes)
for f in ftypes:
    if (f == 'FOR'):
        indirs.append('./ForecastFiles')
    elif (f == 'MRMS'):
        indirs.append('./MRMSReanalysis')
    elif (f == 'NWM1'):
        indirs.append('./NWMMediumMem1_txt')
    elif (f == 'NWM2'):
        indirs.append('./NWMMediumMem2_txt')
    elif (f == 'NWM3'):
        indirs.append('./NWMMediumMem3_txt')
    elif (f == 'NWM4'):
        indirs.append('./NWMMediumMem4_txt')
    elif (f == 'NWM5'):
        indirs.append('./NWMMediumMem5_txt')
    elif (f == 'NWM6'):
        indirs.append('./NWMMediumMem6_txt')
    elif (f == 'NWM7'):
        indirs.append('./NWMMediumMem7_txt')
    else:
        print(f'Ftype not found: {f}')
        exit()

#Get files
stats_flow = []
stats_hgt = []
stats_flow_peak = []
stats_hgt_peak = []
for nws, stream, obs_file in zip(nws_ids, stream_list, obs_files):
    print(nws)
    #Loop through different forecast/reanalysis/NWM models
    files = []
    ftimes = []
    for i, indir in enumerate(indirs):
        print(indir)
        ftmp = sorted(glob.glob(os.path.join(indir, f'*{nws}.txt')))
        #Make file names without RFC/WFO area tag to get unique files
        alt_files = [f.replace(f"_{os.path.basename(f).split('_')[2]}", '')
                for f in ftmp]
        #Get the unique elements
        a, inds = np.unique(alt_files, return_index=True)
        #Updated file list
        files.append(np.array(ftmp)[inds].tolist())
        #Get file times
        ftimes.append([dt.datetime.strptime(os.path.basename(f).split('_')[0] + 
                            os.path.basename(f).split('_')[1], '%Y%m%d%H%M') 
                            for f in files[-1]])
        if (i == 0):
            df = {'Time': ftimes[-1], 'Files0': files[-1]}
            df = pd.DataFrame(df)
        else:
            #Merge new to old
            df_tmp = {'Time': ftimes[-1], f'Files{str(i)}': files[-1]}
            df_tmp = pd.DataFrame(df_tmp)
            df = pd.merge(df, df_tmp, on='Time')

    dims = df.shape
    keys = df.keys()
    if (seasonal):
        seasonal_months = pick_months[season]
        df = df.loc[(df.Time.dt.month == seasonal_months[0]) | 
                        (df.Time.dt.month == seasonal_months[1]) |
                        (df.Time.dt.month == seasonal_months[2])].reset_index(drop=True)
        dims = df.shape

    # Read the ratings data
    hgt_rate, flow_rate = func.get_USGSRating(stream, rate_dir)
    if (len(hgt_rate) < 1):
        print('No Rating...', nws, stream)
        continue

    # Get the gauge data to calculate peak statistics over the time frame
    print('Obs File: ', obs_file)
    gTime_all, gHeight_all, gDis_all, gH_per, gD_per, mean_obs_hgt, mean_obs_flow \
        = func.readData(obs_file, sdate, edate, hgt_rate, flow_rate,
                        filter_pred_times=True, skip_months=skip_months, 
                        return_stats=True)
    # Convert the Time into a list for faster indexing
    peak_hgt = gH_per[6]
    peak_flow = gD_per[6]
    print(len(gHeight_all), len(gDis_all))
    print('Mean: ', mean_obs_hgt, mean_obs_flow)
    print('Top 2%: ', peak_hgt, peak_flow)

    #Create Numpy Arrays
    obs_data = np.concatenate((gTime_all[:, None], gHeight_all[:, None], 
                               gDis_all[:, None]), axis=1)

    #Get the forecast in a data frame with data in forecast hr format
    f_height = []
    f_flow = []
    f_times = []
    f_obs = []
    #For easy indexing
    gTime_all_list = gTime_all.tolist()
    #for t, f in zip(ftimes, files):
    for ii in range(dims[0]):
        t = df['Time'][ii]
        #print(t)
        if (t < sdate or t > edate):
            continue
        #Read all of the data
        f = []
        for ff in range(num_files):
            f.append(df[f'Files{str(ff)}'][ii])
        pred_height, pred_flow, pred_times = func.read_all_forecasts(f, 
                                                                     ftypes, 
                                                                     hgt_rate, 
                                                                     flow_rate)
        #Need a full forecasts
        if (len(pred_times) < (forecast_length / 6) + 1):
            continue
        #print(len(pred_height), len(pred_flow), len(pred_times))
        f_height.append(pred_height)
        f_flow.append(pred_flow)
        f_times.append(pred_times[0])
        #Get the obs for the prediction times
        f_obs_tmp = []
        for tt in pred_times:
            #print(tt)
            #Need exception block in case we dont find the time
            try:
                ind_dis = gTime_all_list.index(tt)
                f_obs_tmp.append([gHeight_all[ind_dis], gDis_all[ind_dis]])
            except:
                f_obs_tmp.append([np.nan, np.nan])
        f_obs.append(f_obs_tmp)

    #Create Numpy Arrays
    f_height = np.array(f_height)
    f_flow = np.array(f_flow)
    f_obs= np.array(f_obs)
    f_times = np.array(f_times)

    #Now loop through forecast hours
    stats_flow_tmp = []
    stats_hgt_tmp = []
    stats_flow_peak_tmp = []
    stats_hgt_peak_tmp = []
    for i in range(len(fh)):
        print(f'Forecast Hour: {fh[i]}')
        #Get the forecast hour data
        pred_hgt = f_height[:, :, i]
        obs_hgt = f_obs[:, i, 0]
        pred_flow = f_flow[:, :, i]
        obs_flow = f_obs[:, i, 1]

        #Get the index for the peak flows
        ind_hgt = np.where(obs_hgt >= peak_hgt)[0]
        ind_flow = np.where(obs_flow >= peak_flow)[0]
        #Get the peak arrays
        pred_hgt_peak = pred_hgt[ind_hgt, :]
        obs_hgt_peak = obs_hgt[ind_hgt]
        pred_flow_peak = pred_flow[ind_flow, :]
        obs_flow_peak = obs_flow[ind_flow]

        #Remove NaNs
        hgt = np.concatenate((pred_hgt, obs_hgt[:, None]), axis=1)
        hgt_new = hgt[~np.isnan(hgt).any(axis=1)]   
        hgt_peak = np.concatenate((pred_hgt_peak, obs_hgt_peak[:, None]), axis=1)
        hgt_new_peak = hgt_peak[~np.isnan(hgt_peak).any(axis=1)]  
        nse_hgt = he.evaluator(he.nse, hgt_new[:, :-1], hgt_new[:, -1])
        kge_hgt, r_hgt, alpha_hgt, beta_hgt = he.evaluator(he.kge, hgt_new[:, :-1], 
                                                           hgt_new[:, -1])
        #Negative predictors to flip the sign 
        # Code does Obs - Prediction but should be Prediction - Obs
        pbias_hgt = he.evaluator(he.pbias, hgt_new[:, :-1], hgt_new[:, -1]) * -1
        nse_hgt_peak = he.evaluator(he.nse, hgt_new_peak[:, :-1], hgt_new_peak[:, -1])
        kge_hgt_peak, r_hgt_peak, alpha_hgt_peak, beta_hgt_peak = \
            he.evaluator(he.kge, hgt_new_peak[:, :-1], hgt_new_peak[:, -1])
        pbias_hgt_peak = he.evaluator(he.pbias, hgt_new_peak[:, :-1], 
                                      hgt_new_peak[:, -1]) * -1

        #Remove NaNs
        flow = np.concatenate((pred_flow, obs_flow[:, None]), axis=1)
        flow_new = flow[~np.isnan(flow).any(axis=1)]
        flow_peak = np.concatenate((pred_flow_peak, obs_flow_peak[:, None]), axis=1)
        flow_new_peak = flow_peak[~np.isnan(flow_peak).any(axis=1)] 

        nse_flow = he.evaluator(he.nse, flow_new[:, :-1], flow_new[:, -1])
        kge_flow, r_flow, alpha_flow, beta_flow = he.evaluator(he.kge, flow_new[:, :-1], 
                                                               flow_new[:, -1])
        #Negative predictors to flip the sign 
        # Code does Obs - Prediction but should be Prediction - Obs
        pbias_flow = he.evaluator(he.pbias, flow_new[:, :-1], flow_new[:, -1]) * -1
        nse_flow_peak = he.evaluator(he.nse, flow_new_peak[:, :-1], 
                                     flow_new_peak[:, -1])
        kge_flow_peak, r_flow_peak, alpha_flow_peak, beta_flow_peak = \
            he.evaluator(he.kge, flow_new_peak[:, :-1], flow_new_peak[:, -1])
        pbias_flow_peak = he.evaluator(he.pbias, flow_new_peak[:, :-1], 
                                      flow_new_peak[:, -1]) * -1

        #Save this in forecast order
        stats_flow_tmp.append([nse_flow, kge_flow, pbias_flow])
        stats_hgt_tmp.append([nse_hgt, kge_hgt, pbias_hgt])
        stats_flow_peak_tmp.append([nse_flow_peak, kge_flow_peak, pbias_flow_peak])
        stats_hgt_peak_tmp.append([nse_hgt_peak, kge_hgt_peak, pbias_hgt_peak])

    stats_flow.append(stats_flow_tmp)
    stats_hgt.append(stats_hgt_tmp)    
    stats_flow_peak.append(stats_flow_peak_tmp)
    stats_hgt_peak.append(stats_hgt_peak_tmp)   

    print(np.array(stats_flow).shape)
    print(np.array(stats_hgt).shape)
    print(np.array(stats_flow_peak).shape)
    print(np.array(stats_hgt_peak).shape)

#Calculate the averages
stats_flow = np.array(stats_flow)
stats_hgt = np.array(stats_hgt)
stats_flow_peak = np.array(stats_flow_peak)
stats_hgt_peak = np.array(stats_hgt_peak)

stats_all_flow = np.array([np.nanmedian(stats_flow, axis=0), np.nanstd(stats_flow, axis=0),
                  np.nanmax(stats_flow, axis=0), np.nanmin(stats_flow, axis=0),
                  np.nanmean(stats_flow, axis=0)])
stats_all_hgt = np.array([np.nanmedian(stats_hgt, axis=0), np.nanstd(stats_hgt, axis=0),
                  np.nanmax(stats_hgt, axis=0), np.nanmin(stats_hgt, axis=0),
                  np.nanmean(stats_hgt, axis=0)])
stats_all_flow_peak = np.array([np.nanmedian(stats_flow_peak, axis=0), 
                                np.nanstd(stats_flow_peak, axis=0),
                                np.nanmax(stats_flow_peak, axis=0), 
                                np.nanmin(stats_flow_peak, axis=0),
                                np.nanmean(stats_flow_peak, axis=0)])
stats_all_hgt_peak = np.array([np.nanmedian(stats_hgt_peak, axis=0), 
                               np.nanstd(stats_hgt_peak, axis=0),
                               np.nanmax(stats_hgt_peak, axis=0), 
                               np.nanmin(stats_hgt_peak, axis=0),
                               np.nanmean(stats_hgt_peak, axis=0)])

#I think the end shape of these are:
# average type stat, forecast hour, stat type, model forcing 
print(stats_all_flow.shape)
print(stats_all_hgt.shape)
print(stats_all_flow_peak.shape)
print(stats_all_hgt_peak.shape)

#Calculate the number of forecasts
num_force = int(stats_all_flow.shape[-1])
print('Number of forcing: ', num_force)

# Forecast loop
print('NSE')
for i in range(len(fh)):
    print('Forecast Hour: ', fh[i])
    print('     ', '     ', '        MRMS', '         WPC', '         GFS', 
          '         NBM', '         NMW')
    print('     ', 'ID   ', "{:>12.5f}".format(stats_all_flow[4, i, 0, 0]), 
          "{:>12.5f}".format(stats_all_flow[4, i, 0, 1]), 
          "{:>12.5f}".format(stats_all_flow[4, i, 0, 2]), 
          "{:>12.5f}".format(stats_all_flow[4, i, 0, 3]),
          "{:>12.5f}".format(stats_all_flow[4, i, 0, 4]))
    for j in range(len(nws_ids)):
        print('     ', nws_ids[j], "{:>12.5f}".format(stats_flow[j, i, 0, 0]), 
              "{:>12.5f}".format(stats_flow[j, i, 0, 1]), 
              "{:>12.5f}".format(stats_flow[j, i, 0, 2]), 
              "{:>12.5f}".format(stats_flow[j, i, 0, 3]), 
              "{:>12.5f}".format(stats_flow[j, i, 0, 4]))
    print(' ')

print('KGE')
for i in range(len(fh)):
    print('Forecast Hour: ', fh[i])
    print('     ', '     ', '        MRMS', '         WPC', '         GFS', 
          '         NBM', '         NMW')
    print('     ', 'ID   ', "{:>12.5f}".format(stats_all_flow[4, i, 1, 0]), 
          "{:>12.5f}".format(stats_all_flow[4, i, 1, 1]), 
          "{:>12.5f}".format(stats_all_flow[4, i, 1, 2]), 
          "{:>12.5f}".format(stats_all_flow[4, i, 1, 3]),
          "{:>12.5f}".format(stats_all_flow[4, i, 1, 4]))
    for j in range(len(nws_ids)):
        print('     ', nws_ids[j], "{:>12.5f}".format(stats_flow[j, i, 1, 0]), 
              "{:>12.5f}".format(stats_flow[j, i, 1, 1]), 
              "{:>12.5f}".format(stats_flow[j, i, 1, 2]), 
              "{:>12.5f}".format(stats_flow[j, i, 1, 3]), 
              "{:>12.5f}".format(stats_flow[j, i, 1, 4]))
    print(' ')


print('Percent Peak Bias')
for i in range(len(fh)):
    print('Forecast Hour: ', fh[i])
    print('     ', '     ', '        MRMS', '         WPC', '         GFS', 
          '         NBM', '         NMW')
    print('     ', 'ID   ', "{:>12.5f}".format(stats_all_flow_peak[4, i, 2, 0]), 
          "{:>12.5f}".format(stats_all_flow_peak[4, i, 2, 1]), 
          "{:>12.5f}".format(stats_all_flow_peak[4, i, 2, 2]), 
          "{:>12.5f}".format(stats_all_flow_peak[4, i, 2, 3]),
          "{:>12.5f}".format(stats_all_flow_peak[4, i, 2, 4]))
    for j in range(len(nws_ids)):
        print('     ', nws_ids[j], "{:>12.5f}".format(stats_flow_peak[j, i, 2, 0]), 
              "{:>12.5f}".format(stats_flow_peak[j, i, 2, 1]), 
              "{:>12.5f}".format(stats_flow_peak[j, i, 2, 2]), 
              "{:>12.5f}".format(stats_flow_peak[j, i, 2, 3]), 
              "{:>12.5f}".format(stats_flow_peak[j, i, 2, 4]))
    print(' ')

#Save data
np.save(os.path.join(outdir, f'Stats_flow_{season}.npy'), stats_all_flow)
np.save(os.path.join(outdir, f'Stats_Peakflow_{season}.npy'), stats_all_flow_peak)

#Plot
func.multi_plot(fh, 
                [stats_all_flow[4, :, 0], stats_all_flow[4, :, 1], stats_all_flow_peak[4, :, 2]], 
                ['LSTM (MRMS QPE)', 'LSTM (WPC QPF)', 'LSTM (GFS QPF)', 'LSTM (NBM QPF)',
                 'NWM Member 2'], 
                ['NSE', 'KGE', 'Bias [%]'],
                ['Mean Nash-Sutcliffe Efficiency', 'Mean Kling-Gupta Efficiency',
                'Mean Peak Bias'], num_force, f'All_Mean_stats_flow_{season}.png',
                 outdir)

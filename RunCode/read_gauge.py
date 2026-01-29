# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import numpy as np
import datetime as dt
import csv


def getData(filenames):

    # Use csv reader to read the file
    Time = []
    gHeight = []
    badfile = []
    count = -1
    for filename in filenames:
        badflag = False
        alt_read = False
        count += 1
        print(count, filename)
        Time_tmp = []
        gHeight_tmp = []
        # Check the first line
        with open(filename, 'r') as f:
            line = f.readline()
            if ('This was generated with Python' in line):
                alt_read = True
        if (alt_read):
            print('Alt read')
            # Use csv reader to read the file
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                # Skip the first line
                next(reader)
                for row in reader:
                    date_string = dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                    Time_tmp.append(date_string)
                    gHeight_tmp.append(float(row[1]))
        else:
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                # Skip the first 29 lines
                check_height = []
                check_discharge = []
                # Header can be variable but this is a good start
                for i in range(50):
                    # This is for a blank file
                    try:
                        line = next(reader)
                    except:
                        badfile.append(count)
                        badflag = True
                        break
                    check_height.append(
                        [True if 'Gage height' in ll else False for ll in line])
                    check_discharge.append(
                        [True if 'Discharge' in ll else False for ll in line])
                    if (i == 0 and '#  No sites found matching all criteria' in line):
                        badfile.append(count)
                        badflag = True
                        break
                    # Break out of the header loop
                    if ('#' not in line[0]):
                        break
                # Check if we need to go to the next file
                if (badflag):
                    continue
                # This will skip the gauge if Gage height is not in the dataset
                if (not any(item == True for sublist in check_height for item in sublist)):
                    badfile.append(count)
                    continue
                # This will set the index for gauge height (some files have discharge in them)
                # Probably should just download the files without discharge
                if (not any(item == True for sublist in check_discharge for item in sublist)):
                    gauge_ind = 4
                else:
                    gauge_ind = 6
                for row in reader:
                    if (row[0] == 'USGS' and row[gauge_ind]):
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
                    try:
                        gHeight_tmp.append(float(row[gauge_ind]))
                        Time_tmp.append(dum)
                    except:
                        pass

        # Only use on the hour data
        marray = [i.minute for i in Time_tmp]
        hind = np.where(np.array(marray) == 0)[0]
        # New arrays
        Time.append(np.array(Time_tmp)[hind])
        gHeight.append(np.array(gHeight_tmp)[hind])

    return Time, gHeight, np.array(badfile)

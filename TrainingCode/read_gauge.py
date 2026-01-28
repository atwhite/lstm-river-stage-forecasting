# __author__ = "Andrew White"
# __version__ = "1.0.0"
# __status__ = "Stable"
# Copyright (c) 2024, Andrew White (UAH)
# All rights reserved.

import numpy as np
import datetime as dt
import csv
import pandas as pd


def getData(filename, sdate, edate, g_offset):

    # Use csv reader to read the file
    Time = []
    # disCharge = []
    gHeight = []
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
                Time.append(date_string)
                gHeight.append(float(row[1]))
    else:
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            # Deal with header, get some info
            check_height = []
            check_discharge = []
            check_order = True
            for i in range(25):
                line = next(reader)
                # Look for what data is in the file
                # 00060 = Discharge
                # 00065 = Gage Height
                if ('00060' in line[0]):
                    check_discharge.append(True)
                if ('00065' in line[0]):
                    check_height.append(True)
                    # Check for Primary/Secondary order
                    #print('Primary' in line[0], 'Secondary' in line[0], check_order)
                    if ('Secondary' in line[0] and check_order):
                        check_order = False
                        offset = 2  # 2 because we have to account for qual flag
                    elif (not check_order):
                        pass
                    else:
                        check_order = False
                        offset = 0

                # Break out of the header loop
                if ('#' not in line[0]):
                    break
            # This will set the index for gauge height (some files have discharge in them)
            # Probably should just download the files without discharge
            # First check for height data
            if (len(check_height) < 1):
                print('Check if height data exists in file...')
                print(filename)
                return None, None
            # Offset accounts for order of gauge height sensors (primary vs secondary)
            if (len(check_discharge) == 1):
                # 1 discharge sensor
                gauge_ind = 6 + offset
            elif (len(check_height) > 1):
                # 2 discharge sensors
                gauge_ind = 8 + offset
            else:
                # No discharge data
                gauge_ind = 4 + offset

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
                if (dum < sdate or dum > edate):
                    continue
                if (len(check_discharge) < 1):
                    if (row[4] != 'Ice' and row[4] != 'Eqp' and row[4] != 'Mnt'):
                        Time.append(dum)
                        gHeight.append(float(row[gauge_ind]))
                else:
                    if (row[4] != 'Ice' and row[6] != 'Ice' and row[4] != 'Eqp' and row[6] != 'Eqp'
                            and row[4] != 'Mnt' and row[6] != 'Mnt'):
                        Time.append(dum)
                        gHeight.append(float(row[gauge_ind]))

    # Only use on the hour data
    marray = [i.minute for i in Time]
    hind = np.where(np.array(marray) == 0)[0]
    # New arrays
    Time = np.array(Time)[hind]
    # disCharge.append(np.array(disCharge_tmp)[hind])
    gHeight = np.array(gHeight)[hind] + g_offset
    return Time, gHeight

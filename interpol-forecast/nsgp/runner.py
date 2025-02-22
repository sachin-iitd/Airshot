import os
import sys
from config import Config
from dates import dates_usa

interpol = 1	# 0 is forecasting mode

if len(sys.argv) > 1: 
    interpol = int(sys.argv[1])

mode_t = ['AB','AC'][interpol]
mode_c = ['AB','AC'][interpol]
mode_p = ['CD','D'][interpol]
locid = 4
nTrainDays = 3 if 'A' in mode_t else 0

model_name = 'nsgp'
optim_name = 'ad'
c_fold = int(sys.argv[2]) if len(sys.argv)>2 else 0
locid = int(sys.argv[3]) if len(sys.argv)>3 else locid
node = 'gpu1'
nsgp_iters = 40
gp_iters = 50
restarts = 5
div = 4
sampling = 'uni' # cont, nn, uni
Xcols = '@'.join(['longitude', 'latitude', 'delta_t'])
kernel = 'rbf' # Order: RBF, M32
time_kernel = 'local_per' # Order RBF, loc_per

def xform_day(day):
    arr = [0, 30, 61]
    w = 0 if day <= 30 else 1 if day <= 61 else 2
    mon = ['2020-11-', '2020-12-', '2021-01-'][w]
    date = mon + '{:02d}'.format(day - arr[w])
    return date

def xform_kolkata(day):
    arr = [0, 31]
    w = 0 if day <= 31 else 1
    mon = ['2023-12-', '2024-01-'][w]
    date = mon + '{:02d}'.format(day - arr[w])
    return date

def create_cmd(date):
    cmd = ' '.join(['python run.py', model_name, optim_name, str(c_fold), node, str(nsgp_iters),str(gp_iters),
                    str(restarts), str(div), sampling, Xcols, kernel, time_kernel, mode_t, mode_c, mode_p, str(locid), ','.join(date)])
    return cmd

def check_break():
    if os.path.exists('break'):
        exit(1)

def proc_canada(locid):
    if locid == 1:
        dates = ['2015-02-12', '2015-02-13', '2015-03-19', '2015-03-25', '2015-03-27', '2015-04-01', '2015-04-10', '2015-04-15',
         '2015-04-17', '2015-04-21', '2015-04-24', '2015-05-12',  '2015-05-14', '2015-05-20', '2015-05-22', '2015-09-10', '2015-09-23', '2015-11-23']
    else:
        dates = ['canada_20{:02d}'.format(i) for i in range(6,17)]
    for day in range(nTrainDays, len(dates)):
        date = []
        for i in range(nTrainDays,-1,-1):
            date.append(dates[day-i])
        cmd = create_cmd(date, locid)
        print(cmd)
        os.system(cmd)
        check_break()

def proc_delhi():
    nTestStartDay = 15
    nTotalDays = [15,91][1]
    for day in range(nTestStartDay, nTotalDays+1):
        date = []
        for i in range(nTrainDays,-1,-1):
            date.append(xform_day(day-i))
        cmd = create_cmd(date)
        print(cmd)
        os.system(cmd)
        check_break()

def proc_kolkata():
    for nTestStartDay,nTotalDays in [(nTrainDays+1,31+6),(31+8+nTrainDays,62)]:
        for day in range(nTestStartDay, nTotalDays+1):
            date = []
            for i in range(nTrainDays,-1,-1):
                date.append(xform_kolkata(day-i))
            cmd = create_cmd(date)
            print(cmd)
            os.system(cmd)
            check_break()

def proc_usa(country):
    dates = dates_usa
    for day in range(nTrainDays, len(dates)):
        date = []
        for i in range(nTrainDays, -1, -1):
            date.append(dates[day - i])
        cmd = create_cmd(date, country)
        print(cmd)
        os.system(cmd)
        check_break()


if locid < 0:
    proc_delhi()
    proc_canada(1)
    proc_canada(2)
    proc_usa(3)
elif locid == 4:
    proc_kolkata()
elif locid == 0:
    proc_delhi()
elif locid == 3:
    proc_usa(3)
else:
    proc_canada(locid)

import os
import copy
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, DateFormatter, HourLocator
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xg


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 30}
matplotlib.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams.update({"text.usetex" : True})


datadir = 'colocation-data/'
IDs = ['VP', 'CP', 'BN', 'VP0']
tscol,tscol2 = "time","time"
pm = 'pm2_5'


def to_dt(time_string):
    return pd.to_datetime(time_string).tz_localize('UTC').tz_convert('Asia/Kolkata')

def get_grid_data(datafile):
    D = pd.read_csv(os.path.join(datadir,datafile+'.csv'), parse_dates = [tscol])
    return D


def ll_str(lat, lon):
    return '{:.2f}_{:.2f}'.format(lat, lon)



# Learn calibration model

intercept = False
xcols,ycol = ((1,2,3),4)
date, date2 = 10, 20
tm1 = datetime.datetime(2023, 12, date)
tm2 = datetime.datetime(2023, 12, date2+1)
mkm = 2
nalgos = 5
sz,szleg,lw = 18,15,2

def proc(train,test,algo=0,reg=None,typ=None,ww=None):
    if len(train) == 0 or len(test) == 0:
        return

    x, y, tx, ty = train.values[:,xcols], train.values[:,ycol], test.values[:,xcols], test.values[:,ycol]

    if algo==0:
        name = 'Linear'
        model = LinearRegression(fit_intercept=intercept)
    elif algo==1:
        name = 'RF'
        model = RandomForestRegressor(random_state=5)
    elif algo==2:
        name = 'GradBoost'
        model = GradientBoostingRegressor(random_state=5)
    elif algo == 3:
        name = 'XGBoost'
        model = xg.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
    elif algo==4:
        name = 'MLP'
        model = MLPRegressor(random_state=5, hidden_layer_sizes=[10,50,10], activation='relu',
                             max_iter=1000, learning_rate_init=0.01)

    if reg is None:
        reg = model.fit(x,y)
        if algo==0:
           fmt = 'Coefs: ' + ' '.join(['{:.2f}'] * (len(xcols)))
           print(fmt.format(*reg.coef_))
        return reg, name, None

    py = reg.predict(tx)
    r = rmse(ty,py)
    return r, name, py

def calib_model():
    MM = []
    d = get_grid_data(IDs[-1])
    print('time', d[tscol2].min(), d[tscol2].max())
    assert d.tc.isna().sum() == 0
    for algo in range(nalgos):
        print('Learning model for algo',algo)
        M, name, _ = proc(d, d, algo=algo)
        MM.append((M,name))
    return MM

def calib_plot(idx, MM):
    d = get_grid_data(IDs[idx])

    figsize=(2.7 * 2.4, 1.8 * 3)
    plt.figure(idx, figsize=figsize)
    plt.plot(d.time, d.pm, label='CAAQMS')
    plt.plot(d.time, d.pm2_5, label='LowCost {:.02f}'.format(rmse(d.pm2_5, d.pm)), lw=lw)
    for algo in range(nalgos):
        M, name = MM[algo]
        r, _, d2 = proc(d, d, algo, M)
        plt.plot(d.time, d2, label=name + ' {:.02f}'.format(r), lw=lw)
    plt.xticks(rotation = 30)
    A = IDs[idx]
    if idx:
        plt.title('Apply Calibration Model to {} {}'.format('Colocation',A), fontsize=sz)
    else:
        plt.title('Learn Calibration Model for Colocation {}'.format(A), fontsize=sz)
    plt.xlabel('Hourly time in Dec {} - {}, 2023'.format(date, date2), size=sz)
    plt.ylabel('PM$_{2.5}$', size=sz)
    plt.tick_params(axis='both', labelsize=sz)
    plt.ylim(bottom=20, top=400)
    plt.legend(fontsize=szleg)
    plt.tight_layout()
    plt.savefig('calib{}.pdf'.format(A), format="pdf")
    plt.show(block=False)

print('Calib Start')
MM = calib_model()
print('Calib Done')

for idx in range(len(IDs)-1):
    calib_plot(idx, MM)


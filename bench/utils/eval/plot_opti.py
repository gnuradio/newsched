#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.style.use('grcon.mplrc')

my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def conf_int(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    if (n < 2) or (se == 0):
        return np.nan
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return h

################################################################
## Line Plot (per # of pipes)
################################################################
d = pd.read_csv('../perf-data/opti.csv')
t = d.groupby(['stages', 'config']).agg({'time': [np.mean, np.var, conf_int]}).reset_index()
t = t.groupby(['config'])

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=.192, left=.11, top=.99, right=.98)

for (config), g in t:
    map_config = {'default': 'Default Configuration', 'opti': 'Manual Optimizations', 'perf': 'Scheduler Emulation'}
    map_linestyle = {'default': '-', 'opti': ':', 'perf': '--'}
    label = map_config[config]
    ls = map_linestyle[config]

    plt.errorbar(g['stages'], g[('time', 'mean')], yerr=g[('time', 'conf_int')], label=label, ls=ls)

plt.setp(ax.get_yticklabels(), rotation=90, va="center")
ax.set_xlabel('Number of Stages')
ax.set_ylabel('Time (in s)')
ax.set_ylim([0, 29])

handles, labels = ax.get_legend_handles_labels()
handles = [x[0] for x in handles]
ax.legend(handles, labels, handlelength=2.95)

plt.savefig('opti.pdf')

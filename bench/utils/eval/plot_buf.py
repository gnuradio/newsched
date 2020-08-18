#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy.stats
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
## Line Plot GR 3.8 vs Yolo
################################################################
yolo = pd.read_csv('../perf-data/buf.csv')
yolo = yolo[yolo['rt'] == 'rt']
yolo = yolo.groupby(['stages', 'rt']).agg({'time': [np.mean, np.var, conf_int]}).reset_index()
yolo['build'] = 'Yolo'

gr = pd.read_csv('/home/basti/src/gr-sched-maint/utils/perf-data/buf.csv')
gr = gr[gr['rt'] == 'rt']
gr = gr.groupby(['stages', 'rt']).agg({'time': [np.mean, np.var, conf_int]}).reset_index()
gr['build'] = 'GNU\,Radio 3.8'

t = pd.concat([yolo, gr])
t = t.groupby(['rt', 'build'])

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=.192, left=.115, top=.99, right=.97)

for (rt, build), g in t:
    ax.errorbar(g['stages']**2, g[('time', 'mean')], yerr=g[('time', 'conf_int')], label=f'{build}')

plt.setp(ax.get_yticklabels(), rotation=90, va="center")
ax.set_xlabel('\#\,Pipes $\\times$ \#\,Stages')
ax.set_ylabel('Time (in s)')

handles, labels = ax.get_legend_handles_labels()
handles = [x[0] for x in handles]
ax.legend(handles, labels, handlelength=2.95, title='Build')

plt.savefig('buf_diff.pdf')
plt.close('all')

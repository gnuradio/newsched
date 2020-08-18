#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.style.use('grcon.mplrc')

def conf_int(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    if (n < 2) or (se == 0):
        return np.nan
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return h

################################################################
## Line Plot (# of stages)
################################################################
d = pd.read_csv('../perf-data/msg.csv')
t = d.groupby(['prio', 'burst_size', 'stages']).agg({'time': [np.mean, np.var, conf_int]})
t = t.reset_index()
t = t.groupby(['prio', 'burst_size'])

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=.192, left=.115, top=.99, right=.97)

for (prio, burst_size), g in t:
    ls = {0: '-', 2000: '--', 4000: ':'}
    plt.errorbar(g['stages']**2, g[('time', 'mean')]*1000, yerr=g[('time', 'conf_int')]*1000, label=str(burst_size), ls=ls[burst_size])

plt.setp(ax.get_yticklabels(), rotation=90, va="center")
ax.set_xlabel('\#\,Pipes $\\times$ \#\,Stages')
ax.set_ylabel('Time (in ms)')

handles, labels = ax.get_legend_handles_labels()
handles = [x[0] for x in handles]
ax.legend(handles, labels, title='Burst Size', handlelength=2.95)

fig.savefig('msg.pdf')
plt.close('all')
#plt.show()


################################################################
## Line Plot (# of stages) DIFF GR vs YOLO
################################################################
d = pd.read_csv('../perf-data/msg.csv')
yolo = d.groupby(['prio', 'burst_size', 'stages']).agg({'time': [np.mean, np.var, conf_int]}).reset_index()
yolo['build'] = 'Yolo'

d = pd.read_csv('/home/basti/src/gr-sched-maint/utils/perf-data/msg.csv')
d = d[d['prio'] == 'rt']
d = d[d['burst_size'] == 1000]
gr = d.groupby(['prio', 'burst_size', 'stages']).agg({'time': [np.mean, np.var, conf_int]}).reset_index()
gr['build'] = 'GNU Radio 3.8'

t = pd.concat([gr, yolo]).reset_index()
t = t.groupby(['build', 'prio', 'burst_size'])

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=.192, left=.115, top=.99, right=.97)

for (build, prio, burst_size), g in t:
    plt.errorbar(g['stages']**2, g[('time', 'mean')], yerr=g[('time', 'conf_int')], label=str(build))

plt.setp(ax.get_yticklabels(), rotation=90, va="center")
ax.set_xlabel('\#\,Pipes $\\times$ \#\,Stages')
ax.set_ylabel('Time (in s)')

handles, labels = ax.get_legend_handles_labels()
handles = [x[0] for x in handles]
ax.legend(handles, labels, title='Build', handlelength=2.95)

fig.savefig('msg_diff.pdf')
plt.close('all')
#plt.show()

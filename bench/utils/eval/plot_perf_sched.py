#!/usr/bin/env python3

from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
# plt.style.use('grcon.mplrc')

my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def map_threads(t):
    if t.startswith('copy'):
        return 'copy'
    elif t.startswith('null_sink'):
        return 'sink'
    elif t.startswith('null_source'):
        return 'source'
    elif t.startswith('inter'):
        return 'interleave'
    elif t.startswith('head'):
        return 'head'
    else:
        return 'flowgraph'

def map_colors(t):
    if t.startswith('copy'):
        return my_colors[0]
    elif t.startswith('null_sink'):
        return my_colors[1]
    elif t.startswith('null_source'):
        return my_colors[2]
    elif t.startswith('head'):
        return my_colors[3]
    else:
        return my_colors[4]


#########################################################################
#### Process Migrations
#########################################################################
d = pd.read_csv('../migrations_rt.csv')
d = d[d['thread'].str.match('copy*|null_source*|null_sink*|interleav*|run_buf_*|head*')]

d['time'] = d['time'] - d['time'].min()
# ignore hyper-thread migrations
d = d[np.abs((d['cpu_from'] - d['cpu_to'])) > 1]

map_cpu = {2:1, 3:1, 6:2, 7:2}
d['cpu_from'] = d['cpu_from'].apply(lambda x: map_cpu[x])
d['cpu_to'] = d['cpu_to'].apply(lambda x: map_cpu[x])

t = d.groupby(['thread'])

fig, ax = plt.subplots(len(t), 1, figsize=(4.2, 7), sharey=True, sharex=True)
fig.subplots_adjust(bottom=.07, left=.21, top=.99, right=.95)

for i, (thread, g) in enumerate(t):

    g = g.reset_index(drop=True)
    times = g['time']
    cpu_to = g['cpu_to']
    cpu_from = g['cpu_from']

    ax[i].step(g['time'], cpu_from, label=thread)

    for x in range(0, len(times)-1):
        ax[i].fill_between([times.iloc[x], times.iloc[x+1]], [cpu_to.iloc[x], cpu_to.iloc[x]], color=my_colors[cpu_to.iloc[x]])

    ax[i].set_ylabel(thread.replace('_', '\_'), rotation='horizontal', ha='right', va='top')
    ax[i].get_xaxis().set_visible(False)

ax[i].get_xaxis().set_visible(True)
ax[i].set_xlabel('Time (in s)')
ax[i].set_yticks([])
ax[i].set_xlim([0.06, 0.08])

fig.savefig('migrations_rt.pdf')

#########################################################################
#### Process Migrations Cumulative
#########################################################################
n = pd.read_csv('../migrations.csv')
n = n[n['thread'].str.match('copy*|null_source*|null_sink*|interleav*|run_buf_*|head*')]
n['time'] = n['time'] - n['time'].min()
# ignore hyper-thread migrations
n = n[np.abs((n['cpu_from'] - n['cpu_to'])) > 1]

r = pd.read_csv('../migrations_rt.csv')
r = r[r['thread'].str.match('copy*|null_source*|null_sink*|interleav*|run_buf_*|head*')]
r['time'] = r['time'] - r['time'].min()
# ignore hyper-thread migrations
r = r[np.abs((r['cpu_from'] - r['cpu_to'])) > 1]

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=.19, left=.155, top=.99, right=.99)

ax.plot(n['time'], np.array([1]*len(n['time'])).cumsum(), label='Normal')
ax.plot(r['time'], np.array([1]*len(r['time'])).cumsum(), label='Real-Time')

ax.set_ylabel('CPU Migrations')
ax.set_xlabel('Time (in s)')
ax.legend()

fig.savefig('migrations_cum.pdf')

#########################################################################
#### Thread interruptions (cumulative normal vs rt)
#########################################################################
n = pd.read_csv('../resched.csv')
n = n[n['thread'].str.match('copy*|null_source*|null_sink*|interleav*|run_buf_*|head*')]
n['time'] = n['time'] - n['time'].min()
n = n[n['state'] == 'R']

r = pd.read_csv('../resched_rt.csv')
r = r[r['thread'].str.match('copy*|null_source*|null_sink*|interleav*|run_buf_*|head*')]
r['time'] = r['time'] - r['time'].min()
r = r[r['state'] == 'R']

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=.19, left=.13, top=.96, right=.99)

ax.semilogy(n['time'], np.array([1]*len(n['time'])).cumsum(), label='Normal')
ax.semilogy(r['time'], np.array([1]*len(r['time'])).cumsum(), label='Real-Time', ls=':')

ax.set_ylabel('Thread Interruptions')
ax.set_xlabel('Time (in s)')
ax.legend(title='Priority', handlelength=2.95)

fig.savefig('resched_cum.pdf')


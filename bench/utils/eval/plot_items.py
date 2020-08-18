#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('grcon.mplrc')

my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

n = pd.read_csv('../items_default.csv')
r = pd.read_csv('../items_realtime.csv')

fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.3))
fig.subplots_adjust(bottom=.362, left=.113, top=.983, right=.99)

w = .4

plt.bar(n['bin']-w/2, n['value']/1000.0, w, color='white', edgecolor=my_colors[0], hatch='.....', label='Normal')
plt.bar(r['bin']+w/2, r['value']/1000.0, w, color='white', edgecolor=my_colors[1], hatch='/////', label='Real-Time')

plt.setp(ax.get_yticklabels(), rotation=90, va="center")

ax.set_xlim(1-1.5*w, 14+w*1.5)
ax.set_ylim(0, 4300)
ax.set_xlabel('Produced Samples per Call')
ax.set_ylabel('Occurrences (in 1k)')

ax.set_xticks([1, 3, 5, 7, 9, 11, 13])
plt.setp(ax.get_xticklabels(), rotation=45, ha= 'center', va="top")
ax.set_xticklabels(['[%d,%d)' % (2**x, 2**(x+1)) for x in ax.get_xticks()])


ax.legend(title='Priority', handlelength=2.95)

plt.savefig('item_distribution.pdf')
plt.close('all')


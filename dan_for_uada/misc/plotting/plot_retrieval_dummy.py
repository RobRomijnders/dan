import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from collections import OrderedDict
from os.path import join
style.use('fivethirtyeight')


def maybe_flip(data):
    if np.min(data) < 0.1:
        return data
    else:
        return 1.0 - data


R = OrderedDict()

R['1not-aligned'] = 0.95*np.ones((49,))
R['2better-aligned'] = 0.8*np.ones((49,))
R['3best-aligned'] = 0.65*np.ones((49,))
# R['3gta5-custombatch'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fullgta-2/extract/retrievals.csv', delimiter=',')
# R['4uda-1'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fulluda-1/extract/retrievals.csv',
#                            delimiter=',')
keyname2legendname = {'1not-aligned':       f'Mediocre alignment',
                      '2better-aligned':    f'Better   alignment',
                      '3best-aligned':      f'Best     alignment'}


num_points = len(list(R.items())[0][1])
range_points = list(range(1, num_points+1))


R = {key: maybe_flip(value)*range_points for key, value in R.items()}
colors = ['k--', 'y-.', 'g-.', 'b:', 'w']

fig = plt.figure(num=None, figsize=(8, 8))
ax = plt.gca()
for i, (name, r) in enumerate(sorted(R.items())):
    plt.plot(range_points, r, colors[i], label=keyname2legendname[name])
plt.plot(range_points, 0.5*np.array(range_points), 'm', label='Perfect  alignment')
plt.legend(prop={'family': 'monospace'})
plt.xlabel('Number of neighbours', fontdict={'family': 'monospace'})
plt.ylabel('Average number of source neighbours', fontdict={'family': 'monospace'})
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')
# plt.show()
fig.savefig(join('/hdd/dropbox/Dropbox/grad/results/retrieval_curves', 'retrieval_dummy_pres.pdf'), dpi=1500, format='pdf', transparent=True)
#
# if experiment == 'pp':
#     fig.savefig(join('/hdd/dropbox/Dropbox/grad/results/retrieval_curves', 'retrieval_pp_batch.pdf'), dpi=1500, format='pdf', transparent=True)
# else:
#     fig.savefig(join('/hdd/dropbox/Dropbox/grad/results/retrieval_curves', 'retrieval_uada1.pdf'), dpi=1500, format='pdf', transparent=True)
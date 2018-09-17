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

# R['gta5-single'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0308/fullgta5-in2-source/extract_sample/retrievals.csv', delimiter=',')
# R['UADA'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0308/fulluda-in2-source/extract_sample/retrievals.csv', delimiter=',')
# R['gta5++'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0419_pp_experiment/fullgta5++/extract/retrievals.csv', delimiter=',')
# R['SDA'] = np.genfromtxt('/hdd/logs_overnight_training/older/overnight_0123/fullSDA/extract_sample/retrievals.csv', delimiter=',')
# R['comb'] = np.genfromtxt('/hdd/logs_overnight_training/older/overnight_0119/combined/extract_sample/retrievals.csv', delimiter=',')
# R['gta5-custombatch'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0403/fullgta5-ema/extract/retrievals.csv', delimiter=',')
# R['gta5-batch'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fullgta-2/extract/retrievals2.csv', delimiter=',')

experiment = 'pp'

if experiment != 'pp':
    R['gta5-1'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fullgta-2/extract/retrievals.csv', delimiter=',')
    R['uda-1'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fulluda-1/extract/retrievals.csv', delimiter=',')
    keyname2legendname = {'gta5-1': 'Single domain',
                          'uda-1': 'UADA'}
else:
    norm_layer_bn = 'Batch norm'
    norm_layer_dan = 'DAN'
    method_multi = 'Multi-domain'
    method_single = 'Single-domain'
    method_uada = 'UADA'
    R['2gta5++'] = np.genfromtxt(
        '/hdd/logs_overnight_training/newer/overnight_0419_pp_experiment/fullgta5++/extract/retrievals.csv',
        delimiter=',')
    R['1gta5-batch'] = np.genfromtxt(
        '/hdd/logs_overnight_training/newer/overnight_0509/fullgta-2/extract/retrievals2.csv', delimiter=',')
    R['3gta5-custombatch'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fullgta-2/extract/retrievals.csv', delimiter=',')
    R['4uda-1'] = np.genfromtxt('/hdd/logs_overnight_training/newer/overnight_0509/fulluda-1/extract/retrievals.csv',
                               delimiter=',')
    keyname2legendname = {'2gta5++':           f'{method_multi:15s}: Batch norm',
                          '3gta5-custombatch': f'{method_multi:15s}: DAN',
                          '1gta5-batch':       f'{method_single:15s}: Batch norm',
                          '4uda-1':            f'{method_uada:15s}: DAN'}


num_points = len(list(R.items())[0][1])
range_points = list(range(1, num_points+1))


R = {key: maybe_flip(value)*range_points for key, value in R.items()}
colors = ['k--', 'g-.', 'b:', 'y-.', 'w']

fig = plt.figure(num=None, figsize=(8, 8))
ax = plt.gca()
for i, (name, r) in enumerate(sorted(R.items())):
    plt.plot(range_points, r, colors[i], label=keyname2legendname[name])
plt.plot(range_points, 0.5*np.array(range_points), 'm', label='Perfect alignment')
plt.legend(prop={'family': 'monospace'})
plt.xlabel('Number of neighbours', fontdict={'family': 'monospace'})
plt.ylabel('Average number of source neighbours', fontdict={'family': 'monospace'})
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')
# plt.show()

if experiment == 'pp':
    fig.savefig(join('/hdd/dropbox/Dropbox/grad/results/retrieval_curves', 'retrieval_pp_batch.pdf'), dpi=1500, format='pdf', transparent=True)
else:
    fig.savefig(join('/hdd/dropbox/Dropbox/grad/results/retrieval_curves', 'retrieval_uada1.pdf'), dpi=1500, format='pdf', transparent=True)
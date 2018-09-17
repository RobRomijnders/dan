from glob import glob
import os
from os.path import join
from itertools import groupby
import numpy as np
import argparse


def main(direc):
    with open(join(direc, 'summary.txt'), 'w') as f_summ:
        results = {}

        # Read the mean IOU's from all the confusion matrices in the direc
        for filename in glob(join(direc, '*.txt')):
            if 'summary.txt' in filename:
                continue
            experiment_eval_name = os.path.basename(filename).replace('.txt', '')
            try:
                experiment_name, eval_data = experiment_eval_name.split('__')
            except ValueError:
                print(experiment_eval_name)
                continue

            experiment_name_split = experiment_name.split('-')
            if experiment_name_split[-1].isdigit():
                if 0 < int(experiment_name_split[-1]) < 20:
                    experiment_name = '-'.join(experiment_name_split[:-1])
            with open(filename, 'r') as f:
                all_lines = f.readlines()
                for line in all_lines:
                    if 'Mean iou' in line:
                        mean_iou = float(line[-7:])
                        results.setdefault(experiment_name, {}).setdefault(eval_data, {}).setdefault('miou', []).append(mean_iou)
                    if 'Global accuracy:' in line:
                        glob_acc = float(line[-7:])
                        results.setdefault(experiment_name, {}).setdefault(eval_data, {}).setdefault('g_acc', []).append(
                            glob_acc)

        for eval_metric in ['miou', 'g_acc']:
            for experiment_name, all_eval_data in sorted(results.items()):
                log_string = f'Exp:{experiment_name:35s} '
                for eval_data, all_metric_results in sorted(all_eval_data.items()):
                    log_string += f'-{eval_data[:3]:4s}'
                    arr = np.array(all_metric_results[eval_metric])
                    log_string += f'{eval_metric} {np.mean(arr):4.1f}+-{np.std(arr):3.1f} ({np.max(arr):4.1f}) '
                # print(log_string)
                f_summ.write(log_string + '\n')
            f_summ.write('\n'*3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize all confusion matrices in directory')
    parser.add_argument('direc', type=str,
                        help='the directory name')
    args = parser.parse_args()
    main(direc=args.direc)

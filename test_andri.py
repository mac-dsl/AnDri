import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from util.util_andri import find_length
from util.util_exp import get_data, get_multi_data, result_f1_acc, save_pickle
from util.TSB_AD.models.andri import AnDri


######## updated 2026-10-08 ##########
parser = argparse.ArgumentParser()
parser.add_argument('-data', choices=['climate', 'traffic', 'NAB', 'Environ', 'CATSv2'], required=True)
parser.add_argument('-nm_len', type=int, default=2)
parser.add_argument('-normalize', default='zero-mean')
parser.add_argument('-k', type=int, default=5)
parser.add_argument('-linkage', default='ward')
########### 2026-10-09 ################
parser.add_argument('-clustering', choices=['adaptive_ahc', 'kshape', 'hc'], default='adaptive_ahc')
parser.add_argument('-max_W', type=int, default=20)
parser.add_argument('-delta_max', type=int, default=10)
parser.add_argument('-rmin', type=float, default=0.005)
parser.add_argument('-step', choices=['True', 'False'], default='True')
parser.add_argument('-rollback', choices=['True', 'False'], default='True')
args = parser.parse_args()


######## updated 2026-10-08 ##########
def score_array(model, length):
    if len(model.scores) == 0:
        return np.zeros(length)
    scores = np.asarray(model.scores_rev, dtype=float)
    scores = np.nan_to_num(scores, nan=np.nanmax(scores))
    scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
    if len(scores) < length:
        scores = np.append(scores, np.full(length - len(scores), scores.mean()))
    return scores[:length]


######## updated 2026-10-08 ##########
def main():
    if args.data == 'CATSv2':
        data_list, label_list, filelist = get_multi_data(args.data)
    else:
        data_list, label_list, filelist = get_data(args.data)
    if not filelist:
        raise FileNotFoundError(f'No CSV files for {args.data}')
    output = Path(__file__).resolve().parent / 'results' / args.data
    output.mkdir(parents=True, exist_ok=True)
    time_all = []
    for data, label, file_name in zip(data_list, label_list, filelist):
        data = np.nan_to_num(data, nan=0).reshape(-1)
        label = label.reshape(-1)
        if args.data in ('climate', 'traffic'):
            window = 24
            train_len = 8760        ## One-year data
        else:
            window = find_length(data)
            while window < 20:
                window *= 2
            if len(data) < 3000 and window > 100:
                window = 60
            elif len(data) >= 3000 and window > 200:
                window = 100
            ######## updated 2026-10-08 ##########
            match = re.search(r'_tr_(\d+)(?:_|$)', file_name)
            if not match:
                raise ValueError(f'No training boundary in {file_name}')
            train_len = int(match.group(1))
        if not 2 < train_len < len(data):
            raise ValueError(f'Invalid training boundary: {train_len}/{len(data)} in {file_name}')

        scores_rev = []
        times = []
        models = []
        ########### 2026-10-09 ################
        for online in ((False, True) if args.clustering == 'adaptive_ahc' else (False,)):
            start = time.time()
            model = AnDri(pattern_length=window, normalize=args.normalize,
                          linkage_method=args.linkage, th_reverse=5, kadj=args.k,
                          nm_len=args.nm_len, overlap=0, max_W=args.max_W,
                          delta_max=args.delta_max, eta=1,
                          clustering=args.clustering)
            model.fit(data, y=label, online=online, training_len=train_len,
                      stepwise=args.step == 'True', min_size=args.rmin,
                      rollback=args.rollback == 'True')
            times.append(time.time() - start)
            scores_rev.append(score_array(model, len(data)))
            models.append(model)
        ######## updated 2026-10-08 ##########
        prefix = f'AnDri_{file_name.replace(".csv__", "__").removesuffix(".csv")}_nm_{args.nm_len}_k_{args.k}'
        ########### 2026-10-09 ################
        if args.clustering != 'adaptive_ahc':
            prefix += f'_{args.clustering}'
        for name, model in zip(('clf_off', 'clf_on'), models):
            save_pickle(output / f'{prefix}_{name}.pickle', model)
        save_pickle(output / f'{prefix}_scores_rev.pickle', scores_rev)
        result_org = result_f1_acc(['AnDri (off)', 'AnDri (on)'][:len(models)], scores_rev, label)
        result_org['file'] = file_name
        result_org.to_csv(output / f'{prefix}_results_org.csv', index=False)
        ########### 2026-10-09 ################
        if args.clustering == 'adaptive_ahc':
            time_all.append({'file': file_name, 'Offline_time': times[0],
                             'Online_time': times[1], 'Offline_flip': models[0].num_flip,
                             'Online_flip': models[1].num_flip})
        else:
            time_all.append({'file': file_name, 'clustering': args.clustering,
                             'Offline_time': times[0], 'Online_time': None,
                             'Offline_flip': models[0].num_flip, 'Online_flip': None})
        time_file = 'time_all.csv' if args.clustering == 'adaptive_ahc' else f'time_all_{args.clustering}.csv'
        pd.DataFrame(time_all).to_csv(output / time_file, index=False)
        print(f'{args.data}/{file_name}: clustering={args.clustering}, train={train_len}, window={window}, off={times[0]}, on={times[1] if len(times) > 1 else None}', flush=True)


if __name__ == '__main__':
    main()

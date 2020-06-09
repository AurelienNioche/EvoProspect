import numpy as np

from parameters.parameters import CONTROL_CONDITIONS, SAME_P, SAME_X
from analysis.stats.stats import iqr


def get_control_data(entries):

    print("Getting the control data...", end=' ', flush=True)

    data = {}
    for cd in CONTROL_CONDITIONS:

        data[cd] = []

        if cd == SAME_P:
            d_cd = entries.filter(is_same_p=True)
        elif cd == SAME_X:
            d_cd = entries.filter(is_same_x=True)
        else:
            raise ValueError(f"Control type not recognized: '{cd}'")

        unq_pairs = np.unique(d_cd.values_list('pair_id'))

        for p_id in unq_pairs:
            d_pair = d_cd.filter(pair_id=p_id)
            n_success = d_pair.filter(choose_best=True).count()
            n = d_pair.count()
            success_rate = n_success/n

            data[cd].append(success_rate)

    print("Done!")

    return data


def get_control_stats(data):

    res = {}
    for cd in CONTROL_CONDITIONS:
        median, _iqr = iqr(data[cd])
        res[cd] = {
            'median': median,
            'iqr': _iqr,
        }

        print(f"Condition '{cd}': n={len(data[cd])}, median={median:.2f}, "
              f"IQR = [{_iqr[0]:.2f}, {_iqr[1]:.2f}], "
              f"IQR comprises values <= 0.5: {_iqr[0] <=0.5}")

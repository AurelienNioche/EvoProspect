import numpy as np

from parameters.parameters import CONTROL_CONDITIONS, SAME_P, SAME_X
from data_interface.models import Data


def _get_chunks(entries, n_chunk=None, n_trials_per_chunk=None,
                randomize=False):

    entries = entries.filter(is_control=True).order_by("id")

    idx = np.array(entries.values_list("id", flat=True))
    n = len(idx)

    if n_chunk is None:
        assert n_trials_per_chunk is not None
        n_chunk = n // n_trials_per_chunk
        remainder = n % n_trials_per_chunk
    else:
        remainder = n % n_chunk

    # Drop the remainder
    if remainder > 0:
        idx = idx[remainder:]

    if randomize:
        np.random.shuffle(idx)

    parts = np.split(idx, n_chunk)

    print(f'Chunk using '
          f'{"chronological" if not randomize else "randomized"} '
          f'order')
    print(f'N trials = {n - remainder}')
    print(f'N parts = {len(parts)} '
          f'(n trials per part = {int((n - remainder) / n_chunk)}, '
          f'remainder = {remainder})')

    return parts


def get_control_history_data(entries, n_chunk=None, n_trials_per_chunk=None,
                             randomize=None):

    print("Getting the control history data...", end=' ', flush=True)

    parts = _get_chunks(entries, n_chunk=n_chunk,
                        n_trials_per_chunk=n_trials_per_chunk,
                        randomize=randomize)

    d_monkey = entries.filter(is_control=True)

    data = {}
    for cd in CONTROL_CONDITIONS:

        data[cd] = []

        if cd == SAME_P:
            d_cd = d_monkey.filter(is_same_p=True)
        elif cd == SAME_X:
            d_cd = d_monkey.filter(is_same_x=True)
        else:
            raise ValueError(f"Control type not recognized: '{cd}'")

        unq_pairs = np.unique(d_cd.values_list('pair_id'))

        for pt in parts:

            data_pt = []

            for p_id in unq_pairs:
                d_pair = d_cd.filter(pair_id=p_id, id__in=pt)
                n_success = d_pair.filter(choose_best=True).count()
                n = d_pair.count()
                success_rate = n_success/n

                data_pt.append(success_rate)
            data[cd].append(data_pt)

    print("Done!")

    return data

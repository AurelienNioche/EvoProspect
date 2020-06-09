import numpy as np

from analysis.sigmoid_fit.sigmoid_fit import sigmoid_fit


def get_freq_risk_data(entries):

    print("Getting the freq risk data...", end=' ', flush=True)

    x = []
    y = []

    d_cd = entries.filter(is_risky=True)
    unq_pairs = np.unique(d_cd.values_list('pair_id'))

    for p_id in unq_pairs:
        d_pair = d_cd.filter(pair_id=p_id)
        n_risky = d_pair.filter(choose_risky=True).count()
        n = d_pair.count()
        rate = n_risky / n

        first_pair = d_pair[0]
        ev_right_minus_ev_left = \
            (first_pair.x1 * first_pair.p1) \
            - (first_pair.x0 * first_pair.p0)

        if first_pair.is_risky_left:
            delta_ev = -ev_right_minus_ev_left
        elif first_pair.is_risky_right:
            delta_ev = ev_right_minus_ev_left
        else:
            raise ValueError("There should be a risky option somewhere!")

        x.append(delta_ev)
        y.append(rate)

    try:
        fit = sigmoid_fit(x=x, y=y)

    except RuntimeError as e:
        print(e)
        fit = None

    data = {'x': x, 'y': y, 'fit': fit}

    print("Done!")

    return data

import numpy as np


def iqr(x):

    n = len(x)

    perc_75, perc_25 = np.percentile(x, [75, 25])

    # print(f'N: {n}')
    #
    # print(f'median: {np.median(x):.02f} '
    #       f'(IQR = {perc_25:.02f} -- {perc_75:.02f})')

    return np.median(x), (perc_25, perc_75)

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

import numpy as np

from data_interface.models import Data

from parameters.parameters import CONTROL_CONDITIONS, \
    BACKUP_FOLDER, GAIN, LOSS

LIMIT_N_TRIAL = 2000
ONE_SIDE_PROP_MAX = 0.80


def get_monkeys():
    selected_monkeys = []

    monkeys = list(np.unique(Data.objects.values_list("monkey")))
    print("All monkeys:", monkeys)

    for m in monkeys:
        keep = True
        entries_m = Data.objects.filter(monkey=m)

        for cond in GAIN, LOSS:

            if cond == GAIN:
                entries = entries_m.filter(is_gain=True)
            elif cond == LOSS:
                entries = entries_m.filter(is_loss=True)
            else:
                raise ValueError

            n_trial = entries.count()
            if n_trial < LIMIT_N_TRIAL:
                print(
                    f"Monkey '{m}' has only {n_trial} trials in condition '{cond}', "
                    f"it will not be included in the analysis")
                keep = False

            n_right = entries.filter(c=1).count()
            prop_right = n_right / n_trial
            if not 1-ONE_SIDE_PROP_MAX <= prop_right <= ONE_SIDE_PROP_MAX:
                print(
                    f"Monkey '{m}' choose the right option {prop_right * 100:.2f}% of the time in condition '{cond}', "
                    f"it will not be included in the analysis")
                keep = False

        if keep:
            selected_monkeys.append(m)

    print("Selected monkeys:", selected_monkeys)
    return selected_monkeys
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

from data_interface.models import Data

import numpy as np
import warnings
import pickle
import collections

from analysis.model.stats import stats_regression_best_values
from analysis.model.model import AgentSideAdditive
# DMSciReports, AgentSoftmax, AgentSide

from parameters.parameters import CONTROL_CONDITIONS, \
    BACKUP_FOLDER, GAIN, LOSS
from analysis.data_preprocessing \
    import get_control_data, get_control_sigmoid_data, \
    get_freq_risk_data, get_info_data, get_control_history_data, \
    get_control_stats

from analysis.model.parameter_estimate import get_parameter_estimate
from analysis.subjects_filtering import get_monkeys


def main():

    monkeys = get_monkeys()
    print()

    for m in monkeys:
        e = Data.objects.filter(monkey=m)
        dates = np.unique(e.values_list("date", flat=True))

        n_trial_day = [e.filter(date=d).count() for d in dates]
        print("m", m)
        print(f"n trial per day mean: {np.mean(n_trial_day):.2f}")
        print(f"n trial per day std: {np.std(n_trial_day):.2f}")
        print("n days", len(dates))
        print("n trial", e.count())
        print()


if __name__ == "__main__":
    main()
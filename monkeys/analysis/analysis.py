import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

from data_interface.models import Data

import numpy as np
import traceback
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

def nested_dict():
    return collections.defaultdict(nested_dict)


class Analysis:

    def __init__(self, class_model, **kwargs):

        self.class_model = class_model

        self.monkeys = None
        self.n_monkey = None

        self.info_data = nested_dict()
        self.control_data = nested_dict()
        self.control_stats = nested_dict()
        self.freq_risk_data = nested_dict()
        self.hist_best_param_data = nested_dict()
        self.hist_control_data = nested_dict()
        self.control_sigmoid_data = nested_dict()

        self.cpt_fit = nested_dict()
        self.risk_sig_fit = nested_dict()
        self.control_sig_fit = nested_dict()

        self._pre_process_data(**kwargs)

    def _pre_process_data(self,
                          skip_exception=True,
                          monkeys=None, **kwargs):

        if monkeys is None:
            monkeys = get_monkeys()

        black_list = []

        for m in monkeys:

            try:
                for cond in (GAIN, LOSS):

                    self._analyse_monkey(m=m, cond=cond, **kwargs)
                    print()

            except Exception as e:
                if skip_exception:
                    track = traceback.format_exc()
                    msg = \
                        f"\nWhile trying to pre-process the data for " \
                        f"monkey '{m}', " \
                        f"I encountered an error. " \
                        "\nHere is the error:\n\n" \
                        f"{track}\n" \
                        f"I will skip the monkey '{m}' " \
                        f"from the rest of the analysis"
                    warnings.warn(msg)
                    black_list.append(m)
                else:
                    raise e

        for m in black_list:
            monkeys.remove(m)
            for cond in GAIN, LOSS:
                self.info_data[cond].pop(m, None)
                self.control_data[cond].pop(m, None)
                self.freq_risk_data[cond].pop(m, None)
                self.hist_best_param_data[cond].pop(m, None)
                self.hist_control_data[cond].pop(m, None)
                self.control_sigmoid_data[cond].pop(m, None)
                self.cpt_fit[cond].pop(m, None)
                self.risk_sig_fit[cond].pop(m, None)
                self.control_sig_fit[cond].pop(m, None)

        self.monkeys = monkeys
        self.n_monkey = len(monkeys)

    def _analyse_monkey(self, m, cond, method,
                        n_trials_per_chunk=None,
                        n_chunk=None,
                        n_trials_per_chunk_control=None,
                        n_chunk_control=None,
                        randomize_chunk_trials=False, force_fit=True,):

        print()
        print("-" * 60 + f" {m} " + "-" * 60 + "\n")

        if cond == GAIN:
            entries = Data.objects.filter(monkey=m, is_gain=True)
        elif cond == LOSS:
            entries = Data.objects.filter(monkey=m, is_loss=True)

        else:
            raise ValueError

        # Sort the data, run fit, etc.
        self.info_data[cond][m] = get_info_data(entries=entries, monkey=m)

        self.control_data[cond][m] = get_control_data(entries)
        self.control_stats[cond][m] = \
            get_control_stats(self.control_data[cond][m])

        self.control_sigmoid_data[cond][m] = \
            get_control_sigmoid_data(entries)
        self.control_sig_fit[cond][m] = \
            {cd: self.control_sigmoid_data[cond][m][cd]['fit']
             for cd in CONTROL_CONDITIONS}

        self.freq_risk_data[cond][m] = get_freq_risk_data(entries)
        self.risk_sig_fit[cond][m] = self.freq_risk_data[cond][m]['fit']

        self.cpt_fit[cond][m] = get_parameter_estimate(
            cond=cond,
            entries=entries,
            force=force_fit,
            n_trials_per_chunk=n_trials_per_chunk,
            n_chunk=n_chunk,
            randomize=randomize_chunk_trials,
            class_model=self.class_model,
            method=method)

        # Stats for comparison of best parameter values
        self.hist_best_param_data[cond][m] = {
            'fit': self.cpt_fit[cond][m],
            'regression':
                stats_regression_best_values(
                    fit=self.cpt_fit[cond][m],
                    class_model=self.class_model)}

        # history of performance for control trials
        self.hist_control_data[cond][m] = \
            get_control_history_data(
                entries=entries,
                n_trials_per_chunk=n_trials_per_chunk_control,
                n_chunk=n_chunk_control)


def run(force_fit=False, use_backup_file=True):
    # for class_model in (AgentSideAdditive, AgentSide,
    #                     AgentSoftmax, DMSciReports):
    class_model = AgentSideAdditive

    print("*" * 150)
    print(f"Using model '{class_model.__name__}'")
    print("*" * 150)
    print()

    bkp_file = os.path.join(BACKUP_FOLDER,
                            f"analysis_{class_model.__name__}")

    if not os.path.exists(bkp_file) or not use_backup_file:
        a = Analysis(
            monkeys=None, # ('Havane', ),""#'Gladys'),
            class_model=class_model,
            n_trials_per_chunk=200,
            n_trials_per_chunk_control=500,
            method='SLSQP',
            force_fit=force_fit,
            skip_exception=False)
        pickle.dump(a, open(bkp_file, "wb"))
    else:
        a = pickle.load(open(bkp_file, "rb"))

    return a

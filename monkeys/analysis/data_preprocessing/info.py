import numpy as np


class Info:

    def __init__(self, entries, monkey):

        self.entries = entries
        self.prop_right = None

        self.text = self._create_text(monkey=monkey)

    def control(self):

        entries = self.entries.filter(is_control=True)
        n_trials = entries.count()
        print("n trials control", n_trials)
        n_success = entries.filter(choose_best=True).count()
        n_pairs = len(np.unique(entries.values_list('pair_id')))
        return n_success, n_trials, n_pairs

    def risk(self):

        entries = self.entries.filter(is_risky=True)
        n_trials = entries.count()
        n_risky = entries.filter(choose_risky=True).count()
        n_pairs = len(np.unique(entries.values_list('pair_id')))
        return n_risky, n_trials, n_pairs

    def choose_right(self):

        entries = self.entries.all()
        n_trials = entries.count()
        n_right = entries.filter(c=1).count()
        n_pairs = len(np.unique(entries.values_list('pair_id')))
        self.prop_right = n_right/n_trials
        return n_right, n_trials, n_pairs

    def _create_text(self, monkey):

        n_success, n_trials_control, n_pairs_control = self.control()
        n_risky, n_trials_risky, n_pairs_risky = self.risk()
        n_right, n_trials, n_pairs = self.choose_right()

        return f"{monkey}\n\n" \
            f"Choose the BEST option when possible = {(n_success/n_trials_control) * 100:.2f}%\n[Ntrials={n_trials_control}, Npairs={n_pairs_control}]\n\n" \
            f"Choose the RISKY option when possible = {(n_risky/n_trials_risky) * 100:.2f}%\n[Ntrials={n_trials_risky}, Npairs={n_pairs_risky}]\n\n" \
            f"Choose RIGHT = {(n_right / n_trials) * 100:.2f}%\n[Ntrials={n_trials}, Npairs={n_pairs}]\n"


def get_info_data(entries, monkey):

    print("Getting the info data...", end=' ', flush=True)
    info = Info(entries=entries, monkey=monkey)
    print("Done!")
    return info

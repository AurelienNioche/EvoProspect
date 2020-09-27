import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import numpy as np
import pandas as pd

from data_interface.models import Data
from analysis.subjects_filtering import get_monkeys
from tqdm import tqdm


def main():

    df = pd.read_csv(os.path.join("table", "table_trials.csv"),
                     index_col=[0])

    print(df)

    monkeys = get_monkeys(verbose=False)

    for m in tqdm(monkeys):

        e = Data.objects.filter(monkey=m)
        dates = np.unique(e.values_list("date", flat=True))

        n_trial_day = [e.filter(date=d).count() for d in dates]

        m_index = m[:3]
        df.loc[m_index, "Mean # trials per day"] = f"{np.mean(n_trial_day):.2f}"
        df.loc[m_index, "STD # trials per day"] = f"{np.std(n_trial_day):.2f}"
        df.loc[m_index, "Min # trials per day"] = str(int(np.min(n_trial_day)))
        df.loc[m_index, "Max # trials per day"] = str(int(np.max(n_trial_day)))
        df.loc[m_index, "Total # trials"] = str(e.count())
        df.loc[m_index, "# days"] = str(len(dates))

    print(df)
    df.to_csv(os.path.join("table", "table_trials_new.csv"))


if __name__ == "__main__":
    main()

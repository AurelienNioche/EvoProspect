import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "MonkeyAnalysis.settings")
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
import numpy as np
import pandas as pd

from data_interface.models import Data
from parameters.parameters import EXPORT_FOLDER, \
    GAIN, LOSS


def export_csv(a):
    data = dict()
    data["monkey"] = [m for m in a.monkeys]
    for cond in GAIN, LOSS:
        for param in "distortion", "risk_aversion", "precision", "side_bias":
            xs = [a.cpt_fit[cond][m][param] for m in a.monkeys]
            x = [np.mean(i) for i in xs]
            data[f"{cond}-{param}"] = x

        if cond == GAIN:
            entries = Data.objects.filter(is_gain=True)
        elif cond == LOSS:
            entries = Data.objects.filter(is_loss=True)
        else:
            raise ValueError
        data[f"{cond}-n_trial"] = [
            entries.filter(monkey=m).count() for m in a.monkeys
        ]

    path_bkp = os.path.join(EXPORT_FOLDER, f"param.csv")
    df = pd.DataFrame(data=data)
    df.to_csv(path_bkp)
    print(f"CSV summary created at '{path_bkp}'")

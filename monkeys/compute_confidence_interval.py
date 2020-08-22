from analysis import analysis
# from analysis.subjects_filtering import get_monkeys
from parameters.parameters import GAIN, LOSS, TABLE_FOLDER

import statsmodels.stats.api as sms

import os
import numpy as np
import pandas as pd


def main():

    a = analysis.run(force_fit=False, use_backup_file=True)
    parameters = a.class_model.param_labels

    neutral = {"distortion": 1, "risk_aversion": 0, "side_bias": 0}

    monkey_list = a.monkeys.copy()
    monkey_list.remove("Havane")
    monkey_list.remove("Gladys")
    monkey_list = ["Havane", "Gladys"] + monkey_list

    for p in parameters:

        row_list = []

        for m in monkey_list:

            if m == "Havane":
                m_name = "Hav"
            elif m == "Gladys":
                m_name = "Gla"
            else:
                m_name = m

            row = {"ID": m_name}

            for cond in GAIN, LOSS:

                x = a.cpt_fit[cond][m][p]
                mean = np.mean(x)
                ic = sms.DescrStatsW(x).tconfint_mean()

                # print(f"{p} {m} {mean:.2f} [{ic[0]:.2f}, {ic[1]:.2f}])
                row[f"{cond.capitalize()} - Mean [CI]"] = f"{mean:.2f} [{ic[0]:.2f}, {ic[1]:.2f}]"

                if p in neutral.keys():
                    could_be_neutral = "Yes" if ic[0] <= neutral[p] <= ic[1] else "No"
                    row[f"{cond.capitalize()} - Neutral"] = could_be_neutral
            row_list.append(row)

        df = pd.DataFrame(row_list)
        df.to_csv(os.path.join(TABLE_FOLDER, f"table_{p}.csv"), index=False)


if __name__ == '__main__':
    main()
from analysis import analysis
# from analysis.subjects_filtering import get_monkeys
from parameters.parameters import GAIN, LOSS, TABLE_FOLDER, CONTROL_CONDITIONS, SIG_MID, SIG_STEEP

import statsmodels.stats.api as sms

import os
import numpy as np
import pandas as pd


def main():

    a = analysis.run(force_fit=False, use_backup_file=True)
    parameters = [SIG_STEEP, SIG_MID]

    neutral = {"sig_steep": 0}

    monkey_list = a.monkeys.copy()
    monkey_list.remove("Havane")
    monkey_list.remove("Gladys")
    monkey_list = ["Havane", "Gladys"] + monkey_list

    for cd in (CONTROL_CONDITIONS + ('risk_sig_fit', )):

        row_list = []

        for m in monkey_list:
            if m == "Havane":
                m_name = "Hav"
            elif m == "Gladys":
                m_name = "Gla"
            else:
                m_name = m

            row = dict()
            row["ID"] = m_name

            for p in parameters:
                row["Parameter"] = p

                for cond in GAIN, LOSS:

                    if cd == "risk_sig_fit":
                        mean = a.risk_sig_fit[cond][m][p]
                        ic = a.risk_sig_fit[cond][m][f'{p}-CI']
                    else:
                        mean = a.control_sigmoid_data[cond][m][cd]['fit'][p]
                        ic = a.control_sigmoid_data[cond][m][cd]['fit'][f'{p}-CI']

                    print("monkey", m, p, "mean", mean, "ic", ic)

                    # print(f"{p} {m} {mean:.2f} [{ic[0]:.2f}, {ic[1]:.2f}])
                    row[f"{cond.capitalize()} - Mean [CI]"] = \
                        f"{mean:.2f} [{ic[0]:.2f}, {ic[1]:.2f}]"

                    if p in neutral.keys():
                        could_be_neutral = "Yes" \
                            if ic[0] <= neutral[p] <= ic[1] else "No"
                        row[f"{cond.capitalize()} - Neutral"] = \
                            could_be_neutral
                    else:
                        row[f"{cond.capitalize()} - Neutral"] = "NA"

                row_list.append(row.copy())

        df = pd.DataFrame(row_list)
        df.to_csv(os.path.join(TABLE_FOLDER, f"table_ic_{cd}.csv"), index=False)


if __name__ == '__main__':
    main()
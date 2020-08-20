from analysis import analysis
# from analysis.subjects_filtering import get_monkeys
from parameters.parameters import GAIN, LOSS

import statsmodels.stats.api as sms

def main():

    a = analysis.run(force_fit=False, use_backup_file=True)
    parameters = a.class_model.param_labels

    neutral = {"distortion": 1, "risk_aversion": 0}

    for cond in GAIN, LOSS:

        for p in parameters:

            for m in a.monkeys:

                x = a.cpt_fit[cond][m][p]
                ic = sms.DescrStatsW(x).tconfint_mean()

                if p in neutral.keys():
                    could_be_neutral = ic[0] <= neutral[p] <= ic[1]
                else:
                    could_be_neutral = ""
                print(p, m, ic, could_be_neutral)


if __name__ == '__main__':
    main()
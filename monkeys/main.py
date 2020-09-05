from analysis import analysis
from analysis.summary import summary
from plot.figure_1 import figure_1
from plot.figure_2 import figure_2
from plot.figure_supplementary import figure_supplementary


def main():

    a = analysis.run(force_fit=False, use_backup_file=True )
    summary.export_csv(a)
    figure_supplementary(a)

    figure_1(a)
    figure_2(a)


if __name__ == '__main__':
    main()

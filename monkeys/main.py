from analysis import analysis
from analysis.summary import summary
from plot.figure_1 import figure_1
from plot.figure_supplementary import figure_supplementary


def main():

    a = analysis.run(force=False)
    summary.export_csv(a)
    figure_1(a)
    figure_supplementary(a)


if __name__ == '__main__':
    main()

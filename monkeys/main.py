from analysis import analysis
from analysis.summary import summary


def main():

    a = analysis.run()
    summary.export_csv(a)



if __name__ == '__main__':
    main()

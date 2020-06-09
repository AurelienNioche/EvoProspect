import numpy as np
import statsmodels.stats
import statsmodels.stats.multitest
from statsmodels import api as sm


def regression(x, y):

    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()   # print(res.summary())
    f, p = res.fvalue, res.f_pvalue
    intercept, beta = res.params
    n = len(y)
    return f, p, n, intercept, beta


def stats_regression_best_values(fit, class_model):

    fs, ps, ns, alphas, betas = [], [], [], [], []

    for param in class_model.param_labels:

        y = fit[param]

        f, p, n, alpha, beta = regression(np.arange(len(y)), y)
        fs.append(f)
        ps.append(p)
        ns.append(n)
        alphas.append(alpha)
        betas.append(beta)

    try:
        valid, p_corr, alpha_c_sidak, alpha_c_bonf = \
            statsmodels.stats.multitest.multipletests(pvals=ps, alpha=0.01,
                                                      method="b")
    except FloatingPointError:
        valid, p_corr, alpha_c_sidak, alpha_c_bonf = None, None, None, None

    rgr_line_param = {}

    for i, param in enumerate(class_model.param_labels):
        p = p_corr[i] < 0.01 if p_corr is not None else None
        rgr_line_param[param] = \
            alphas[i], betas[i], p

    print(f'Linear regression for best-fit parameter value over time: ')

    labels = [f"{param}"
              for param in class_model.param_labels]

    for i in range(len(labels)):
    # for label, f, p, p_c, n, alpha, beta \
    #         in zip(labels, fs, ps, p_corr, ns, alphas, betas):

        str_p_c = f"{p_corr[i]:.3f}" if p_corr is not None else 'None'
        print(f'{labels[i]}: '
              f'F = {fs[i]:.3f}, p = {ps[i]:.3f}, p_c = {str_p_c}, n={ns[i]}'
              f', alpha = {alphas[i]:.2f}, beta = {betas[i]:.2f}')
    print()
    return rgr_line_param

    # if print_latex:
    #     print("[LATEX TABLE CONTENT]")
    #     for monkey, param, fstat, p, p_c, n, alpha, beta in zip(
    #             ["Monkey H", ] * 6 + ["Monkey G", ] * 6,
    #             [i for i in [
    #             r"\omega_G",
    #             r"\omega_L",
    #             r"\alpha_G",
    #             r"\alpha_L",
    #             r"\lambda_G",
    #             r"\lambda_L"]]*2, fs, ps, p_corr, ns, alphas, betas):
    #
    #         p_str = "p<0.001" if p == 0 else f"p={p:.3f}"
    #         p_c_str = "p<0.001" if p_c == 0 else f"p={p_c:.3f}"
    #
    #         if p_c < 0.01:
    #             p_c_str += '^*'
    #         print(f"{monkey} & ${param}$ &" + r'$2\times'
    #               + f"{n}$ & ${alpha:.2f}$ & ${beta:.2f}$ & "
    #               + f"{fstat:.2f} & ${p_str}$ & ${p_c_str}$" + r"\\")
    #     print("[LATEX TABLE CONTENT]\n")


# def display_table_content(fit):
#
#     print("[LATEX TABLE CONTENT]")
#     monkeys = sorted(fit.keys())
#     for monkey in monkeys:
#
#         # To display
#         dsp = ""
#
#         for param_pos, param_neg in (
#             ('pos_risk_aversion', 'neg_risk_aversion'),
#             ('pos_distortion', 'neg_distortion'),
#             ('pos_precision', 'neg_precision')
#
#         ):
#             mean_pos = np.mean(fit[monkey][param_pos])
#             std_pos = np.std(fit[monkey][param_pos])
#
#             mean_neg = np.mean(fit[monkey][param_neg])
#             std_neg = np.std(fit[monkey][param_neg])
#
#             dsp += f"${mean_pos:.2f}$ " + r"($\pm " + f"{std_pos:.2f}$); " \
#                 f"${mean_neg:.2f}$ " + r"($\pm " + f"{std_neg:.2f}$) &"
#
#         dsp = dsp[:-1] + "\\"
#         log(f"Line for {monkey} for table displaying best parameter values",
#             name=NAME)
#         print(dsp)
#     print("[LATEX TABLE CONTENT]\n")

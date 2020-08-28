import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import t


from parameters.parameters import SIG_MID, SIG_STEEP


PARAM_LABELS = 'x0', 'k'


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def sigmoid_fit(x, y, n_points=100, max_eval=10000):

    p_opt, p_cov = curve_fit(f=sigmoid, xdata=x, ydata=y,
                             maxfev=max_eval)

    x_fit = np.linspace(min(x), max(x), n_points)
    y_fit = sigmoid(x_fit, *p_opt)

    fit_stats = stats(
        n=len(y),
        p_opt=p_opt,
        p_cov=p_cov,
        p_labels=(SIG_MID, SIG_STEEP)
    )

    return {
        'x': x_fit,
        'y': y_fit,
        SIG_MID: p_opt[0],
        SIG_STEEP: p_opt[1],
        **fit_stats
    }


def stats(n, p_opt, p_cov, p_labels, alpha=0.05):

    """
    For explanations on this, see:
    https://www.mathworks.com/help/curvefit/confidence-and-prediction-bounds.html
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
    """

    # Compute p err -------------------------
    try:
        p_err = np.sqrt(np.diag(p_cov))
    except FloatingPointError:
        d = np.diag(p_cov)
        d.setflags(write=True)
        d[d < 0] = 0
        p_err = np.sqrt(d)

    # Compute t -----------------------------
    p = len(p_opt)  # number of parameters

    dof = max(0, n - p)  # number of degrees of freedom

    #  Inverse of Student's t cumulative distribution function
    tval = t.ppf(1.0 - alpha / 2., dof)

    # Compute ci ---------------------------
    r = {}
    for i, (pr, std) in enumerate(zip(p_opt, p_err)):

        ci = std * tval
        print(f'{PARAM_LABELS[i]}: {pr:.2f} [{pr - ci:.2f}  {pr + ci:.2f}]')
        r[f"{p_labels[i]}-CI"] = (pr - ci, pr + ci)

    return r

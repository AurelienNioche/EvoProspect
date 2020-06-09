import numpy as np
from scipy.special import expit
from abc import abstractmethod

EPS = np.finfo(float).eps


class DecisionMakingModel:

    param_labels = None
    bounds = None
    init_guess = None

    def __init__(self, param):
        self.param = param

    @abstractmethod
    def p_choice(self, p0, x0, p1, x1, c):
        pass

    @classmethod
    def objective(cls, param, *args):

        data = args[0]

        model = cls(param=param)

        n = len(data)
        ll = np.zeros(n)

        for i in range(n):
            pi = model.p_choice(**data[i])
            ll[i] = np.log(pi + EPS)

        return -ll.sum()


class DMSciReports(DecisionMakingModel):

    param_labels = ['distortion', 'precision', 'risk_aversion']
    bounds = [(0.2, 1.8), (0.0, 10.0), (-0.99, 0.99)]
    init_guess = (1.00, 0.00, 0.00)

    def __init__(self, param):

        self.distortion, self.precision, self.risk_aversion = param
        super().__init__(param)

    @classmethod
    def pi(cls, p, alpha):
        if isinstance(p, np.ndarray):
            to_return = np.zeros(p.shape)
            unq_zero = p != 0
            to_return[unq_zero] = np.exp(-(-np.log(p)) ** alpha)
            return to_return
        else:
            if p == 0:
                return 0
            else:
                return np.exp(-(-np.log(p)) ** alpha)

    @staticmethod
    def omega(p0, x0, p1, x1):

        """
        Compute the value of the risk aversion parameter for which U(L0)=U(L1)
        U(L0) = U(L1)
        p0*x0**(1-w) = p1*x1**(1-w)
        p0*x0**(1-w) / (p1*x1**(1-w)) = 1
        log(p0*x0**(1-w) / (p1*x1**(1-w))) = log(1)
        log(p0/p1) + log(m0**(1-w)/log(m1**(1-w)) = 0
        log(p0/p1) + (1-w)*log(m0/m1) = 0
        log(p0/p1) + log(m0/m1) - w*log(m0/m1) = 0
        log(p0/p1) + log(m0/m1) - w*log(m0/m1) = 0
        w = (log(p0/p1) + log(m0/m1)) / log(m0/m1)
        w = log((p0*m0)/(p1*m1)) / log(m0/m1)
        w = (log(p0 * x0) - log(p1 * x1)) / (log(x0) - log(x1))
        """

        return (np.log(p0 * x0) - np.log(p1 * x1)) / (np.log(x0) - np.log(x1))

    def rho(self, p0, x0, p1, x1):

        """
        from http://www.econ.upf.edu/~apesteguia/Monotone_Stochastic_Choice_Models.pdf
        (formula end of page 77)
        """

        omega = self.omega(p0=p0, p1=p1, x0=x0, x1=x1)

        a = np.exp(self.precision*omega)
        b = np.exp(self.precision*self.risk_aversion)
        return a / (a+b)

    def p(self, p0, x0, p1, x1):

        assert x0 > 0 and x1 > 0
        lo_0_riskiest = p0 < p1 and x0 > x1
        lo_1_riskiest = p0 > p1 and x0 < x1

        dist_p0 = self.pi(p0, self.distortion)
        dist_p1 = self.pi(p1, self.distortion)

        p_choose_risk = self.rho(p0=dist_p0, x0=x0,
                                 p1=dist_p1, x1=x1)

        p = np.zeros(2)
        if lo_0_riskiest:
            p[:] = p_choose_risk, 1 - p_choose_risk
        elif lo_1_riskiest:
            p[:] = 1 - p_choose_risk, p_choose_risk
        else:
            raise ValueError("One lottery should be riskier than the other")

        return p

    def p_choice(self, p0, x0, p1, x1, c):

        p = self.p(p0=p0, x0=x0, p1=p1, x1=x1)
        return p[c]

    @staticmethod
    def u(x, risk_aversion):
        return x ** (1-risk_aversion)


class AgentSoftmax(DMSciReports):

    param_labels = ['distortion', 'precision', 'risk_aversion']
    bounds = [(0.2, 1.8), (0.1, 10.0), (-0.99, 0.99)]
    init_guess = (1.00, 1.00, 0.00)

    @classmethod
    def softmax(cls, v, precision):
        return expit(v/precision)

    @staticmethod
    def u(x, risk_aversion):
        if isinstance(x, np.ndarray):
            raise Exception
        else:
            if x >= 0:
                return x ** (1-risk_aversion)
            else:
                return - np.abs(x) ** (1 + risk_aversion)

    def p(self, p0, x0, p1, x1):

        v0 = self.pi(p0, self.distortion) * self.u(x0, self.risk_aversion)
        v1 = self.pi(p1, self.distortion) * self.u(x1, self.risk_aversion)
        p = np.zeros(2)
        p[0] = self.softmax(v0-v1, self.precision)
        p[1] = 1 - p[0]
        return p


class AgentSide(AgentSoftmax):

    param_labels = ['distortion', 'precision', 'risk_aversion', 'side_bias']
    bounds = [(0.2, 1.8), (0.1, 10.0), (-0.99, 0.99), (-1.0, 1.0)]
    init_guess = (1.00, 1.00, 0.00, 0.50)

    def __init__(self, param):

        self.distortion, self.precision, self.risk_aversion, \
            self.side_bias = param

    def p(self, p0, x0, p1, x1):
        np.seterr(all="raise")

        v0 = self.pi(p0, self.distortion) * self.u(x0, self.risk_aversion) \
            * (1+max(0, -self.side_bias))
        v1 = self.pi(p1, self.distortion) * self.u(x1, self.risk_aversion) \
            * (1+max(0, +self.side_bias))
        p = np.zeros(2)
        p[0] = self.softmax(v0-v1, self.precision)
        p[1] = 1 - p[0]
        return p


class AgentSideAdditive(AgentSoftmax):

    param_labels = ['distortion', 'precision', 'risk_aversion', 'side_bias']
    bounds = [(0.2, 1.8), (0.1, 10.0), (-0.99, 0.99), (-10.0, 10.0)]
    init_guess = (1.00, 1.00, 0.00, 0.50)

    def __init__(self, param):

        self.distortion, self.precision, self.risk_aversion, \
            self.side_bias = param

    def p(self, p0, x0, p1, x1):
        np.seterr(all="raise")

        v0 = self.pi(p0, self.distortion) * self.u(x0, self.risk_aversion) \
            + max(0, -self.side_bias)
        v1 = self.pi(p1, self.distortion) * self.u(x1, self.risk_aversion) \
            + max(0, +self.side_bias)
        p = np.zeros(2)
        p[0] = self.softmax(v0-v1, self.precision)
        p[1] = 1 - p[0]
        return p

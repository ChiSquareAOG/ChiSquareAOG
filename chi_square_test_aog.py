import numpy as np
import scipy.stats as stats
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sympy import *
from scipy.stats import chi2
import statsmodels.api as sm
import aog_tables
import math


class ScaleParametricDistribution:

    def __init__(self, name, scale, count, rvs):
        self.name = name
        self.param=[]
        self.param.append(0)
        self.param.append(scale)
        self.count = count
        self.rvs = rvs


    def mle(self):
        if self.param[1] is None:
            mle = self.name.fit(self.rvs)
            self.scale_estimation = true
            self.param[1] = mle[1]
        else:
            self.param[1] = scale
            self.scale_estimation = false

    def estimate_parameters(self, points, ks):
        if self.scale_estimation:
            res = minimize(self.lnl_scale, x0=[self.param[1]], method='Nelder-Mead',
                           args=(points, ks),
                           options={'xtol': 1e-6})
            self.param[1] = float(res.x)

    def lnl_scale(self, x, points, ks):
        # log-likelihood function (scale parameter)
        s = 0
        for i in range(0, self.count - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1], scale=x[0], loc=0) - self.name.cdf(points[i], scale=x[0],loc=0))
        s -= ks[count - 1] * log(1 - self.name.cdf(points[count - 1], scale=x[0], loc=0))
        return s

    def get_freedom_value(self):
        freedom_value=self.count-1
        if self.scale_estimation:
            freedom_value = freedom_value-1
        return freedom_value

    def calculate_probabilities(self, points):
        probs = np.zeros(count)
        for i in range(1, count):
            probs[i - 1] = self.name.cdf(points[i], scale=self.param[1]) - self.name.cdf(
                points[i - 1], scale=self.param[1])
        probs[count - 1] = 1 - self.name.cdf(points[count - 1], scale=self.param[1])

        return probs


class LocScaleParametricDistribution:
    def __init__(self, name, loc, scale, count, rvs):
        self.name = name
        self.param=[]
        self.param.append(loc)
        self.param.append(scale)
        self.count = count
        self.rvs = rvs


    def mle(self):
        self.scale_estimation = false
        self.loc_estimation = false
        mle = self.name.fit(self.rvs)
        if self.param[0] is None:
            self.param[0]=mle[0]
            self.loc_estimation = true
        if self.param[1] is None:
            self.param[1]=mle[1]
            self.scale_estimation=true

    def estimate_parameters(self, points, ks):
        if self.loc_estimation:
            if self.scale_estimation:
                res = minimize(self.lnl_loc_scale, x0=[self.param[0], self.param[1]], method='Nelder-Mead',
                               args=(points, ks),
                               options={'xtol': 1e-6})
                self.param[0], self.param[1] = res.x

            else:
                res = minimize(self.lnl_loc, self.param[0], method='Nelder-Mead', args=(points, ks),
                               options={'xtol': 1e-6})
                self.param[0] = float(res.x)
        else:
            if self.scale_estimation:
                res = minimize(self.lnl_scale, self.param[1], method='Nelder-Mead', args=(points, ks),
                               options={'xtol': 1e-6})
                self.param[1] = float(res.x)

    def lnl_scale(self, x, points, ks):
        # log-likelihood function (scale parameter)
        s = 0
        for i in range(0, self.k - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1], scale=x[0], loc=self.param[0]) - self.name.cdf(points[i], scale=x[0],loc=self.param[0]))
        s -= ks[k - 1] * log(1 - self.name.cdf(points[k - 1], scale=x[0],loc=self.param[0]))
        return s
    def lnl_loc (self,x,points,ks):
        s = 0
        for i in range(0, self.k - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1], scale=self.param[1], loc=x[0]) - self.name.cdf(points[i], scale=self.param[1],
                                                                                            loc=x[0]))
        s -= ks[k - 1] * log(1 - self.name.cdf(points[k - 1], scale=self.param[1], loc=x[0]))
        return s
    def lnl_loc_scale (self,x,points,ks):
        s = 0
        for i in range(0, self.k - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1], scale=x[1], loc=x[0]) - self.name.cdf(points[i], scale=x[1],
                                                                                            loc=x[0]))
        s -= ks[k - 1] * log(1 - self.name.cdf(points[k - 1], scale=x[1], loc=x[0]))
        return s
    def get_freedom_value(self):
        freedom_value=self.k-1
        if self.scale_estimation:
            freedom_value=freedom_value-1
        if self.loc_estimation:
            freedom_value = freedom_value - 1
        return freedom_value

    def calculate_probabilities(self, points):
        probs = np.zeros(k)
        for i in range(1, k):
            probs[i - 1] = self.name.cdf(points[i], loc=self.param[0], scale=self.param[1]) - self.name.cdf(
                points[i - 1], loc=self.param[0], scale=self.param[1])
        probs[k - 1] = 1 - self.name.cdf(points[k - 1], loc=self.param[0], scale=self.param[1])

        return probs



class ShapeScaleParametricDistribution:
    def __init__(self, name, scale, shape, k, rvs):
        self.name = name
        self.param=[]
        self.param.append(0)
        self.param.append(scale)
        self.param.append(shape)
        self.k = k
        self.rvs = rvs


    def mle(self):
        self.scale_estimation = false
        self.loc_estimation = false
        self.shape_estimation = false
        mle = self.name.fit(data= self.rvs, loc=0)
        if self.param[1] is None:
            self.param[1]=mle[2]
            self.scale_estimation=true
        if self.param[2] is None:
            self.param[2]=mle[0]
            self.shape_estimation=true

    def estimate_parameters(self, points, ks):
        if self.shape_estimation:
            if self.scale_estimation:
                res = minimize(self.lnl_shape_scale, x0=[self.param[1], self.param[2]], method='Nelder-Mead',
                               args=(points, ks),
                               options={'xtol': 1e-6})
                self.param[1], self.param[2] = res.x

            else:
                res = minimize(self.lnl_shape, self.param[2], method='Nelder-Mead', args=(points, ks),
                               options={'xtol': 1e-6})
                self.param[2] = float(res.x)
        else:
            if self.scale_estimation:
                res = minimize(self.lnl_scale, self.param[1], method='Nelder-Mead', args=(points, ks),
                               options={'xtol': 1e-6})
                self.param[1] = float(res.x)

    def lnl_scale(self, x, points, ks):
        # log-likelihood function (scale parameter)
        s = 0
        for i in range(0, self.k - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1], self.param[2], loc=self.param[0], scale=x[0] ) - self.name.cdf(points[i], self.param[2], loc=self.param[0], scale=x[0],))
        s -= ks[k - 1] * log(1 - self.name.cdf(points[k - 1], self.param[2],loc=self.param[0], scale=x[0]))
        return s
    def lnl_shape (self,x,points,ks):
        s = 0
        for i in range(0, self.k - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1],x[0], scale=self.param[1], loc=self.param[0]) - self.name.cdf(points[i],
                                                                                            x[0],scale=self.param[1], loc=self.param[0]))
        s -= ks[k - 1] * log(1 - self.name.cdf(points[k - 1], x[0],scale=self.param[1], loc=self.param[0]))
        return s
    def lnl_shape_scale (self,x,points,ks):
        s = 0
        for i in range(0, self.k - 1):
            s -= ks[i] * log(
                self.name.cdf(points[i + 1],x[1],loc=self.param[0], scale=x[0]) - self.name.cdf(points[i],x[1], loc=self.param[0],scale=x[0]))
        s -= ks[k - 1] * log(1 - self.name.cdf(points[k - 1], x[1],loc=self.param[0],scale=x[0]))
        return s
    def get_freedom_value(self):
        freedom_value=self.k-1
        if self.scale_estimation:
            freedom_value=freedom_value-1
        if self.shape_estimation:
            freedom_value = freedom_value - 1
        return freedom_value

    def calculate_probabilities(self, points):
        probs = np.zeros(k)
        for i in range(1, k):
            probs[i - 1] = self.name.cdf(points[i], self.param[2], loc=self.param[0], scale=self.param[1]) - self.name.cdf(
                points[i - 1], self.param[2],loc=self.param[0], scale=self.param[1])
        probs[k - 1] = 1 - self.name.cdf(points[k - 1], self.param[2], loc=self.param[0],scale=self.param[1])
        return probs


def chi_square_aog(distname, rvs, loc=None, scale=None, shape=None, alpha=0.05, k=5):
    '''
        perform chisquare test for random continuous distribution using asymptotically optimal grouping
        Parameters:
        distname : string, name of distribution function
        rvs : sequence, random variates
        loc : location parameter of distribution
        scale : scale parameter of distribution
        alpha : float, significance level, threshold for p-value
        k :
    '''
    # define parameters for test
    rvs = sorted(rvs)
    n = len(rvs)
    ks = np.zeros(k)
    points = np.zeros(k + 1)
    probabilities = np.zeros(k)
    k = np.zeros(k)

    # exponential distribution  (scale)
    if distname == stats.expon:
        fun=ScaleParametricDistribution(distname, scale, k, rvs)
        fun.mle()

        try:
            index = aog_tables.exp_k.index(k)
            probabilities = aog_tables.exp_prob[index]
        except ValueError:
            for i in range(0, k):
                probabilities[i] = 1 / k

    if distname == stats.norm:
        fun = LocScaleParametricDistribution(distname, loc, scale, k, rvs)
        fun.mle()

        if fun.loc_estimation:
            if fun.scale_estimation:
                try:
                    index = aog_tables.norm_loc_scale_k.index(k)
                    probabilities = aog_tables.norm_loc_scale_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
            else:
                try:
                    index = aog_tables.norm_loc_k.index(k)
                    probabilities = aog_tables.norm_loc_prob[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
        else:
            if fun.scale_estimation:
                try:
                    index = aog_tables.norm_scale_k.index(k)
                    probabilities = aog_tables.norm_scale_prob[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
            else:
                try:
                    index = aog_tables.norm_loc_scale_k.index(k)
                    probabilities = aog_tables.norm_loc_scale_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k


                        # logistic distribution

    if distname == stats.logistic:
        fun=LocScaleParametricDistribution(distname, loc, scale, k, rvs)
        fun.mle()
        if fun.loc_estimation:
            if fun.scale_estimation:
                try:
                    index = aog_tables.logistic_loc_scale_k.index(k)
                    probabilities = aog_tables.logistic_loc_scale_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
            else:
                for i in range(0, k):
                    probabilities[i] = 1 / k
        else:
            if fun.scale_estimation:
                try:
                    index = aog_tables.logistic_scale_k.index(k)
                    probabilities = aog_tables.logistic_scale_prob[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
            else:
                try:
                    index = aog_tables.logistic_loc_scale_k.index(k)
                    probabilities = aog_tables.logistic_loc_scale_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k

    if distname == stats.weibull_min:
        fun = ShapeScaleParametricDistribution(distname, shape=shape, scale=scale, k=k, rvs=rvs)
        fun.mle()

        if fun.shape_estimation:
            if fun.scale_estimation:
                try:
                    index = aog_tables.weibull_scale_shape_k.index(k)
                    probabilities = aog_tables.weibull_scale_shape_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
            else:
                try:
                    index=aog_tables.weibull_shape_k.index(k)
                    probabilities=aog_tables.weibull_shape_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
        else:
            if fun.scale_estimation:
                try:
                    index = aog_tables.weibull_scale_k.index(k)
                    probabilities = aog_tables.weibull_scale_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k
            else:
                try:
                    index = aog_tables.weibull_scale_shape_k.index(k)
                    probabilities = aog_tables.weibull_scale_shape_probs[index]
                except ValueError:
                    for i in range(0, k):
                        probabilities[i] = 1 / k

    # if distname == stats.gamma:
    #     fun = ShapeScaleParametricDistribution(distname, shape=shape, scale=scale, k=k, rvs=rvs)
    #     fun.mle()
    #
    #     if fun.shape_estimation:
    #         if fun.scale_estimation:
    #             try:
    #                 index = aog_tables.gamma_scale_shape_list.index(k)
    #
    #                 from scipy import interpolate
    #
    #                 for i in range(0, k):
    #                     prob = interpolate.interp1d(aog_tables.gamma_scale_shape_values,
    #                                                 aog_tables.gamma_scale_shape_probs[index][i], kind='cubic')
    #                     probabilities[i] = (prob(fun.param[2]))
    #
    #
    #             except ValueError:
    #                 for i in range(0, k):
    #                     probabilities[i] = 1 / k
    #         else:
    #             try:
    #                 index = aog_tables.gamma_shape_list.index(k)
    #
    #                 from scipy import interpolate
    #
    #                 for i in range (0, k):
    #                     prob = interpolate.interp1d(aog_tables.gamma_shape_values, aog_tables.gamma_shape_probs[index][i], kind='linear')
    #                     probabilities[i]=(prob(fun.param[2]))
    #
    #
    #
    #             except ValueError:
    #                 for i in range(0, k):
    #                     probabilities[i] = 1 / k
    #     else:
    #         if fun.scale_estimation:
    #             try:
    #                 index = aog_tables.gamma_scale_list.index(k)
    #
    #                 from scipy import interpolate
    #
    #                 for i in range(0, k):
    #                     prob = interpolate.interp1d(aog_tables.gamma_scale_values,
    #                                                 aog_tables.gamma_scale_probs[index][i], kind='linear')
    #                     probabilities[i] = (prob(fun.param[2]))
    #
    #             except ValueError:
    #                 for i in range(0, k):
    #                     probabilities[i] = 1 / k
    #         else:
    #             try:
    #                 index = aog_tables.gamma_scale_shape_list.index(k)
    #                 probabilities = aog_tables.gamma_scale_shape_probs[index]
    #             except ValueError:
    #                 for i in range(0, k):
    #                     probabilities[i] = 1 / k
    sum = 0
    for i in range(0, k - 1):
        tmp[i] = round(n * probabilities[i])
        sum += tmp[i]

        ks[i] = int(tmp[i])
    tmp[k - 1] = n - sum
    ks[k - 1] = int(tmp[k - 1])


    sum = 0
    points[0] = stats.norm.a
    for i in range(0, k - 1):
        sum += ks[i]
        points[i + 1] = rvs[int(sum) - 1]
    points[k] = distname.b
    fun.estimate_parameters(points, ks)
    distsupp = []
    for i in range(0, k + 1):
        distsupp.append(points[i])
    probs = fun.calculate_probabilities(points)
    distmass = []
    for i in range(0, k):
        distmass.append(probs[i])
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)
    freq, hsupp = np.histogram(rvs, distsupp)
    chi = 0
    for i in range(0, k):
        chi = chi + (freq[i] / n - distmass[i]) * (freq[i] / n - distmass[i]) / distmass[i]
    chi = chi * n
    freedom_val=fun.get_freedom_value()

    pval = stats.distributions.chi2.sf(chi, freedom_val)
    #return chi
    return chi, pval, (pval > alpha), 'chisquare - test for %s at scale = %s with pval = %s' % (str(distname.name), str(fun.param), str(pval))

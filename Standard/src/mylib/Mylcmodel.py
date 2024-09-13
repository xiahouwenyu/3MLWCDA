import astropy.units as astropy_units
import astromodels.functions.numba_functions as nb_func
from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation
import astropy.units as u
import numpy as np
from Mylightcurve import *

def powerlawlc(t, A, index, t0):
    result = A*(t-t0)**(index)
    result[t<=237]=0
    return result

import numpy as np

def fSBPL(xx, par):
    xx = np.array(xx)
    result = par[0] * pow(pow((xx - par[5]) / par[1], -par[2] * par[3]) + pow((xx - par[5]) / par[1], -par[2] * par[4]), -1 / par[2])
    result = np.array(result)
    cut = xx < par[5]
    result[cut]=0
    return result

def FRP(xx, par):
    xx = np.array(xx)
    result = fSBPL(par[9], par) * ((xx / par[9]) ** par[10])
    result=np.array(result)
    cut1 = (xx < par[5])
    result[cut1]=0
    cut2 = xx >= par[9]
    result[cut2]=fSBPL(xx[cut2], par)
    result = np.array(result)
    return result

def fSSQPL(xx, par):
    xx = np.array(xx)
    part1 = FRP(xx, par) ** (-par[6])
    part2 = (FRP(par[7], par) * ((xx / par[7]) ** par[8])) ** (-par[6])
    return (part1 + part2) ** (-(1 / par[6]))


class SBPL(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  SBPL
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            F0 :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                transformation : log10
                min : 1e-5
                max : 1e20
                delta : 0.1

            tb :
                desc : break time
                initial value : 10
            omega :
                desc : break smooth
                min : 0
                initial value : 1
            alpha1 :
                desc : index 1
                initial value : 1
            alpha2 :
                desc : index 2
                initial value : -2
            t0 :
                desc : start time
                initial value : 230
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.alpha1.unit = u.dimensionless_unscaled
            self.omega.unit =  u.dimensionless_unscaled
            self.alpha2.unit = u.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            self.F0.unit = u.dimensionless_unscaled

            # The normalization has the same units as the y

            self.tb.unit = u.s
            self.t0.unit = u.s


        def evaluate(self, x, F0, tb, omega, alpha1, alpha2, t0):

            result = F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega)
            result[x<t0]=0
            return result

class STPL(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  STPL
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            F0 :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                transformation : log10
                min : 1e-5
                max : 1e20
                delta : 0.1

            tb :
                desc : break time
                initial value : 10
            omega :
                desc : break smooth
                min : 0
                initial value : 1
            alpha1 :
                desc : index 1
                initial value : 1
            alpha2 :
                desc : index 2
                initial value : -2
            t0 :
                desc : start time
                initial value : 230

            omega2 :
                desc : break smooth
                min : 0
                initial value : 1

            tb2 :
                desc : break time
                initial value : 10

            alpha3 :
                desc : index 2
                initial value : -2
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.alpha1.unit = u.dimensionless_unscaled
            self.omega.unit =  u.dimensionless_unscaled
            self.alpha2.unit = u.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            self.F0.unit = u.dimensionless_unscaled

            # The normalization has the same units as the y

            self.tb.unit = u.s
            self.t0.unit = u.s

            self.omega2.unit =  u.dimensionless_unscaled
            self.alpha3.unit = u.dimensionless_unscaled
            self.tb2.unit = u.s


        def evaluate(self, x, F0, tb, omega, alpha1, alpha2, t0, omega2, tb2, alpha3):

            result = F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega)
            result2 = F0 * pow(pow((tb2 - t0) / tb, -omega * alpha1) + pow((tb2 - t0) / tb, -omega * alpha2), -1 / omega)
            result = (result**(-omega2)+(result2*(x/tb2)**alpha3)**(-omega2))**(-1/omega2)
            result[x<t0]=0
            return result

class SSQPL(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  SSQPL
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            F0 :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                transformation : log10
                min : 1e-5
                max : 1e20
                delta : 0.1

            tb :
                desc : break time
                initial value : 10
            omega :
                desc : break smooth
                min : 0
                initial value : 1
            alpha1 :
                desc : index 1
                initial value : 1
            alpha2 :
                desc : index 2
                initial value : -2
            t0 :
                desc : start time
                initial value : 230

            omega2 :
                desc : break smooth
                min : 0
                initial value : 1

            tb2 :
                desc : break time
                initial value : 10

            alpha3 :
                desc : index 2
                initial value : -2

            tb0 :
                desc : break time
                initial value : 10

            alpha0 :    
                desc : index 0
                initial value : 10
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.alpha1.unit = u.dimensionless_unscaled
            self.omega.unit =  u.dimensionless_unscaled
            self.alpha2.unit = u.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            self.F0.unit = u.dimensionless_unscaled

            # The normalization has the same units as the y

            self.tb.unit = u.s
            self.t0.unit = u.s

            self.omega2.unit =  u.dimensionless_unscaled
            self.alpha3.unit = u.dimensionless_unscaled
            self.tb2.unit = u.s

            self.tb0.unit = u.s
            self.alpha0.unit = u.dimensionless_unscaled


        def evaluate(self, x, F0, tb, omega, alpha1, alpha2, t0, omega2, tb2, alpha3, tb0, alpha0):

            result = F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega)
            result2 = F0 * pow(pow((tb0 - t0) / tb, -omega * alpha1) + pow((tb0 - t0) / tb, -omega * alpha2), -1 / omega)
            result3 = F0 * pow(pow((tb2 - t0) / tb, -omega * alpha1) + pow((tb2 - t0) / tb, -omega * alpha2), -1 / omega)

            fFRP = result2 * ((x / tb0) ** alpha0)
            cut1 = (x < t0)
            fFRP[cut1]=0
            cut2 = x >= tb0
            fFRP[cut2]=result[cut2]

            fFRP = result2 * ((x / tb0) ** alpha0)
            cut1 = (x < t0)
            fFRP[cut1]=0
            cut2 = x >= tb0
            fFRP[cut2]=result[cut2]

            part1 = fFRP ** (-omega2)
            part2 = (result3 * ((x / tb2) ** alpha3)) ** (-omega2)
            fresult = (part1 + part2) ** (-(1 / omega2))
            fresult[x < t0]=0
            return fresult
        
class TPLM(Function1D, metaclass=FunctionMeta): ##An alternative way to describe the GRB X-ray afterglow is a two-component phenomenological model proposed by O’Brien et al. (2006)andWillingale et al. (2007).
        r"""
        description :
            A  TPLM
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            Fc :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                min : 1e-5
                max : 1e20
                delta : 0.1

            alpha_c :
                desc : index 1
                initial value : 2

            Tc :
                desc : break time
                initial value : 100

            tc :
                desc : start
                initial value : 20
            
            t0 :
                desc : start time
                initial value : 226
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.Fc.unit = u.dimensionless_unscaled
            self.alpha_c.unit =  u.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            # The normalization has the same units as the y

            self.Tc.unit = u.s
            self.tc.unit = u.s
            self.t0.unit = u.s

        
        def evaluate(self, x, Fc, alpha_c, Tc, tc, t0):
            results = np.zeros(len(x))
            x=x-t0
            # Tc=Tc+t0
            results[x < Tc] = Fc * np.exp(alpha_c - (x[x < Tc] * alpha_c / Tc)) * np.exp(-tc / x[x < Tc])
            results[x >= Tc] = Fc * (x[x >= Tc] / Tc) ** -alpha_c * np.exp(-tc / x[x >= Tc])
            results[x<=0]=0
            return results
        
class TPLM1(Function1D, metaclass=FunctionMeta): ##An alternative way to describe the GRB X-ray afterglow is a two-component phenomenological model proposed by O’Brien et al. (2006)andWillingale et al. (2007).
        r"""
        description :
            A  TPLM
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            Fc :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                min : 1e-5
                max : 1e20
                delta : 0.1

            alpha_c :
                desc : index 1
                initial value : 2

            Tc :
                desc : break time
                initial value : 100

            tc :
                desc : start
                initial value : 20
            
            t0 :
                desc : start time
                initial value : 226
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.Fc.unit = u.dimensionless_unscaled
            self.alpha_c.unit =  u.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            # The normalization has the same units as the y

            self.Tc.unit = u.s
            self.tc.unit = u.s
            self.t0.unit = u.s

        
        def evaluate(self, x, Fc, alpha_c, Tc, tc, t0):
            results = np.zeros(len(x))
            x=x-t0
            # Tc=Tc+t0
            results = Fc * np.exp(alpha_c - (x * alpha_c / Tc)) * np.exp(-tc / x)
            # results[x >= Tc] = Fc * (x[x >= Tc] / Tc) ** -alpha_c * np.exp(-tc / x[x >= Tc])
            results[x<=0]=0
            return results

class TPLM2(Function1D, metaclass=FunctionMeta): ##An alternative way to describe the GRB X-ray afterglow is a two-component phenomenological model proposed by O’Brien et al. (2006)andWillingale et al. (2007).
        r"""
        description :
            A  TPLM
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            Fc :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                min : 1e-5
                max : 1e20
                delta : 0.1

            alpha_c :
                desc : index 1
                initial value : 2

            Tc :
                desc : break time
                initial value : 100

            tc :
                desc : start
                initial value : 20
            
            t0 :
                desc : start time
                initial value : 226
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.Fc.unit = u.dimensionless_unscaled
            self.alpha_c.unit =  u.dimensionless_unscaled

            # The pivot energy has always the same dimension as the x variable
            # The normalization has the same units as the y

            self.Tc.unit = u.s
            self.tc.unit = u.s
            self.t0.unit = u.s

        
        def evaluate(self, x, Fc, alpha_c, Tc, tc, t0):
            results = np.zeros(len(x))
            x=x-t0
            # Tc=Tc+t0
            # results = Fc * np.exp(alpha_c - (x * alpha_c / Tc)) * np.exp(-tc / x)
            results = Fc * (x / Tc) ** -alpha_c * np.exp(-tc / x)
            results[x<=0]=0
            return results
        
from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation
import astropy.units as u

class HEBS(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            HEBS template
        latex : $  A*f(t-dt) $
        parameters :
            A :
                desc : Normalization
                initial value : 1000.0
                is_normalization : True
                transformation : log10
                min : 1e-10
                max : 1e20
                delta : 0.1

            ts :
                desc : template start time
                initial value : 250
                fix : yes
            te :
                desc : template end time
                min : 0
                initial value : 270
                fix : yes
            dt :
                desc : time delay
                initial value : 1.6
            bint :
                desc : binning
                initial value : 1
                fix : yes
        """


        def _set_units(self, x_unit, y_unit):
            # The index is always dimensionless
            self.A.unit = u.dimensionless_unscaled
            self.ts.unit =  u.s
            self.te.unit = u.s
            self.dt.unit = u.s
            self.bint.unit = u.s


        def evaluate(self, x, A, ts, te, dt, bint):
            hebsdata = np.loadtxt("../../data/lc_data/GRB221009A/hebs-2.txt")
            from scipy.interpolate import interp1d
            from scipy.integrate import quad
            xx = hebsdata[:,0]
            y = hebsdata[:,1]

            y = y-powerlawlc(xx, 0.5e5, -1.5, 230)

            xx = nprebinmean(xx, int(bint/0.05))
            y = nprebin(y, int(bint/0.05))
            
            tcut = (xx-dt > ts) & (xx-dt < te)
            f = interp1d(xx, y-min(y[tcut]))
            def integrand(x):
                return f(x)
            norm, error = quad(integrand, ts, te)

            result=[]
            for xxx in x:
                if xxx-dt > ts and xxx-dt < te:
                    result.append(A/norm*f(xxx-dt))
                else:
                    result.append(0)
            
            return np.array(result)
        
class Powerlaw_x0(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A simple power-law

    latex : $ K~\frac{x}{piv}^{index} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1000.0
            is_normalization : True

            min : 0
            max : 1e4
            delta : 0.1

        piv :

            desc : Pivot value
            initial value : 250
            fix : yes

        index :

            desc : Photon index
            initial value : -2.01
            min : -10
            max : 10
        
        x0 :
            desc : x0
            initial value : 0
            min : -200
            max : 1e3

    tests :
        - { x : 10, function value: 0.01, tolerance: 1e-20}
        - { x : 100, function value: 0.0001, tolerance: 1e-20}

    """

    def _set_units(self, x_unit, y_unit):
        # The index is always dimensionless
        self.index.unit = astropy_units.dimensionless_unscaled

        # The pivot energy has always the same dimension as the x variable
        self.piv.unit = x_unit

        # The normalization has the same units as the y

        self.K.unit = y_unit
        self.x0.unit = x_unit

    # noinspection PyPep8Naming
    def evaluate(self, x, K, piv, index, x0):

        if isinstance(x, astropy_units.Quantity):
            index_ = index.value
            K_ = K.value
            piv_ = piv.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, piv_, x_, index_ = K, piv, x, index
        
        x_=x_-x0

        if x_>0:
            result = nb_func.plaw_eval(x_, K_, index_, piv_)
        else:
            result = 0

        return result * unit_

class StepFunction(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A function which is constant on the interval lower_bound - upper_bound and 0 outside the interval. The
        extremes of the interval are counted as part of the interval.

    latex : $ f(x)=\begin{cases}0 & x < \text{lower_bound} \\\text{value} & \text{lower_bound} \le x \le \text{upper_bound} \\ 0 & x > \text{upper_bound} \end{cases}$

    parameters :

        mean :

            desc : Lower bound for the interval
            initial value : 0

        sigma :

            desc : Upper bound for the interval
            initial value : 5

        value :

            desc : Value in the interval
            initial value : 5.0

    tests :
        - { x : 0.5, function value: 1.0, tolerance: 1e-20}
        - { x : -0.5, function value: 0, tolerance: 1e-20}

    """

    def _set_units(self, x_unit, y_unit):
        # Lower and upper bound has the same unit as x
        self.mean.unit = x_unit
        self.sigma.unit = x_unit

        # value has the same unit as y
        self.value.unit = y_unit

    def evaluate(self, x, mean, sigma, value):
        # The value * 0 is to keep the units right

        result = np.zeros(x.shape) * value * 0

        idx = (x >= mean-sigma/2) & (x <= mean+sigma/2)
        result[idx] = value

        return result
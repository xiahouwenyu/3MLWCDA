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
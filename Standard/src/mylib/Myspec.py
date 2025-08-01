import astropy.units as astropy_units
import astromodels.functions.numba_functions as nb_func
from astromodels.functions.function import (Function1D, FunctionMeta,
                                            ModelAssertionViolation)

try:
    from threeML.config.config import threeML_config

    _has_threeml = True

except ImportError:

    _has_threeml = False

import numpy as np
from past.utils import old_div


from astromodels.utils.logging import setup_logger

log = setup_logger(__name__)

__author__ = 'giacomov'
# DMFitFunction and DMSpectra add by Andrea Albert (aalbert@slac.stanford.edu) Oct 26, 2016

erg2keV = 6.24151e8

class PowerlawM(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A simple power-law

    latex : $ K~\frac{x}{piv}^{index} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1.0
            is_normalization : True

            min : -1e5
            max : 1e5
            delta : 0.1

        piv :

            desc : Pivot value
            initial value : 1
            fix : yes

        index :

            desc : Photon index
            initial value : -2.01
            min : -10
            max : 10

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

    # noinspection PyPep8Naming
    def evaluate(self, x, K, piv, index):

        if isinstance(x, astropy_units.Quantity):
            index_ = index.value
            K_ = K.value
            piv_ = piv.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, piv_, x_, index_ = K, piv, x, index

        result = nb_func.plaw_eval(x_, K_, index_, piv_)

        return result * unit_

class PowerlawN(Function1D, metaclass=FunctionMeta): #            transformation : log10            
    r"""
    description :

        A simple power-law

    latex : $ K~\frac{x}{piv}^{index} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1.0
            is_normalization : True
            min : 1e-15
            max : 1e10
            delta : 0.1

        Kn:

            desc : Normalization of K
            initial value : 1.0
            min : 1e-30
            max : 1e2
            fix : yes

        piv :

            desc : Pivot value
            initial value : 1
            fix : yes

        index :

            desc : Photon index
            initial value : -2.01
            min : -10
            max : 10

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
        self.Kn.unit = astropy_units.dimensionless_unscaled

        self.K.unit = y_unit

    # noinspection PyPep8Naming
    def evaluate(self, x, K, Kn, piv, index):

        if isinstance(x, astropy_units.Quantity):
            index_ = index.value
            K_ = K.value
            Kn_ = Kn.value
            piv_ = piv.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, Kn_, piv_, x_, index_ = K, Kn, piv, x, index

        result = nb_func.plaw_eval(x_, K_*Kn_, index_, piv_)

        return result * unit_
    
class Log_parabolaM(Function1D, metaclass=FunctionMeta):  #            transformation : log10
    r"""
    description :

        A log-parabolic function. NOTE that we use the high-energy convention of using the natural log in place of the
        base-10 logarithm. This means that beta is a factor 1 / log10(e) larger than what returned by those software
        using the other convention.

    latex : $ K \left( \frac{x}{piv} \right)^{\alpha -\beta \log{\left( \frac{x}{piv} \right)}} $

    parameters :

        K :

            desc : Normalization
            initial value : 1.0
            is_normalization : True
            min : -1e5
            max : 1e5

        piv :
            desc : Pivot (keep this fixed)
            initial value : 1
            fix : yes

        alpha :

            desc : index
            initial value : -2.0

        beta :

            desc : curvature (positive is concave, negative is convex)
            initial value : 1.0

    """

    def _set_units(self, x_unit, y_unit):

        # K has units of y

        self.K.unit = y_unit

        # piv has the same dimension as x
        self.piv.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, piv, alpha, beta):

        # print("Receiving %s" % ([K, piv, alpha, beta]))

        xx = np.divide(x, piv)

        try:

            return K * xx ** (alpha - beta * np.log(xx))

        except ValueError:

            # The current version of astropy (1.1.x) has a bug for which quantities that have become
            # dimensionless because of a division (like xx here) are not recognized as such by the power
            # operator, which throws an exception: ValueError: Quantities and Units may only be raised to a scalar power
            # This is a quick fix, waiting for astropy 1.2 which will fix this

            xx = xx.to("")

            return K * xx ** (alpha - beta * np.log(xx))

    @property
    def peak_energy(self):
        """
        Returns the peak energy in the nuFnu spectrum

        :return: peak energy in keV
        """

        # Eq. 6 in Massaro et al. 2004
        # (http://adsabs.harvard.edu/abs/2004A%26A...413..489M)

        return self.piv.value * pow(
            10, old_div(((2 + self.alpha.value) * np.log(10)),
                        (2 * self.beta.value))
        )

class Cutoff_powerlawM(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A power law multiplied by an exponential cutoff

    latex : $ K~\frac{x}{piv}^{index}~\exp{-x/xc} $

    parameters :

        K :

            desc : Normalization (differential flux at the pivot value)
            initial value : 1.0
            is_normalization : True
            min : -1e3
            max : 1e3
            delta : 0.1

        piv :

            desc : Pivot value
            initial value : 1
            fix : yes

        index :

            desc : Photon index
            initial value : -2
            min : -10
            max : 10

        xc :

            desc : Cutoff energy
            initial value : 10.0
            transformation : log10
            min: 1.0

    """

    def _set_units(self, x_unit, y_unit):
        # The index is always dimensionless
        self.index.unit = astropy_units.dimensionless_unscaled

        # The pivot energy has always the same dimension as the x variable
        self.piv.unit = x_unit

        self.xc.unit = x_unit

        # The normalization has the same units as the y

        self.K.unit = y_unit

    # noinspectionq PyPep8Naming

    def evaluate(self, x, K, piv, index, xc):

        if isinstance(x, astropy_units.Quantity):
            index_ = index.value
            K_ = K.value
            piv_ = piv.value
            xc_ = xc.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, piv_, x_, index_, xc_ = K, piv, x, index, xc

        result = nb_func.cplaw_eval(x_, K_, xc_, index_, piv_)

        return result * unit_
    
class SmoothlyBrokenPowerLawM(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A Smoothly Broken Power Law

    Latex : $  $

    parameters :

        K :

            desc : normalization
            initial value : 1
            min : -1e3
            is_normalization : True


        alpha :

            desc : power law index below the break
            initial value : -1
            min : -3
            max : 2

        break_energy:

            desc: location of the peak
            initial value : 300
            fix : no
            min : 10

        break_scale :

            desc: smoothness of the break
            initial value : 2
            min : 0.
            max : 10.
            fix : yes

        beta:

            desc : power law index above the break
            initial value : -2.
            min : -5.0
            max : -1.6

        pivot:

            desc: where the spectrum is normalized
            initial value : 100.
            fix: yes


    """

    def _set_units(self, x_unit, y_unit):

        # norm has same unit as energy
        self.K.unit = y_unit

        self.break_energy.unit = x_unit

        self.pivot.unit = x_unit

        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled
        self.break_scale.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, break_energy, break_scale, beta, pivot):

        if isinstance(x, astropy_units.Quantity):
            alpha_ = alpha.value
            beta_ = beta.value
            K_ = K.value
            pivot_ = pivot.value
            break_energy_ = break_energy.value
            break_scale_ = break_scale.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            K_, pivot_, x_, alpha_, beta_, break_scale_, break_energy_ = (
                K,
                pivot,
                x,
                alpha,
                beta,
                break_scale,
                break_energy,
            )

        result = nb_func.sbplaw_eval(
            x_, K_, alpha_, break_energy, break_scale_, beta_, pivot_
        )

        return result * unit_

class Line_ratio(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        A linear function

    latex : $ b * x + a $

    parameters :

        a :

            desc :  intercept
            initial value : 0

        b :

            desc : coeff
            initial value : 1

    """
    def _set_units(self, x_unit, y_unit):
        # a has units of y_unit / x_unit, so that a*x has units of y_unit
        self.a.unit = y_unit

        # b has units of y
        self.b.unit = y_unit / x_unit

    def evaluate(self, x, a, b):
        return a*(b * x + 1)
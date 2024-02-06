from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation
import astropy.units as u

class SBPL(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  SBPL
        latex : $  F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega) $
        parameters :
            F0 :
                desc : Normalization
                initial value : 100.0
                is_normalization : True
                transformation : log10
                min : 1e-30
                max : 1e3
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
            self.F0.unit = y_unit

            # The normalization has the same units as the y

            self.tb.unit = x_unit
            self.t0.unit = x_unit


        def evaluate(self, x, F0, tb, omega, alpha1, alpha2, t0):

            result = F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega)
            result[x<t0]=0
            return result
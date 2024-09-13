import numpy as np
from astromodels.functions.function import Function3D, FunctionMeta
from astromodels.functions.function import Function2D, FunctionMeta
from past.utils import old_div
from astromodels.utils.angular_distance import angular_distance_fast
from astromodels.utils.angular_distance import angular_distance
import astropy.units as u
from scipy.integrate import quad

class Continuous_injection_diffusion_ellipse2D(Function2D, metaclass=FunctionMeta):
    r"""
        description :

            Positron and electrons diffusing away from the accelerator

        latex : $\left(\frac{180^\circ}{\pi}\right)^2 \frac{1.2154}{\sqrt{\pi^3} r_{\rm diff} ({\rm angsep} ({\rm x, y, lon_0, lat_0})+0.06 r_{\rm diff} )} \, {\rm exp}\left(-\frac{{\rm angsep}^2 ({\rm x, y, lon_0, lat_0})}{r_{\rm diff} ^2} \right)$

        parameters :

            lon0 :

                desc : Longitude of the center of the source
                initial value : 0.0
                min : 0.0
                max : 360.0

            lat0 :

                desc : Latitude of the center of the source
                initial value : 0.0
                min : -90.0
                max : 90.0

            rdiff0 :

                desc : Projected diffusion radius. The maximum allowed value is used to define the truncation radius.
                initial value : 1.0
                min : 0
                max : 20
            incl :

                desc : inclination of semimajoraxis to a line of constant latitude
                initial value : 0.0
                min : -90.0
                max : 90.0
                fix : yes

            elongation :

                desc : elongation of the ellipse (b/a)
                initial value : 1.
                min : 0.1
                max : 10.

        """

    def _set_units(self, x_unit, y_unit, z_unit):

        # lon0 and lat0 and rdiff have most probably all units of degrees. However,
        # let's set them up here just to save for the possibility of using the
        # formula with other units (although it is probably never going to happen)

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit
        self.rdiff0.unit = x_unit

        # Delta is of course unitless
        self.incl.unit = x_unit
        self.elongation.unit = u.dimensionless_unscaled


    def evaluate(self, x, y, lon0, lat0, rdiff0, incl, elongation):

        lon, lat = x,y

        angsep = angular_distance(lon0, lat0, lon, lat)

        rdiff_a=rdiff0

        rdiff_b = rdiff_a * elongation
        ang = np.arctan2(lat - lat0, (lon - lon0) *
                         np.cos(lat0 * np.pi / 180.))
        
        theta = np.arctan2(old_div(np.sin(ang-incl*np.pi/180.),
                        elongation), np.cos(ang-incl*np.pi/180.))

        rdiffs = np.sqrt(rdiff_a ** 2 * np.cos(theta) **
                         2 + rdiff_b ** 2 * np.sin(theta) ** 2)
        
        pi = np.pi

        def pdf(angsep):
            return 1 / (rdiff_a * np.sqrt(elongation) * (angsep + 0.085 * rdiffs)) * np.exp(-1.54*(angsep/rdiffs)**1.52)
        def pdf2(angsep):
            return 2*np.pi*angsep / (rdiff0 * (angsep + 0.085 * rdiff0)) * np.exp(-1.54*(angsep/rdiff0)**1.52)

        integral, _ = quad(pdf2, 0, 100) #/(2*np.pi*angsep)

        # results = np.power(old_div(180.0, pi), 2) * 1.22 / (pi * np.sqrt(pi) * rdiff_a * np.sqrt(
        #     elongation) * (angsep + 0.06 * rdiffs)) * np.exp(old_div(-np.power(angsep, 2), rdiffs ** 2))

        return (180/np.pi)**2*(1/integral)*pdf(angsep)


    def get_boundaries(self):

        # Truncate the function at the max of rdiff allowed

        maximum_rdiff = self.rdiff0.max_value

        min_latitude = max(-90., self.lat0.value - maximum_rdiff)
        max_latitude = min(90., self.lat0.value + maximum_rdiff)

        max_abs_lat = max(np.absolute(min_latitude), np.absolute(max_latitude))

        if max_abs_lat > 89. or old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.)) >= 180.:

            min_longitude = 0.
            max_longitude = 360.

        else:

            min_longitude = self.lon0.value - \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))
            max_longitude = self.lon0.value + \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))

            if min_longitude < 0.:

                min_longitude += 360.

            elif max_longitude > 360.:

                max_longitude -= 360.

        return (min_longitude, max_longitude), (min_latitude, max_latitude)


    def get_total_spatial_integral(self, z=None):
        """
        Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
        needs to be implemented in subclasses.

        :return: an array of values of the integral (same dimension as z).
        """

        if isinstance(z, u.Quantity):
            z = z.value
        return np.ones_like(z)
    
    
class Continuous_injection_diffusion2D(Function2D, metaclass=FunctionMeta):
    r"""
        description :
            Positron and electrons diffusing away from the accelerator

        latex : $\left(\frac{180^\circ}{\pi}\right)^2 \frac{1.2154}{\sqrt{\pi^3} r_{\rm diff} ({\rm angsep} ({\rm x, y, lon_0, lat_0})+0.06 r_{\rm diff} )} \, {\rm exp}\left(-\frac{{\rm angsep}^2 ({\rm x, y, lon_0, lat_0})}{r_{\rm diff} ^2} \right)$

        parameters :

            lon0 :

                desc : Longitude of the center of the source
                initial value : 0.0
                min : 0.0
                max : 360.0
                delta : 0.1

            lat0 :

                desc : Latitude of the center of the source
                initial value : 0.0
                min : -90.0
                max : 90.0
                delta : 0.1

            rdiff0 :

                desc : Projected diffusion radius limited by the cooling time. The maximum allowed value is used to define the truncation radius.
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1
        """

    def _set_units(self, x_unit, y_unit, z_unit):

        # lon0 and lat0 and rdiff have most probably all units of degrees. However,
        # let's set them up here just to save for the possibility of using the
        # formula with other units (although it is probably never going to happen)

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit
        self.rdiff0.unit = x_unit

    def evaluate(self, x, y, lon0, lat0, rdiff0):

        lon, lat = x,y

        angsep = angular_distance(lon0, lat0, lon, lat)


        def pdf(angsep):
            return 1 / (rdiff0 * (angsep + 0.085 * rdiff0)) * np.exp(-1.54*(angsep/rdiff0)**1.52)
        def pdf2(angsep):
            return 2*np.pi*angsep / (rdiff0 * (angsep + 0.085 * rdiff0)) * np.exp(-1.54*(angsep/rdiff0)**1.52)

        integral, _ = quad(pdf2, 0, 100) #/(2*np.pi*angsep)

        return (180/np.pi)**2*(1/integral)*pdf(angsep)

    def get_boundaries(self):

        # Truncate the function at the max of rdiff allowed

        maximum_rdiff = self.rdiff0.max_value

        min_latitude = max(-90., self.lat0.value - maximum_rdiff)
        max_latitude = min(90., self.lat0.value + maximum_rdiff)

        max_abs_lat = max(np.absolute(min_latitude), np.absolute(max_latitude))

        if max_abs_lat > 89. or old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.)) >= 180.:

            min_longitude = 0.
            max_longitude = 360.

        else:

            min_longitude = self.lon0.value - \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))
            max_longitude = self.lon0.value + \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))

            if min_longitude < 0.:

                min_longitude += 360.

            elif max_longitude > 360.:

                max_longitude -= 360.

        return (min_longitude, max_longitude), (min_latitude, max_latitude)


    def get_total_spatial_integral(self, z=None):  
        """
        Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
        needs to be implemented in subclasses.

        :return: an array of values of the integral (same dimension as z).
        """
        
        if isinstance( z, u.Quantity):
            z = z.value
        return np.ones_like( z )

class Beta_function(Function2D, metaclass=FunctionMeta):
    r"""
        description :
            Positron and electrons diffusing away from the accelerator

        latex : $\left(\frac{180^\circ}{\pi}\right)^2 \frac{1.2154}{\sqrt{\pi^3} r_{\rm diff} ({\rm angsep} ({\rm x, y, lon_0, lat_0})+0.06 r_{\rm diff} )} \, {\rm exp}\left(-\frac{{\rm angsep}^2 ({\rm x, y, lon_0, lat_0})}{r_{\rm diff} ^2} \right)$

        parameters :

            lon0 :

                desc : Longitude of the center of the source
                initial value : 0.0
                min : 0.0
                max : 360.0
                delta : 0.1

            lat0 :

                desc : Latitude of the center of the source
                initial value : 0.0
                min : -90.0
                max : 90.0
                delta : 0.1

            rc1 :
                desc : 
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1

            beta1 :
                desc :
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1
        """

    def _set_units(self, x_unit, y_unit, z_unit):

        # lon0 and lat0 and rdiff have most probably all units of degrees. However,
        # let's set them up here just to save for the possibility of using the
        # formula with other units (although it is probably never going to happen)

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit
        self.rc1.unit = x_unit

    def evaluate(self, x, y, lon0, lat0, rc1, beta1):

        lon, lat = x,y

        angsep = angular_distance(lon0, lat0, lon, lat)

        def projected_intensity(R, rc1, beta1):

            # 定义分布函数
            def f1(z):
                return (1 + ((np.sqrt(R**2 + z**2) / rc1)**2))**(-3 * beta1)

            # 计算积分
            I1, _ = quad(lambda z: 2 * f1(z), -5*rc1, 5*rc1)

            return I1

        def pdf(angsep):
            return projected_intensity(angsep, rc1, beta1)
        
        def pdf2(angsep):
            return 2*np.pi*angsep * projected_intensity(angsep, rc1, beta1)

        integral, _ = quad(pdf2, 0, 100) #/(2*np.pi*angsep)

        vectorized_pdf = np.vectorize(pdf)
        result = (180/np.pi)**2 * (1/integral) * vectorized_pdf(angsep)

        return result
        # return (180/np.pi)**2*(1/integral)*pdf(angsep)

    def get_boundaries(self):

        # Truncate the function at the max of rdiff allowed

        maximum_rdiff = self.rc1.max_value

        min_latitude = max(-90., self.lat0.value - maximum_rdiff)
        max_latitude = min(90., self.lat0.value + maximum_rdiff)

        max_abs_lat = max(np.absolute(min_latitude), np.absolute(max_latitude))

        if max_abs_lat > 89. or old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.)) >= 180.:

            min_longitude = 0.
            max_longitude = 360.

        else:

            min_longitude = self.lon0.value - \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))
            max_longitude = self.lon0.value + \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))

            if min_longitude < 0.:

                min_longitude += 360.

            elif max_longitude > 360.:

                max_longitude -= 360.

        return (min_longitude, max_longitude), (min_latitude, max_latitude)


    def get_total_spatial_integral(self, z=None):  
        """
        Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
        needs to be implemented in subclasses.

        :return: an array of values of the integral (same dimension as z).
        """
        
        if isinstance( z, u.Quantity):
            z = z.value
        return np.ones_like( z )
    
class Double_Beta_function(Function2D, metaclass=FunctionMeta):
    r"""
        description :
            Positron and electrons diffusing away from the accelerator

        latex : $\left(\frac{180^\circ}{\pi}\right)^2 \frac{1.2154}{\sqrt{\pi^3} r_{\rm diff} ({\rm angsep} ({\rm x, y, lon_0, lat_0})+0.06 r_{\rm diff} )} \, {\rm exp}\left(-\frac{{\rm angsep}^2 ({\rm x, y, lon_0, lat_0})}{r_{\rm diff} ^2} \right)$

        parameters :

            lon0 :

                desc : Longitude of the center of the source
                initial value : 0.0
                min : 0.0
                max : 360.0
                delta : 0.1

            lat0 :

                desc : Latitude of the center of the source
                initial value : 0.0
                min : -90.0
                max : 90.0
                delta : 0.1

            rc1 :
                desc : 
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1

            beta1 :
                desc :
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1

            rc2 :
                desc : 
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1

            beta2 :
                desc :
                initial value : 1.0
                min : 0
                max : 20
                delta : 0.1
        """

    def _set_units(self, x_unit, y_unit, z_unit):

        # lon0 and lat0 and rdiff have most probably all units of degrees. However,
        # let's set them up here just to save for the possibility of using the
        # formula with other units (although it is probably never going to happen)

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit
        self.rc1.unit = x_unit
        self.rc2.unit = x_unit

    def evaluate(self, x, y, lon0, lat0, rc1, beta1, rc2, beta2):

        lon, lat = x,y

        angsep = angular_distance(lon0, lat0, lon, lat)

        def projected_intensity(R, rc1, beta1, rc2, beta2):

                # 定义分布函数
                def f1(z):
                    return ((1 + ((np.sqrt(R**2 + z**2) / rc1)**2))**(-3/2 * beta1) + (1 + ((np.sqrt(R**2 + z**2) / rc2)**2))**(-3/2 * beta2))**2

                # 计算积分
                I1, _ = quad(lambda z: 2 * f1(z), -5, 5)

                return I1

        def pdf(angsep):
            return projected_intensity(angsep, rc1, beta1, rc2, beta2)
        
        def pdf2(angsep):
            return 2*np.pi*angsep * projected_intensity(angsep, rc1, beta1, rc2, beta2)

        integral, _ = quad(pdf2, 0, 100) #/(2*np.pi*angsep)

        vectorized_pdf = np.vectorize(pdf)
        result = (180/np.pi)**2 * (1/integral) * vectorized_pdf(angsep)

        return result
        # return (180/np.pi)**2*(1/integral)*pdf(angsep

    def get_boundaries(self):

        # Truncate the function at the max of rdiff allowed

        maximum_rdiff = self.rc1.max_value + self.rc2.max_value

        min_latitude = max(-90., self.lat0.value - maximum_rdiff)
        max_latitude = min(90., self.lat0.value + maximum_rdiff)

        max_abs_lat = max(np.absolute(min_latitude), np.absolute(max_latitude))

        if max_abs_lat > 89. or old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.)) >= 180.:

            min_longitude = 0.
            max_longitude = 360.

        else:

            min_longitude = self.lon0.value - \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))
            max_longitude = self.lon0.value + \
                old_div(maximum_rdiff, np.cos(max_abs_lat * np.pi / 180.))

            if min_longitude < 0.:

                min_longitude += 360.

            elif max_longitude > 360.:

                max_longitude -= 360.

        return (min_longitude, max_longitude), (min_latitude, max_latitude)


    def get_total_spatial_integral(self, z=None):  
        """
        Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
        needs to be implemented in subclasses.

        :return: an array of values of the integral (same dimension as z).
        """
        
        if isinstance( z, u.Quantity):
            z = z.value
        return np.ones_like( z )
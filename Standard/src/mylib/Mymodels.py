import numpy as np
from astromodels.functions.function import Function3D, FunctionMeta
from astromodels.functions.function import Function2D, FunctionMeta
from past.utils import old_div
from astromodels.utils.angular_distance import angular_distance_fast
from astromodels.utils.angular_distance import angular_distance
import astropy.units as u
from scipy.integrate import quad

from astropy import wcs
from astropy.coordinates import ICRS, BaseCoordinateFrame, SkyCoord
from astropy.io import fits

import hashlib

from astromodels.utils.logging import setup_logger

log = setup_logger(__name__)


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
    

# class SpatialTemplate_2D(Function2D, metaclass=FunctionMeta):
#     r"""
#         description :
        
#             User input Spatial Template.  Expected to be normalized to 1/sr
        
#         latex : $ hi $
        
#         parameters :
        
#             K :
        
#                 desc : normalization
#                 initial value : 1
#                 fix : yes
#             hash :
                
#                 desc: hash of model map [needed for memoization]
#                 initial value: 1
#                 fix: yes
#             ihdu:
#                 desc: header unit index of fits file
#                 initial value: 0
#                 fix: True 
#                 min: 0
        
#         properties:
#             fits_file:
#                 desc: fits file to load
#                 defer: True
#                 function: _load_file
#             frame:
#                 desc: coordinate frame
#                 initial value: icrs
#                 allowed values:
#                     - icrs
#                     - galactic
#                     - fk5
#                     - fk4
#                     - fk4_no_e
#     """
    
#     def _set_units(self, x_unit, y_unit, z_unit):
        
#         self.K.unit = z_unit
    
#     # This is optional, and it is only needed if we need more setup after the
#     # constructor provided by the meta class
    
#     # def _setup(self):
        
#         # self._frame = "icrs"
#         # self._fitsfile = None
#         # self._map = None
    
#     def _load_file(self):

#         self._fitsfile=self.fits_file.value
        
#         with fits.open(self._fitsfile) as f:
    
#             self._wcs = wcs.WCS( header = f[int(self.ihdu.value)].header )
#             self._map = f[int(self.ihdu.value)].data
              
#             self._nX = f[int(self.ihdu.value)].header['NAXIS1']
#             self._nY = f[int(self.ihdu.value)].header['NAXIS2']

#             #note: map coordinates are switched compared to header. NAXIS1 is coordinate 1, not 0. 
#             #see http://docs.astropy.org/en/stable/io/fits/#working-with-image-data
#             assert self._map.shape[1] == self._nX, "NAXIS1 = %d in fits header, but %d in map" % (self._nX, self._map.shape[1])
#             assert self._map.shape[0] == self._nY, "NAXIS2 = %d in fits header, but %d in map" % (self._nY, self._map.shape[0])
            
#             #test if the map is normalized as expected
#             area = wcs.utils.proj_plane_pixel_area( self._wcs )
#             dOmega = (area*u.deg*u.deg).to(u.sr).value
#             total = self._map.sum() * dOmega

#             if not np.isclose( total, 1,  rtol=1e-2):
#                 log.warning("2D template read from {} is normalized to {} (expected: 1)".format(self._fitsfile, total) )
            
#             #hash sum uniquely identifying the template function (defined by its 2D map array and coordinate system)
#             #this is needed so that the memoization won't confuse different SpatialTemplate_2D objects.
#             h = hashlib.sha224()
#             h.update( self._map)
#             h.update( repr(self._wcs).encode('utf-8') )
#             self.hash = int(h.hexdigest(), 16)
            

#     # def to_dict(self, minimal=False):

#     #      data = super(Function2D, self).to_dict(minimal)

#     #      if not minimal:
         
#     #         data['extra_setup'] = {"_fitsfile": self._fitsfile, "_frame": self._frame }
  
#     #      return data
        
    
#     # def set_frame(self, new_frame):
#     #     """
#     #         Set a new frame for the coordinates (the default is ICRS J2000)
            
#     #         :param new_frame: a coordinate frame from astropy
#     #         :return: (none)
#     #         """
#     #     assert new_frame.lower() in ['icrs', 'galactic', 'fk5', 'fk4', 'fk4_no_e' ]
                
#     #     self._frame = new_frame
    
#     def evaluate(self, x, y, K, hash, ihdu):
                  
#         # We assume x and y are R.A. and Dec
#         coord = SkyCoord(ra=x, dec=y, frame=self.frame.value, unit="deg")
        
#         #transform input coordinates to pixel coordinates; 
#         #SkyCoord takes care of necessary coordinate frame transformations.
#         Xpix, Ypix = coord.to_pixel(self._wcs)
        
#         Xpix = np.atleast_1d(Xpix.astype(int))
#         Ypix = np.atleast_1d(Ypix.astype(int))
        
#         # find pixels that are in the template ROI, otherwise return zero
#         #iz = np.where((Xpix<self._nX) & (Xpix>=0) & (Ypix<self._nY) & (Ypix>=0))[0]
#         iz = (Xpix<self._nX) & (Xpix>=0) & (Ypix<self._nY) & (Ypix>=0)
#         out = np.zeros_like(Xpix).astype(float)
#         out[iz] = self._map[Ypix[iz], Xpix[iz]]
        
#         return np.multiply(K, out)

#     def get_boundaries(self):
    
#         # if self._map is None:
            
#         #     self.load_file(self._fitsfile)
          
#         #We use the max/min RA/Dec of the image corners to define the boundaries.
#         #Use the 'outside' of the pixel corners, i.e. from pixel 0 to nX in 0-indexed accounting.
    
#         Xcorners = np.linspace(0, self._nX+1, 1000) #np.array( [0, 0,        self._nX, self._nX] )
#         Ycorners = np.linspace(0, self._nY+1, 1000) #np.array( [0, self._nY, 0,        self._nY] )
#         Xcorners = np.concatenate((Xcorners, np.array( [0, 0,        self._nX, self._nX] )))
#         Ycorners = np.concatenate((Ycorners, np.array( [0, self._nY, 0,        self._nY] )))

#         corners = SkyCoord.from_pixel( Xcorners, Ycorners, wcs=self._wcs, origin = 0).transform_to(self.frame.value)  
     
#         min_lon = min(corners.ra.degree)
#         max_lon = max(corners.ra.degree)
        
#         min_lat = min(corners.dec.degree)
#         max_lat = max(corners.dec.degree)
        
#         return (min_lon, max_lon), (min_lat, max_lat)



#     def get_total_spatial_integral(self, z=None):  
#         """
#         Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
#         needs to be implemented in subclasses.

#         :return: an array of values of the integral (same dimension as z).
#         """

#         if isinstance( z, u.Quantity):
#             z = z.value
#         return np.multiply(self.K.value,np.ones_like( z ) )
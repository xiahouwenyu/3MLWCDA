from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import astropy_mpl_style, imshow_norm
from astropy.coordinates import Angle


def edm2gal(ra1,dec1):
    coord=SkyCoord(ra1*u.deg,dec1*u.deg,frame='icrs').transform_to('galactic')
    l,b=coord.l.degree,coord.b.degree
    return l,b

def gal2edm(l1,b1):
    coord=SkyCoord(l1*u.deg,b1*u.deg,frame='galactic').transform_to('icrs')
    ra,dec=coord.ra.degree,coord.dec.degree
    return ra,dec
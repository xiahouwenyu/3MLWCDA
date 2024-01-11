from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time

import healpy as hp

import numpy as np


def edm2gal(ra1,dec1):
    coord=SkyCoord(ra1*u.deg,dec1*u.deg,frame='icrs').transform_to('galactic')
    l,b=coord.l.degree,coord.b.degree
    return l,b

def gal2edm(l1,b1):
    coord=SkyCoord(l1*u.deg,b1*u.deg,frame='galactic').transform_to('icrs')
    ra,dec=coord.ra.degree,coord.dec.degree
    return ra,dec

def icrs2altaz(
    mjd = 59861.55347211,
    longitude = 100.138794639,
    latitude = 29.357656306,
    source_ra = 288.263,
    source_dec = 19.803
    ):
    # 将MJD转换为Time对象
    obs_time = Time(mjd, format='mjd')

    # 创建EarthLocation对象，表示观测位置
    obs_location = EarthLocation(lat=latitude, lon=longitude)

    # 创建SkyCoord对象，表示目标源的天球坐标
    source_coord = SkyCoord(ra=source_ra, dec=source_dec, unit="deg", frame="icrs")

    # 计算目标源在给定时间和位置上的天顶坐标
    altaz_coord = source_coord.transform_to(AltAz(obstime=obs_time, location=obs_location))

    # 获取天顶角和方位角
    zenith_angle = 90-altaz_coord.alt.deg
    azimuth_angle = altaz_coord.az.deg

    # print(f"天顶角：{zenith_angle} 度")
    # print(f"方位角：{azimuth_angle} 度")
    return zenith_angle,azimuth_angle

def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map

    Parameters
    ----------
    m : map or array of maps
      map(s) to be rotated
    coord : sequence of two character
      First character is the coordinate system of m, second character
      is the coordinate system of the output map. As in HEALPIX, allowed
      coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Example
    -------
    The following rotate m from galactic to equatorial coordinates.
    Notice that m can contain both temperature and polarization.
    >>>> change_coord(m, ['G', 'C'])
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]

def icrs2j200(RA, DEC):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # 定义ICRS坐标
    icrs_coord = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, frame='icrs')

    # 将ICRS坐标转换为J2000
    j2000_coord = icrs_coord.transform_to('fk5')  # 'fk5' 是J2000的默认坐标系

    ra = j2000_coord.ra.value
    dec = j2000_coord.dec.value
    return ra,dec

def distance(ra1, dec1, ra2, dec2):
    # 将角度转换为弧度
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)

    # 计算角距离
    cos_theta = np.sin(dec1_rad) * np.sin(dec2_rad) + np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad)
    
    # 通过反余弦函数获取角距离（弧度）
    theta_rad = np.arccos(cos_theta)
    
    # 将弧度转换为度
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg

def skyangle(ra1,dec1,ra2,dec2):
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)
    angle= np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))*180./np.pi
    return angle
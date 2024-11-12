from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time

import healpy as hp

import numpy as np


def edm2gal(ra1,dec1):
    """
        赤道转银道
    """
    coord=SkyCoord(ra1*u.deg,dec1*u.deg,frame='icrs').transform_to('galactic')
    l,b=coord.l.degree,coord.b.degree
    return l,b

def gal2edm(l1,b1):
    """
        银道转赤道
    """
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
    """
        赤道转地平

        Parameters:
            longitude, latitude: 地球经纬度
            
        Returns:
            zenith_angle,azimuth_angle
    """
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
    """
        天球角距离

        Parameters:
            
        Returns:
            角距离 degree
    """
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
    """
        天球角距离

        Parameters:
            
        Returns:
            角距离 degree
    """
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)
    angle= np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))*180./np.pi
    return angle

def hms_to_deg(ra_hours, ra_minutes, ra_seconds, dec_degrees, dec_minutes, dec_seconds):
    """
        转换时角为角度

        Parameters:
            
        Returns:
            ra_degrees, dec_degrees
    """ 
    ra_degrees = (ra_hours + ra_minutes/60 + ra_seconds/3600) * 15

    # 转换赤纬为角度
    sign = -1 if dec_degrees < 0 else 1
    dec_degrees = abs(dec_degrees)
    dec_degrees += dec_minutes/60 + dec_seconds/3600
    dec_degrees *= sign

    return ra_degrees, dec_degrees
def deg_to_hms(ra_degrees, dec_degrees):
    """
        转换角度为时角

        Parameters:
            
        Returns:
            ra_hours, ra_minutes, ra_seconds, dec_degrees, dec_minutes, dec_seconds
    """
    # 转换赤经为时角
    ra_hours = int(ra_degrees // 15)
    ra_minutes = int((ra_degrees % 15) * 4)
    ra_seconds = ((ra_degrees % 15) * 4 - ra_minutes) * 60

    # 转换赤纬为度、分、秒
    sign = -1 if dec_degrees < 0 else 1
    dec_degrees = abs(dec_degrees)
    dec_deg = int(dec_degrees)
    dec_minutes = int((dec_degrees - dec_deg) * 60)
    dec_seconds = ((dec_degrees - dec_deg) * 60 - dec_minutes) * 60
    dec_deg *= sign

    return ra_hours, ra_minutes, ra_seconds, dec_deg, dec_minutes, dec_seconds

def eql2hcs(ha, dn):
    import math
    first = True
    latmydet = 29.357656306
    ch, sh, cd, sd =np. cos(ha),np.sin(ha),np.cos(dn),np.sin(dn)
    if (first):
        sinlat = np.sin(np.radians(latmydet))
        coslat = np.cos(np.radians(latmydet))
        first = False

    x = -ch*cd*sinlat + sd*coslat
    y = -sh*cd
    z =  ch*cd*coslat + sd*sinlat

    r = np.sqrt(x*x+y*y)
    if (r==0.):
        az = 0.
    else:
        az = math.atan2(y,x)

    if (az<0.):
        az = az + 2*np.pi

    ze = np.pi/2 - math.atan2(z,r)
    return ze, az

def angle_to_kpc(angle, distance_mpc, unit='arcsec'):
    """
    将张角转换为千秒差距 (kpc)。

    参数:
    angle (float): 角度值，可以是弧度、弧度秒或度。
    distance_mpc (float): 目标天体的距离，单位是兆秒差距（Mpc）。
    unit (str): 角度单位，'arcsec' 表示弧度秒，'degree' 表示度，'radian' 表示弧度。默认是 'arcsec'。

    返回:
    float: 张角对应的kpc值。
    """
    if unit == 'arcsec':
        # 将弧度秒转换为弧度
        angle_rad = np.deg2rad(angle / 3600.0)
    elif unit == 'degree':
        # 将度转换为弧度
        angle_rad = np.deg2rad(angle)
    elif unit == 'radian':
        angle_rad = angle
    else:
        raise ValueError("unit 必须是 'arcsec', 'degree', 或 'radian'")

    # 将弧度转换为kpc
    distance_kpc = distance_mpc * 1e3
    kpc = distance_kpc * np.tan(angle_rad)

    return kpc

def kpc_to_angle(kpc, distance_mpc, unit='arcsec'):
    """
    将千秒差距 (kpc) 转换为张角。

    参数:
    kpc (float): 千秒差距值 (kpc)。
    distance_mpc (float): 目标天体的距离，单位是兆秒差距（Mpc）。
    unit (str): 输出角度的单位，'arcsec' 表示弧度秒，'degree' 表示度，'radian' 表示弧度。默认是 'arcsec'。

    返回:
    float: kpc对应的张角值，单位由unit参数指定。
    """
    # 将距离转换为千秒差距 (kpc)
    distance_kpc = distance_mpc * 1e3
    
    # 计算弧度
    angle_rad = np.arctan(kpc / distance_kpc)
    
    # 根据指定的单位进行转换
    if unit == 'arcsec':
        angle = np.rad2deg(angle_rad) * 3600.0  # 弧度转换为弧度秒
    elif unit == 'degree':
        angle = np.rad2deg(angle_rad)  # 弧度转换为度
    elif unit == 'radian':
        angle = angle_rad  # 保持弧度
    else:
        raise ValueError("unit 必须是 'arcsec', 'degree', 或 'radian'")
    
    return angle

def redshift_to_mpc(z, H0=70.0):
    """
    将红移z转换为以Mpc为单位的距离。

    参数:
    z (float): 红移值
    H0 (float): 哈勃常数，单位为 km/s/Mpc。默认值为 70 km/s/Mpc。

    返回:
    float: 距离，单位为 Mpc。
    """
    # 光速，单位为 km/s
    c = 299792.458
    
    # 使用哈勃定律计算距离
    distance_mpc = c * z / H0
    
    return distance_mpc

from astropy.cosmology import Planck18 as cosmo  # 使用Planck18宇宙学模型
from astropy.cosmology import z_at_value  # 导入 z_at_value 函数
import astropy.units as u  # 导入 astropy.units

def Mpc_to_redshift(distance_mpc):
    """
    将距离（Mpc）转换为红移（redshift）。
    
    参数:
    - distance_mpc: 距离，以Mpc为单位
    
    返回:
    - 红移值
    """
    # 确保距离具有Mpc的单位
    distance = distance_mpc * u.Mpc
    
    # 计算红移
    redshift = z_at_value(cosmo.comoving_distance, distance)
    return redshift

def redshift_to_Mpc(redshift):
    """
    将红移（redshift）转换为距离（Mpc）。
    
    参数:
    - redshift: 红移值
    
    返回:
    - 距离，以Mpc为单位
    """
    # 计算共动距离并转换为Mpc
    distance_mpc = cosmo.comoving_distance(redshift).to(u.Mpc).value
    return distance_mpc
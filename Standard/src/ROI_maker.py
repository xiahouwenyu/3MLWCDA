#!/usr/bin/python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from ROOT import TFile,TTree
import healpy as hp
import numpy as np
import matplotlib
from pylab import cm
from scipy.optimize import curve_fit
from array import array
from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.io import fits as pyfits
from astropy.wcs import wcs
import sys
import argparse

nside=2**10
npix=hp.nside2npix(nside)
pixarea = 4 * np.pi/npix


pixIdx = hp.nside2npix(nside) # number of pixels I can get from this nside
pixIdx = np.arange(pixIdx) # pixel index numbers
new_lats = hp.pix2ang(nside, pixIdx)[0] # thetas I need to populate with interpolated theta values
new_lons = hp.pix2ang(nside, pixIdx)[1] # phis, same
c_icrs = SkyCoord(ra=new_lons*180/np.pi*u.degree, dec=90*u.degree-new_lats*180/np.pi*u.degree, frame='icrs')
c_l=c_icrs.galactic.l.deg
c_b=c_icrs.galactic.b.deg

for i,gl in enumerate(range(0,180,5)):
    print(i,gl)
    c_gal = SkyCoord(l=gl*u.degree, b=10*u.degree, frame='galactic')
    RA_center=c_gal.icrs.ra.deg
    Dec_center=c_gal.icrs.dec.deg
    signal=np.zeros(npix,dtype=np.float64)
    mask = ( (c_l< gl + 5) & (c_l > gl - 5) & (c_b <=15.) & (c_b>5) &(new_lats<110/180*np.pi )  & (new_lats > 10/180*np.pi ) )
    signal[mask]=1
    hp.write_map("Gl_%d_%d_Gb_5_15.fits"%(gl-5,gl+5),signal,overwrite=True)

    hp.mollview(signal,title="ROI",norm='hist')
    hp.graticule()
    plt.savefig("Gl_%d_%d_Gb_5_15.pdf"%(gl-5,gl+5))

    if(i==0):
        with open('ROI_5_15.txt', "w") as f:
            f.write("%.2f %.2f %.2f %.2f ROI/Gl_%d_%d_Gb_5_15.fits"%(RA_center,Dec_center,gl-5,gl+5,gl-5,gl+5))
            f.write("\n")
    else:
        with open('ROI_5_15.txt', "a") as f:
            f.write("%.2f %.2f %.2f %.2f ROI/Gl_%d_%d_Gb_5_15.fits"%(RA_center,Dec_center,gl-5,gl+5,gl-5,gl+5))
            f.write("\n")


import matplotlib, sys
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
#from PyQt5 import QtCore, QtGui, uic
import matplotlib.pyplot as plt
sys.path.append('/cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/lib/python3.7/site-packages/threeML')
from threeML import *
silence_warnings()
sys.path.append('/home/lhaaso/gmxiang/lib/pip_lib')
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
import numpy as np 
import healpy as hp
import argparse

# maptree = "/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/2021032202_Cocoon_bin123.root"
# response = "/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/DR_crabPSF_newmap_pinc_neomc_1pe_bin1to4-6_bin2to78_bin12to9-11_bin13to6-11.root"
#ra_Cocoon, dec_Cocoon = 307.17, 41.17
#ra_crab, dec_crab = 40.67, -0.013  #crab
#ra_crab, dec_crab =  83.63,22.02
# ra_crab, dec_crab =  42.19,60.35
# ra_crab, dec_crab =  42.38, 60.37    #J0248
# ra_crab, dec_crab = 194.9529,27.9806 #Coma
# ra_crab, dec_crab = 49.9507,41.5117    #Perseus
# ra_crab, dec_crab = 186.63,12.72       #Virgo
ra_crab, dec_crab =  279.37, -6.96       #G25

lon_crab = np.radians(ra_crab)
colat_crab = np.radians(90-dec_crab)
vec_crab = hp.ang2vec(colat_crab,lon_crab)
data_radius = 6.0  # in degree 
model_radius = 8.0
#vec=hp.ang2vec(np.radians(90-ra_crab),np.radians(dec_crab))
pixid=hp.query_disc(2**10,vec_crab,np.radians(int(data_radius)))
print(len(pixid))
#roi = HealpixConeROI(data_radius=data_radius, model_radius=model_radius, ra=ra_crab, dec=dec_crab)
#roi=HealpixMapROI(ra=ra_Cocoon,dec=dec_Cocoon,data_radius=data_radius,model_radius=model_radius, roifile='/home/lhaaso/tangruiyi/analysis/cocoonstuff/roi.fits')
#WCDA = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
#bin_lists=["1","2"]
#WCDA.set_active_measurements(0,9)
#pixid=roi.active_pixels(2**10)



p = argparse.ArgumentParser(description="create fits")
p.add_argument("-n",dest="n")
p.add_argument("-o",dest="o")
args = p.parse_args()
name=args.n
out=args.o
a=np.loadtxt(name)
print(len(a))
if len(a) != len(pixid):
    raise Exception("input dimention do not equal dimention of output!!!")

S=[]
ipx=0
nside=2**10
npix=hp.nside2npix(nside)
for i in range(npix):
    if i in pixid:
        S.append(a[ipx])
        ipx=ipx+1
    else:
        S.append(hp.UNSEEN)
S = hp.ma(S)
hp.write_map("sigts_%s.fits.gz"%out,S,overwrite=True)


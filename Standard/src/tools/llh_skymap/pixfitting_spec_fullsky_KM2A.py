import matplotlib, sys
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
#from PyQt5 import QtCore, QtGui, uic
import matplotlib.pyplot as plt
from threeML import *
silence_warnings()
try:
    from hawc_hal import HAL, HealpixConeROI, HealpixMapROI
    from hawc_hal.psf_fast.psf_convolutor import PYFFTW_AVAILABLE
    PYFFTW_AVAILABLE = False
except:
    from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
import healpy as hp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
silence_warnings()
from threeML.minimizer.minimization import (CannotComputeCovariance,CannotComputeErrors,FitFailed,LocalMinimizer)
from functions import Powerlaw as PowLaw
import threeML
from scipy.optimize import curve_fit
import argparse

def go(args):
    maptree =args.mtfile#"/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/2021032202_Cocoon_bin123.root"
    response = args.rsfile#"/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/DR_crabPSF_newmap_pinc_neomc_1pe_bin1to4-6_bin2to78_bin12to9-11_bin13to6-11.root"
    #ra_Cocoon, dec_Cocoon = 307.17, 41.17
    # ra_crab, dec_crab = 83.694, 21.98  #crab
    data_radius = 6.5  # in degree 
    model_radius = 8.0
    #Splitiing the sky into 768 equal-solid-angle areas as the pixels ranged in nested healpix map with nside=8
    no=args.area
    #vec_center=hp.pix2vec(2**3,no,nest=True)
    clat_c,lon_c=hp.pix2ang(2**3,no,nest=True)
    dec_c=90-clat_c*180/np.pi
    ra_c=lon_c*180/np.pi
    print(ra_c, dec_c)
    # if dec_c<=-20:
    #     dec_c=-15
    #     data_radius = 15.0  # in degree 
    #     model_radius = 17.0
    # if dec_c>=80:
    #     dec_c=75
    #     data_radius = 15.0  # in degree 
    #     model_radius = 17.0
    
    name=args.name
    # roi=HealpixMapROI(ra=ra_Cocoon,dec=dec_Cocoon,data_radius=data_radius,model_radius=model_radius, roifile='/home/lhaaso/tangruiyi/analysis/cocoonstuff/roi.fits')
    roi = HealpixConeROI(data_radius=data_radius, model_radius=model_radius, ra=ra_c, dec=dec_c)
    WCDA = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
    
    # Use from bin 1 to bin 9
    # WCDA.set_active_measurements(0,6)
    WCDA.set_active_measurements(4,13)

#    pixid=roi.active_pixels(roi._original_nside)
    #pixid=roi.active_pixels(1024)
    pix_nest=np.linspace(16384*no,16384*(no+1)-1,16384,dtype=int)
    pixid=hp.nest2ring(2**10,pix_nest)
  # for i in range(len(pixid)):
       # print(i)
       # pid=pixid[i]
    spectrum=PowLaw()
        #ra_pix , dec_pix = hp.pix2ang(1024,pid,lonlat=True) 
    source=PointSource("Pixel",
                           ra=ra_c,
                           dec=dec_c,
                           spectral_shape=spectrum)
    # fluxUnit=1./(u.TeV* u.cm**2 * u.s)
        #source.position.ra=ra_pix
       # source.position.ra.fix=True
        #source.position.dec=dec_pix
      #  source.position.dec.fix=True 

    #WCDA
    # spectrum.K=1e-12*fluxUnit
    # spectrum.K.fix=False
    # spectrum.K.bounds=(-1e-13*fluxUnit, 1e-10*fluxUnit)
    # spectrum.piv= 3.*u.TeV
    # spectrum.piv.fix=True
    # spectrum.index=-2.6
    # spectrum.index.fix=True

    #KM2A
    fluxUnit=1e-9
    spectrum.K=1e-16 *fluxUnit
    spectrum.K.fix=False
    spectrum.K.bounds=(-1e-16*fluxUnit, 1e-14*fluxUnit)
    spectrum.K.delta=1e-16*fluxUnit
    spectrum.piv= 50.*u.TeV
    spectrum.piv.fix=True
    spectrum.index=-3.6
    spectrum.index.fix=True
    
    model=Model(source)
    WCDA.psf_integration_method="fast"

#        actbin=args.actBin
    for i in range(len(pixid)):
#    for pid in range(args.StartPix,args.StopPix):    
        print(i)
        pid=pixid[i]
        ra_pix , dec_pix = hp.pix2ang(1024,pid,lonlat=True)
        if(dec_pix<=-20. or dec_pix>=80.):
            sig=hp.UNSEEN
            with open("/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/skytxt2/sig_no%i.txt"%no,"a+") as fs:
                fs.write(str(pid)+" "+str(sig)+" "+str(hp.UNSEEN)+"\n")
            continue

        if dec_c<=-13.5:
            roi = HealpixConeROI(data_radius=np.max([1.5,dec_pix+19.5]), model_radius=np.max([2,dec_pix+19.5]), ra=ra_pix, dec=dec_pix)
            WCDA = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
            WCDA.set_active_measurements(4,13)
            WCDA.psf_integration_method="fast"
        elif dec_c>=73.5:
            roi = HealpixConeROI(data_radius=np.max([1.5,79.5-dec_pix]), model_radius=np.max([2,79.5-dec_pix]), ra=ra_pix, dec=dec_pix)
            WCDA = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
            WCDA.set_active_measurements(4,13)
            WCDA.psf_integration_method="fast"


        source.position.ra=ra_pix
        source.position.ra.fix=True
        source.position.dec=dec_pix
        source.position.dec.fix=True
        # WCDA.set_active_measurements(0,6)
        # WCDA.set_active_measurements(2,7)
        WCDA.set_active_measurements(4,13)
        data = DataList(WCDA)
        jl = JointLikelihood(model, data, verbose=False)
        jl.set_minimizer("ROOT")
        try:
            param_df, like_df = jl.fit()
        except Exception as e: # (threeML.minimizer.minimization.CannotComputeCovariance,OverflowError,FitFailed,RuntimeError)
            print(no, pid, e, ra_pix, dec_pix)
            sig=0 #hp.UNSEEN
            K_fitted=0
            errid=pid
            with open("/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/skytxt2/erridlist_%s.txt"%name,"a+") as fs:
                fs.write(str(errid)+"\n")
        else:
            results = jl.results
            # #WCDA.get_log_like()
            TS=jl.compute_TS("Pixel",like_df)
            ts=TS.values[0][2]
            # print("TS:",ts)
            # ts_list.append(ts)
            K_fitted=results.optimized_model.Pixel.spectrum.main.Powerlaw.K.value
            # sig = K_fitted
            if(ts>=0):
                if(K_fitted>=0):
                    sig=np.sqrt(ts)
                else:
                    sig=-np.sqrt(ts)
            else:
                sig=0
            
          #  sig_list.append(sig)
        with open("/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/skytxt2/sig_no%i.txt"%no,"a+") as fs:
            fs.write(str(pid)+" "+str(sig)+" "+str(K_fitted)+"\n")
        
#    np.savetxt(r'siglist_%s.txt'%name,sig_list)
#    np.savetxt(r'erridlist_%s.txt'%name,errid_list,fmt='%i')

#np.savetxt('ts_list2.txt',ts_list)
#np.savetxt('err_list2.txt',errid_list)    
#np.savetxt('sig_list2.txt',sig_list)
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Example spectral fit")
    p.add_argument("-a", "--area", dest="area",type=int,help="which area", default=0)
    p.add_argument("-m", "--maptreefile", dest="mtfile",help="MapTree ROOT file", default="/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/2021032202_Cocoon_bin123.root")
    p.add_argument("-r", "--responsefile", dest="rsfile",help="detector response ROOT file", default="/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/DR_crabPSF_newmap_pinc_neomc_1pe_bin1to4-6_bin2to78_bin12to9-11_bin13to6-11.root")
    p.add_argument("--actBin", dest="actBin", default=2, type=int,help="Starting analysis bin [0..13]")
    p.add_argument("--name",default="crab",type=str,help="out put figure name")
    p.add_argument("--StartPix", dest="StartPix", default=0, type=int,help="Starting analysis pixel")
    p.add_argument("--StopPix", dest="StopPix", default=1000, type=int,help="Stopping analysis pixel")
    args = p.parse_args()

    go(args)



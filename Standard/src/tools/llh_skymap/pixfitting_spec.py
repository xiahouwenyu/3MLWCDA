import matplotlib, sys
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
#from PyQt5 import QtCore, QtGui, uic
import matplotlib.pyplot as plt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.append('/cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/lib/python3.7/site-packages/threeML')
from threeML import *
silence_warnings()
sys.path.append('/home/lhaaso/gmxiang/lib/pip_lib')
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
from threeML.utils.progress_bar import trange
# os.system('rm -rf siglist_%s.txt; touch siglist_%s.txt'%(name,name))
def go(args):
    maptree =args.mtfile#"/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/2021032202_Cocoon_bin123.root"
    response = args.rsfile#"/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/DR_crabPSF_newmap_pinc_neomc_1pe_bin1to4-6_bin2to78_bin12to9-11_bin13to6-11.root"
    #ra_Cocoon, dec_Cocoon = 307.17, 41.17
    
    # ra_crab, dec_crab =  42.38, 60.37 #J0248
    # ra_crab, dec_crab = 194.9529,27.9806 #Coma
    # ra_crab, dec_crab = 49.9507,41.5117    #Perseus
    # ra_crab, dec_crab = 186.63,12.72       #Virgo
    # ra_crab, dec_crab =  279.37, -6.96       #G25
    ra_crab, dec_crab =  14,63.5             #J0057


    colat_crab = np.radians(90-dec_crab)
    lon_crab = np.radians(ra_crab)
    vec_crab = hp.ang2vec(colat_crab,lon_crab)
    data_radius = 3.0  # in degree 
    model_radius = 5.0
    roi = HealpixConeROI(data_radius=data_radius, model_radius=model_radius, ra=ra_crab, dec=dec_crab)
    name=args.name
    #roi=HealpixMapROI(ra=ra_Cocoon,dec=dec_Cocoon,data_radius=data_radius,model_radius=model_radius, roifile='/home/lhaaso/tangruiyi/analysis/cocoonstuff/roi.fits')
    WCDA = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
    # Use from bin 1 to bin 9
    # WCDA.set_active_measurements(0,5)
    nside=2**10
#    pixid=roi.active_pixels(roi._original_nside)
#    pixid=roi.active_pixels(1024)
    pixid=hp.query_disc(nside,vec_crab,np.radians(int(data_radius)))

  # for i in range(len(pixid)):
       # print(i)
       # pid=pixid[i]
    spectrum=PowLaw()
        #ra_pix , dec_pix = hp.pix2ang(1024,pid,lonlat=True) 
    source=PointSource("Pixel",
                           ra=ra_crab,
                           dec=dec_crab,
                           spectral_shape=spectrum)
    fluxUnit=1./(u.TeV* u.cm**2 * u.s)
        #source.position.ra=ra_pix
       # source.position.ra.fix=True
        #source.position.dec=dec_pix
      #  source.position.dec.fix=True 
    spectrum.K=0 *fluxUnit
    spectrum.K.fix=False
    spectrum.K.bounds=(-1e-12*fluxUnit, 1e-12*fluxUnit)
    spectrum.piv= 3.*u.TeV
    spectrum.piv.fix=True
    spectrum.index=-2.4
    spectrum.index.fix=True
    WCDA.psf_integration_method="fast"
    model=Model(source)


    actbin=args.actBin
    print("bin>",actbin)
    WCDA.set_active_measurements(actbin,5)
    data = DataList(WCDA)
    jl = JointLikelihood(model, data, verbose=False)
    jl.set_minimizer("MINUIT")
    for i in trange(len(pixid), desc="pixid progress"):
#    for pid in range(args.StartPix,args.StopPix):    
        print(i)
        pid=pixid[i]
        ra_pix , dec_pix = hp.pix2ang(1024,pid,lonlat=True)
        source.position.ra=ra_pix
        source.position.ra.fix=True
        source.position.dec=dec_pix
        source.position.dec.fix=True
        try:
            param_df, like_df = jl.fit()
        except (threeML.minimizer.minimization.CannotComputeCovariance,OverflowError,FitFailed,RuntimeError):
            sig=hp.UNSEEN
            errid=pid
            with open("erridlist_%s.txt"%name,"a+") as fs:
                fs.write(str(errid)+"\n")
        else:
            results = jl.results
            #WCDA.get_log_like()
            TS=jl.compute_TS("Pixel",like_df)
            ts=TS.values[0][2]
            print("TS:",ts)
            #ts_list.append(ts)
            K_fitted=results.optimized_model.Pixel.spectrum.main.Powerlaw.K.value
            if(ts>=0):
                if(K_fitted>=0):
                    sig=np.sqrt(ts)
                else:
                    sig=-np.sqrt(ts)
            else:
                sig=0
          #  sig_list.append(sig)
        
        with open("/data/home/cwy/Science/3MLWCDA0.91/Standard/src/tools/llh_skymap/siglist_%s.txt"%name,"a+") as fs:
            fs.write(str(sig)+"\n")
        
#    np.savetxt(r'siglist_%s.txt'%name,sig_list)
#    np.savetxt(r'erridlist_%s.txt'%name,errid_list,fmt='%i')

#np.savetxt('ts_list2.txt',ts_list)
#np.savetxt('err_list2.txt',errid_list)    
#np.savetxt('sig_list2.txt',sig_list)
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Example spectral fit")
    p.add_argument("-m", "--maptreefile", dest="mtfile",help="MapTree ROOT file", default="/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/2021032202_Cocoon_bin123.root")
    p.add_argument("-r", "--responsefile", dest="rsfile",help="detector response ROOT file", default="/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/DR_crabPSF_newmap_pinc_neomc_1pe_bin1to4-6_bin2to78_bin12to9-11_bin13to6-11.root")
    p.add_argument("--actBin", dest="actBin", default=2, type=int,help="Starting analysis bin [0..13]")
    p.add_argument("--name",default="crab",type=str,help="out put figure name")
    p.add_argument("--StartPix", dest="StartPix", default=0, type=int,help="Starting analysis pixel")
    p.add_argument("--StopPix", dest="StopPix", default=1000, type=int,help="Stopping analysis pixel")
    args = p.parse_args()

    go(args)
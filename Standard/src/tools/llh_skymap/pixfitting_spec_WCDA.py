import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


from threeML import *
silence_warnings()
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
import healpy as hp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
silence_warnings()
from threeML.minimizer.minimization import (CannotComputeCovariance,CannotComputeErrors,FitFailed,LocalMinimizer)
from functions import Powerlaw as PowLaw
import argparse
from tqdm import tqdm
from multiprocessing import Pool

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Example spectral fit")
    p.add_argument("-part", dest="ii",help="part", default=0, type=int)
    p.add_argument("-ra", dest="ra",help="RA", default= 83.63, type=float)
    p.add_argument("-dec", dest="dec",help="DEC", default= 66.01, type=float)
    p.add_argument("-radius", dest="radius",help="radius", default= 6, type=float)
    p.add_argument("-m", "--maptreefile", dest="mtfile",help="MapTree ROOT file", default="/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/2021032202_Cocoon_bin123.root")
    p.add_argument("-r", "--responsefile", dest="rsfile",help="detector response ROOT file", default="/home/lhaaso/tangruiyi/analysis/cocoonstuff/maptreeinc/DR_crabPSF_newmap_pinc_neomc_1pe_bin1to4-6_bin2to78_bin12to9-11_bin13to6-11.root")
    p.add_argument("--s", dest="s", default=0, type=int,help="Starting analysis bin [0..13]")
    p.add_argument("--e", dest="e", default=5, type=int,help="Starting analysis bin [0..13]")
    p.add_argument("--name",default="crab",type=str,help="out put figure name")
    p.add_argument("--jc",dest="jc", default=10,type=int,help="进程数")
    p.add_argument("--sn",dest="sn", default=100,type=int,help="单作业负载")
    p.add_argument("--o",dest="outdir", default="/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap",type=str,help="输出文件夹")
    args = p.parse_args()

    outdir= args.outdir
    maptree = args.mtfile
    response = args.rsfile
    ra_crab, dec_crab = args.ra, args.dec
    data_radius = args.radius
    model_radius = data_radius+3
    print(maptree, response, ra_crab, dec_crab, data_radius)

    roi = HealpixConeROI(data_radius=data_radius, model_radius=model_radius, ra=ra_crab, dec=dec_crab)

    WCDA = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
    spectrum=PowLaw()
    source=PointSource("Pixel",
                           ra=ra_crab,
                           dec=dec_crab,
                           spectral_shape=spectrum)
    fluxUnit=1./(u.TeV* u.cm**2 * u.s)
    spectrum.K=0 *fluxUnit
    spectrum.K.fix=False
    spectrum.K.bounds=(-1e-12*fluxUnit, 1e-12*fluxUnit)
    spectrum.piv= 3.*u.TeV
    spectrum.piv.fix=True
    spectrum.index=-2.4
    spectrum.index.fix=True
    WCDA.psf_integration_method="fast"
    model=Model(source)
    quiet_mode()
    WCDA.set_active_measurements(args.s,args.e)
    data = DataList(WCDA)
    jl = JointLikelihood(model, data, verbose=False)
    jl.set_minimizer("MINUIT")

    def getllhskymap(pixels):
        rr = []
        quiet_mode()
        for pid in tqdm(pixels):
            ra_pix , dec_pix = hp.pix2ang(1024,pid,lonlat=True)
            source.position.ra=ra_pix
            source.position.ra.fix=True
            source.position.dec=dec_pix
            source.position.dec.fix=True
            try:
                param_df, like_df = jl.fit(quiet=True)
            except (CannotComputeCovariance,OverflowError,FitFailed,RuntimeError):
                rr.append([pid, hp.UNSEEN])
            else:
                results = jl.results
                TS=jl.compute_TS("Pixel",like_df)
                ts=TS.values[0][2]
                # print("TS:",ts)
                K_fitted=results.optimized_model.Pixel.spectrum.main.Powerlaw.K.value
                if(ts>=0):
                    if(K_fitted>=0):
                        sig=np.sqrt(ts)
                    else:
                        sig=-np.sqrt(ts)
                else:
                    sig=hp.UNSEEN
                rr.append([pid, sig])
        return rr

    nside=2**10
    colat_crab = np.radians(90-dec_crab)
    lon_crab = np.radians(ra_crab)
    vec_crab = hp.ang2vec(colat_crab,lon_crab)
    pixel=hp.query_disc(nside,vec_crab,np.radians(int(data_radius)))
    ii = args.ii
    sn = args.sn
    try:
        pixel=pixel[ii*sn:(ii+1)*sn]
    except:
        pixel=pixel[ii*sn:-1]
    plong = args.jc
    num = int(len(pixel)/plong)
    par = []
    for i in range(plong+1):
        try:
            par.append(pixel[i*num : (i+1)*num])
        except:
            par.append(pixel[i*num:-1])
    
    results = []
    with Pool(processes=plong) as pool:
        result = pool.map(getllhskymap, par)
        results.append(np.array(result))
    results=np.array(results)
    name=args.name
    if not os.path.exists(f'{outdir}/sourcetxt/WCDA_{name}/'):
        os.system(f'mkdir {outdir}/sourcetxt/WCDA_{name}/')
    np.save(f"{outdir}/sourcetxt/WCDA_{name}/{name}_{ii}.npy", results)
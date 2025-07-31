import subprocess
import os 

import Mycoord
import inspect

import os
import numpy as np
import healpy as hp
from multiprocessing import Pool, current_process
from threeML import *
# 依然推荐使用 notebook 版本的 tqdm，它在 vscode 中表现最好
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    from hawc_hal import HAL, HealpixConeROI, HealpixMapROI
    # from hawc_hal.psf_fast.psf_convolutor import PYFFTW_AVAILABLE
    # from hawc_hal.obsolete.ts_map import ParallelTSmap
    PYFFTW_AVAILABLE = False
except:
    from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
    from WCDA_hal.obsolete.ts_map import ParallelTSmap
from Myspec import PowerlawM as PowLaw
# 导入 ipyparallel
import ipyparallel as ipp

libdir = os.path.dirname(os.path.dirname(inspect.getfile(Mycoord)))
# libdir = subprocess.run("pwd -P", shell=True, capture_output=True, text=True).stdout.replace("\n","")

def runllhskymap(roi, maptree, response, ra1, dec1, data_radius, region_name, detector="WCDA", ifres=False, jc=10, sn=1000, s=None,e=None):
    """
        交作业跑显著性天图, 结果用Mysigmap里面的getllhskymap查看

        Parameters:
            ifres: 是否标记为残差显著性天图
            jc: 每一个作业进程数
            sn: 每一个作业跑多少个pixel?
        Returns:
            >>> None
    """ 
    parts = int(len(roi.active_pixels(1024))/sn)+1
    if ifres:
        region_name=region_name+"_res"
    if detector=="WCDA":
        if s is None:
            s=0
        if e is None:
            e=6
        os.system(f"cd {libdir}/tools/llh_skymap/; rm -rf ./output/*; ./runwcdaall.sh {maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
        print(f"cd {libdir}/tools/llh_skymap/; rm -rf ./output/*; ./runwcdaall.sh {maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
    elif detector=="KM2A":
        if s is None:
            s=4
        if e is None:
            e=13
        os.system(f"cd {libdir}/tools/llh_skymap/; rm -rf ./output/*; ./runkm2aall.sh {maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
        print(f"cd {libdir}/tools/llh_skymap/; rm -rf ./output/*; ./runkm2aall.sh {maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")

jl_global = None
source_global = None

def init_worker(jl, source):
    global jl_global, source_global
    jl_global = jl
    source_global = source
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

def getllh_for_one_pixel(pid):
    ra_pix, dec_pix = hp.pix2ang(1024, pid, lonlat=True)
    source_global.position.ra = ra_pix
    source_global.position.dec = dec_pix
    try:
        param_df, like_df = jl_global.fit(quiet=True)
        results = jl_global.results
        TS = jl_global.compute_TS("Pixel", like_df)
        ts = TS.values[0][2]
        K_fitted = results.optimized_model.Pixel.spectrum.main.Powerlaw.K.value
        if ts >= 0:
            sig = np.sqrt(ts) if K_fitted >= 0 else -np.sqrt(ts)
        else:
            sig = hp.UNSEEN
        return [pid, sig]
    except Exception as e:
        return [pid, hp.UNSEEN]
    
def runllhskymap_mp(roi, maptree, response, ra1, dec1, data_radius, region_name, Modelname, detector="WCDA", ifres=False, jc=30, s=None, e=None, ifplot=False, index=None, indexf=True, name = None):
    """
        交作业跑显著性天图，使用多进程并行计算

        Parameters:
            ifres: 是否标记为残差显著性天图
            jc: 每一个作业进程数
        Returns:
            >>> None
    """
    # 这部分函数保持不变
    silence_logs()
    silence_warnings()
    outdir = f"{libdir}/../res/{region_name}/{Modelname}"
    if ifres:
        region_name=region_name+"_res"
    if name is None:
        name = f"{region_name}"
    # 硬编码参数保持不变
    ra, dec = ra1, dec1
    radius = data_radius
    mtfile = maptree
    rsfile = response
    s, e = s, e
    if detector=="WCDA":
        if s is None:
            s=0
        if e is None:
            e=6
        piv=3
        fluxUnit=1e-9
        flux = 1e-13*fluxUnit
        fluxb = (-1e-14*fluxUnit, 1e-11*fluxUnit)
        if index is None:
            index = -2.6
    elif detector=="KM2A":
        if s is None:
            s=4
        if e is None:
            e=13
        piv=50
        fluxUnit=1e-9
        flux = 1e-16*fluxUnit
        fluxb = (-1e-17*fluxUnit, 1e-14*fluxUnit)
        if index is None:
            index = -3.6
    
    # 数据加载和模型构建部分保持不变
    print("Loading data and building model (ONCE!)...")
    # roi = HealpixConeROI(data_radius=radius, model_radius=radius + 1, ra=ra, dec=dec)
    WCDA = HAL("WCDA", mtfile, rsfile, roi, flat_sky_pixels_size=0.17)
    spectrum = PowLaw()
    source = PointSource("Pixel", ra=ra, dec=dec, spectral_shape=spectrum)
    
    spectrum.K=flux 
    spectrum.K.fix=False
    spectrum.K.bounds=fluxb
    spectrum.K.delta=1e-16*fluxUnit
    spectrum.piv= piv*u.TeV
    spectrum.piv.fix=True
    spectrum.index=index
    spectrum.index.fix=indexf
    model = Model(source)
    WCDA.set_active_measurements(s, e)
    data = DataList(WCDA)
    jl = JointLikelihood(model, data, verbose=False)
    jl.set_minimizer("ROOT")
    print("Data and model loaded.")

    # 获取所有像素部分保持不变
    nside = 2**10
    vec_crab = hp.ang2vec(np.radians(90 - dec), np.radians(ra))
    all_pixels = hp.query_disc(nside, vec_crab, np.radians(int(radius)))
    print(f"Total pixels to compute: {len(all_pixels)}")

    # ==================== 最终的、最高效的并行与进度条解决方案 ====================
    chunksize, extra = divmod(len(all_pixels), jc * 4)
    if chunksize == 0:
        chunksize = 1
    print(f"Using {jc} processes with a chunksize of {chunksize}...")

    results = []
    with Pool(processes=jc, initializer=init_worker, initargs=(jl, source)) as pool:
        # 创建一个 tqdm 实例
        pbar = tqdm(total=len(all_pixels), desc="Fitting Pixels")
        
        # 使用 pool.imap_unordered 来获得最佳性能
        # 它会在任何一个任务完成时立即返回结果，确保核心利用率最大化
        for result in pool.imap_unordered(getllh_for_one_pixel, all_pixels, chunksize=chunksize):
            # 每收到一个结果，就将其添加到列表中
            results.append(result)
            # 并手动更新进度条
            pbar.update(1)
        
        # 关闭进度条
        pbar.close()
    # ========================================================================

    results_array = np.array(results)
    # 为了安全起见，我们根据第一列（pid）对结果进行排序，以确保像素顺序是固定的
    results_array = results_array[results_array[:, 0].argsort()]
    nside = 1024
    npix = hp.nside2npix(nside)
    skymap = np.full(npix, hp.UNSEEN)
    skymap[results_array[:, 0].astype(int)] = results_array[:, 1]
    skymap = hp.ma(skymap)
    outpath = f"{outdir}/{detector}_{name}_fullmap.fits"
    hp.write_map(f"{outdir}/{detector}_{name}_fullmap.fits", skymap, overwrite=True)
    # np.save(outpath, results_array)
    print(f"All done. Full results saved to {outpath}")
    if ifplot:
        print("Plotting the significance map...")
        plt.figure(figsize=(12, 8))
        hp.mollview(skymap, title="Significance Map", unit="sigma", cmap="viridis") #, coord=['G', 'C'], rot=[ra, dec], min=-5, max=5
        hp.graticule()
        # Show the plot
        plt.show()
    activate_logs()
    activate_warnings()
    return results_array

def runllhskymap_ipy(roi, maptree, response, ra1, dec1, data_radius, region_name, Modelname,
                            detector="WCDA", ifres=0, s=None, e=None, pixelsize = 0.1, index=-2.63, indexf=False):
    """
        交作业跑显著性天图，使用 ipyparallel 并行计算

        Parameters:
            ifres: 是否标记为残差显著性天图
            jc: 每一个作业进程数
            sn: 每一个作业跑多少个pixel?
        Returns:
            >>> None
    """
    outdir = f"{libdir}/../res/{region_name}/{Modelname}"
    if ifres:
        region_name=region_name+"_res"
    # 这里的参数保持不变
    if detector=="WCDA":
        if s is None:
            s=0
        if e is None:
            e=6
        piv=3
    elif detector=="KM2A":
        if s is None:
            s=4
        if e is None:
            e=13
        piv=50

    threeML_config["parallel"]["use_parallel"] = True
    threeML_config["parallel"]["use_joblib"] = True
    threeML_config["parallel"]["profile_name"] = "slurm"
    # 1. (可选但推荐) 明确连接到你的集群，以确认一切正常
    try:
        rc = ipp.Client(profile='slurm')
        print(f"成功连接到 {len(rc.ids)} 个计算引擎。")
        print("并行计算环境已激活。")
    except Exception as e:
        print(f"无法连接到 ipyparallel 集群: {e}")
        print("将以串行模式运行，可能会非常慢或因资源不足而失败。")
    ra, dec = ra1, dec1
    radius = data_radius
    mtfile = maptree
    rsfile = response
    s, e = s, e
    P = ParallelTSmap(mtfile, rsfile, ra, dec, 2*radius/pixelsize, 2*radius/pixelsize, pixelsize,s,e,piv=piv,index=index, indexf=indexf, projection="AIT", roi_radius=radius)
    map = P.go()
    fig = P.plot()
    P.to_fits(f"{outdir}/ipy_res.fits")
    return map
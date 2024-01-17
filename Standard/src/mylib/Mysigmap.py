from tqdm import tqdm
import healpy as hp
import numpy as np

import matplotlib, sys
# sys.path.append(__file__[:-12])
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

import Mymap as mt

import ROOT

import copy

import root_numpy as rn

import matplotlib.colors as mcolors

from scipy.optimize import curve_fit

from Mycoord import *

from Myspeedup import libdir, runllhskymap

import MapPalette


def getmap(WCDA, roi, name="J0248", signif=17, smoothsigma = [0.42, 0.32, 0.25, 0.22, 0.18, 0.15], 
           save = False, 
           binc="all",
           stack=[],
           modelindex=None,
           pta=[], exta=[],
           smooth=False,
            stack_sigma=None
           ):  # sourcery skip: default-mutable-arg, low-code-quality# sourcery skip: default-mutable-arg
    """Get counts map.

        Args:
            pta=[1,0,1], exta=[0,0]: if you have 3 pt sources and 2 ext sources, and you only want to keep 1st and 3st sources,you do like this.
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    
    #Initialize
    amap = []
    nside=2**10
    npix=hp.nside2npix(nside)
    pixarea = 4 * np.pi/npix
    pixIdx = np.arange(npix)
    pixid=roi.active_pixels(1024)

    signal=np.full(npix, hp.UNSEEN, dtype=np.float64)
    background=np.full(npix, hp.UNSEEN, dtype=np.float64)
    modelmap=np.full(npix, hp.UNSEEN, dtype=np.float64)
    modelbkg=np.full(npix, hp.UNSEEN, dtype=np.float64)
    alpha=np.full(npix, hp.UNSEEN, dtype=np.float64)

    new_lats = hp.pix2ang(nside, pixIdx)[0]
    new_lons = hp.pix2ang(nside, pixIdx)[1]
    mask = ((-new_lats + np.pi/2 < -20./180*np.pi) | (-new_lats + np.pi/2 > 80./180*np.pi))

    if binc=="all":
        binc=WCDA._active_planes

    for bin in binc:
        smooth_sigma=smoothsigma[int(bin)]
        active_bin = WCDA._maptree._analysis_bins[bin]
        if modelindex:
            model=WCDA._get_expectation(active_bin,bin,modelindex[0],modelindex[1])
        else:
            model=WCDA._get_expectation(active_bin,bin,0,0)
            for i,pt in enumerate(pta):
                if not pt:
                    model += WCDA._get_expectation(active_bin,bin,i+1,0)
                    if i != 0:
                        model -= WCDA._get_expectation(active_bin,bin,i,0)
            
            for i,ext in enumerate(exta):
                if not ext:
                    model += WCDA._get_expectation(active_bin,bin,0,i+1)
                    if i != 0:
                        model -= WCDA._get_expectation(active_bin,bin,0,i)

        obs_raw=active_bin.observation_map.as_partial()
        bkg_raw=active_bin.background_map.as_partial()
        res_raw=obs_raw-model
        for i,pix in enumerate(tqdm(pixid)):
            signal[pix]=obs_raw[i]
            background[pix]=bkg_raw[i]
            modelmap[pix]=model[i]
            modelbkg[pix]=model[i]+bkg_raw[i]
            theta, phi = hp.pix2ang(nside, pix)
            theta = np.pi/2 - theta
            alpha[pix]=2*smooth_sigma*1.51/60./np.sin(theta) #

        print("Mask all")
        signal=hp.ma(signal)
        background=hp.ma(background)
        modelmap=hp.ma(modelmap)
        modelbkg=hp.ma(modelbkg)
        alpha=hp.ma(alpha)

        if smooth:
            print("Smooth Sig")
            signal_smoothed=hp.sphtfunc.smoothing(signal,sigma=np.radians(smooth_sigma))
            signal_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(signal,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

            print("Smooth bkg")
            background_smoothed=hp.sphtfunc.smoothing(background,sigma=np.radians(smooth_sigma))
            background_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(background,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

            print("Smooth Modelbkg")
            modelbkg_smoothed=hp.sphtfunc.smoothing(modelbkg,sigma=np.radians(smooth_sigma))
            modelbkg_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(modelbkg,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea
        else:
            signal_smoothed=np.array([])
            signal_smoothed2=np.array([])
            background_smoothed=np.array([])
            background_smoothed2=np.array([])
            modelbkg_smoothed=np.array([])
            modelbkg_smoothed2=np.array([])

        if save:
            print("Save!")
            hp.mollview(signal_smoothed,title="Mollview image RING",norm='hist',unit='Excess')
            hp.graticule()
            plt.savefig("../res/%s_excess_nHit0%s_%.2f.pdf"%(name, bin, smooth_sigma))
            hp.write_map("../data/%s_nHit0%s_%.2f.fits.gz"%(name, bin, smooth_sigma),[signal, background, signal_smoothed, background_smoothed, signal_smoothed2, background_smoothed2, modelmap, alpha],overwrite=True)

        amap.append([signal, background, modelbkg, 
                     signal_smoothed, background_smoothed, modelbkg_smoothed,
                     signal_smoothed2, background_smoothed2, modelbkg_smoothed2,
                     modelmap, alpha])
    if stack != []:
        summap = copy.deepcopy(amap)
        for i, bin in enumerate(binc):
        # for i, weight in enumerate(stack):
            weight=stack[int(bin)]
            for j in range(10):
                summap[i][j] *= weight
                if j in [6, 7, 8]:
                    summap[i][j] *= weight
        outmap = [np.ma.sum([bin[i] for bin in summap],axis=0) for i in tqdm(range(11))]
        # outmap[-1] = np.ma.sqrt(np.ma.sum([bin[-1]**2*stack[i] for i,bin in enumerate(amap) if i<6],axis=0))
        if stack_sigma:
            smooth_sigma=stack_sigma
        else:
            print("Set stack_sigma automatelly!!!")
            stack_sigma=smoothsigma[len(WCDA._maptree._analysis_bins)] #int(list(WCDA._maptree._analysis_bins.keys())[-1])+1
        for i,pix in enumerate(tqdm(pixid)):
            theta, phi = hp.pix2ang(nside, pix)
            theta = np.pi/2 - theta
            alpha[pix]=2*stack_sigma*1.51/60./np.sin(theta)
        alpha=hp.ma(alpha)
        outmap[-1]=alpha

        for mapp in outmap:
            mapp.fill_value=hp.UNSEEN
        amap.append(outmap)
    return amap

def stack_map(map, stack=None):
    """stack map together.

        Args:
            stack: weight, usually signal to noise ratio.
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    if stack is None:
        return map
    summap = copy.deepcopy(map)
    for i, weight in enumerate(stack):
        for j in range(11):
            summap[i][j] *= weight
            if j in [6, 7, 8]:
                summap[i][j] *= weight
    outmap = [np.ma.sum([bin[i] for bin in summap],axis=0) for i in tqdm(range(10))]
    outmap[-1] = np.ma.sqrt(np.ma.sum([bin[-1]**2*stack[i] for i,bin in enumerate(map) if i<6],axis=0))
    for mapp in outmap:
        mapp.fill_value=hp.UNSEEN
    return outmap

def smoothmap(mapall, smooth_sigma = 0.2896):
    """Get smooth map.

        Args:
            smooth_sigma:
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    nside=2**10
    npix=hp.nside2npix(nside)
    pixarea = 4 * np.pi/npix
    # for i in tqdm([0,1,2]):
    #     mapall[i] = hp.ma(mapall[i])

    print("Smooth Sig")
    mapall[3]=hp.sphtfunc.smoothing(mapall[0],sigma=np.radians(smooth_sigma))
    mapall[6]=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(mapall[0],sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    print("Smooth bkg")
    mapall[4]=hp.sphtfunc.smoothing(mapall[1],sigma=np.radians(smooth_sigma))
    mapall[7]=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(mapall[1],sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    print("Smooth Modelbkg")
    mapall[5]=hp.sphtfunc.smoothing(mapall[2],sigma=np.radians(smooth_sigma))
    mapall[8]=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(mapall[2],sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    print("Mask all")
    for i in tqdm(range(3,8)):
        mapall[i] = hp.ma(mapall[i])

    return mapall


import math
def Draw_ellipse(e_x, e_y, a, e, e_angle, color, linestyle, alpha=0.5, coord="C"):
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    b=a*np.sqrt(1-e**2)
    for angles in angles_circle:
        or_x = a * np.cos(angles)
        or_y = b * np.sin(angles)
        length_or = np.sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + e_angle/180*np.pi
        new_x = e_x + length_or * np.cos(new_theta) #
        new_y = e_y + length_or * np.sin(new_theta)
        dnew_x = new_x-e_x
        new_x = e_x+dnew_x/np.cos(np.radians(new_y))
        x.append(new_x)
        y.append(new_y)
    if coord=="G":
        x,y = edm2gal(x,y)
    plt.plot(x,y, color=color, linestyle=linestyle,alpha=alpha)


def high_pass_filter(image, cutoff_freq):
    # 进行二维傅里叶变换
    f_transform = np.fft.fft2(image)
    
    # 将零频率分量移到中心
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 获取图像大小
    rows, cols = image.shape
    
    # 创建一个高通滤波器
    high_pass_filter = np.ones((rows, cols))
    center_row, center_col = rows // 2, cols // 2
    high_pass_filter[center_row - cutoff_freq:center_row + cutoff_freq, 
                     center_col - cutoff_freq:center_col + cutoff_freq] = 0
    
    # 进行傅里叶逆变换
    filtered_transform_shifted = f_transform_shifted * high_pass_filter
    filtered_transform = np.fft.ifftshift(filtered_transform_shifted)
    filtered_image = np.abs(np.fft.ifft2(filtered_transform))
    
    return filtered_image


def drawfits(fits_file_path = '/data/home/cwy/Science/3MLWCDA/Standard/res/S147/S147_mosaic.fits', fig=None, vmin=None, vmax=None, drawalpha=False, iffilter=False, cmap=plt.cm.Greens, cutl=0.2, cutu=1, filter=1, alpha=1):
    from astropy.io import fits
    from astropy.wcs import WCS
# fits_file_path = '/data/home/cwy/Science/3MLWCDA/Standard/res/S147/S147_mosaic.fits'; vmin=-15; vmax=30
    # 打开 FITS 文件
    hdul = fits.open(fits_file_path)
    print(hdul.info())

    # 获取数据和坐标信息
    data = hdul[0].data
    wcs = WCS(hdul[0].header)

    shape = wcs.array_shape
    a = wcs.pixel_to_world(0, 0)
    b = wcs.pixel_to_world(shape[1], shape[0])
    print(wcs, shape, a,b)

    # 关闭 FITS 文件
    hdul.close()

    # 检测坐标系类型
    if "RA" in wcs.wcs.ctype[0] and "DEC" in wcs.wcs.ctype[1]:
        # 如果包含 "RA" 和 "DEC"，则是赤道坐标
        xlabel = 'RA (J2000)'
        ylabel = 'Dec (J2000)'
    elif "GLON" in wcs.wcs.ctype[0] and "GLAT" in wcs.wcs.ctype[1]:
        # 如果包含 "GLON" 和 "GLAT"，则是银道坐标
        xlabel = 'Galactic Longitude'
        ylabel = 'Galactic Latitude'
    else:
        # 如果无法判断，默认使用 "X-axis" 和 "Y-axis"
        xlabel = 'X-axis'
        ylabel = 'Y-axis'

    data[np.isnan(data)]=0
    # 绘制图像

    if (not vmin) or (not vmax):
        vmin = data.min()
        vmax = data.max()
    if fig:
        ax = fig.gca()
        if not drawalpha:
            from matplotlib.colors import Normalize

            alphas = Normalize(vmin, vmax, clip=True)(data)
            alphas = np.clip(alphas, cutl, cutu)
            alphas[alphas<=cutl]=0
            if iffilter:
                alphas = high_pass_filter(alphas, filter)
                alphas = Normalize(vmin, vmax, clip=True)(alphas)
                alphas = np.clip(alphas, cutl, cutu)
            alphas[alphas<=cutl]=0

            colors = Normalize(vmin, vmax)(data)
            colors = cmap(colors)

            colors[..., -1] = alphas

            im = ax.imshow(colors, origin='lower', extent=[a.ra.value, b.ra.value, a.dec.value, b.dec.value], vmin=0, vmax=(cutl+cutu)/2, interpolation='bicubic', alpha=alpha)
            return fig
        else:
            im = ax.imshow(data, cmap=cmap, origin='lower', extent=[a.ra.value, b.ra.value, a.dec.value, b.dec.value], vmin=vmin, vmax=vmax, interpolation='bicubic', alpha=alpha)
            return fig
    else:
        fig, ax = plt.subplots(subplot_kw={'projection': wcs})
        im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='bicubic')
        # 添加坐标轴标签
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # 添加坐标网格
        ax.coords.grid()

        # 显示坐标轴的度标尺
        ax.coords[0].set_format_unit('degree')
        ax.coords[1].set_format_unit('degree')

        # 添加色标
        cbar = plt.colorbar(im, ax=ax, label='Intensity')
        plt.show()
        return fig, wcs, data
    
def heal2fits(map, name, ra_min = 82, ra_max = 88, xsize=0.1, dec_min=26, dec_max=30, ysize=0.1, nside=1024, ifplot=False, ifnorm=True, check=False, alpha=1):
    from astropy.io import fits
    from astropy.wcs import WCS
    # 将RA和DEC范围转换为SkyCoord对象
    ra=np.arange(ra_min, ra_max, xsize); dec=np.arange(dec_min, dec_max, ysize)
    print(len(ra), len(dec))
    X,Y = np.meshgrid(ra, dec)
    coords = SkyCoord(ra=X, dec=Y, unit="deg", frame="icrs")
    # 使用SkyCoord对象获取对应的HEALPix像素索引
    npix=hp.nside2npix(nside)
    pixarea = 4*np.pi/npix
    pix_indices = hp.ang2pix(nside, coords.ra.degree, coords.dec.degree, lonlat=True)
    map[map==hp.UNSEEN]=0
    if ifplot:
        plt.imshow(map[pix_indices], extent=[ra_min, ra_max, dec_min, dec_max], origin="lower", aspect='auto')
        plt.gca().invert_xaxis()
        plt.colorbar()

    # 创建一个新的FITS文件，其中包含指定方形区域的数据
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = int(len(ra))
    header["NAXIS2"] = int(len(dec))
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = ra.mean()
    header["CRVAL2"] = dec.mean()
    header["CRPIX1"] = header["NAXIS1"] / 2
    header["CRPIX2"] = header["NAXIS2"] / 2
    header["CD1_1"] = xsize*np.cos(np.radians(header["CRVAL2"]))
    header["CD2_2"] = ysize

    wcs = WCS(header)

    # 创建一个空的二维数组，用于存储提取的数据
    extracted_data = np.zeros((header["NAXIS2"], header["NAXIS1"]))
    
    # 将HEALPix数据的指定区域复制到新数组中
    extracted_data = map[pix_indices]

    if ifnorm:
        extracted_data = extracted_data-extracted_data.min()
        extracted_data = extracted_data**alpha
        area = np.radians(xsize)*np.radians(ysize)*np.ones((len(dec), len(ra)))*np.cos(np.radians(Y))
        integral = extracted_data*area #/pixarea*area
        extracted_data = extracted_data/integral.sum()

    if check:
        plt.figure()
        plt.imshow(area, extent=[ra_min, ra_max, dec_min, dec_max], origin="lower")
        plt.gca().invert_xaxis()

    # 将提取的数据保存到FITS文件
    fits.writeto(name, np.array(extracted_data.data), header, overwrite=True)

def drawmap(region_name, Modelname, sources, map, ra1, dec1, rad=6, contours=[3, 5], save=False, savename=None, cat={ "LHAASO": [0, "P"],"TeVCat": [0, "s"], "PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"], "4FGL": [0, "d"], "YMC": [0, "^"], "GYMC":[0, "v"], "WR":[0, "X"], "size": 20, "markercolor": "grey", "labelcolor": "black", "angle": 60, "catext": 1}, color="Fermi", colorlabel="", legend=True, Drawdiff=False, ifdrawfits=False, fitsfile=None, vmin=None, vmax=None, drawalpha=False, iffilter=False, cmap=plt.cm.Greens, cutl=0.2, cutu=1, filter=1, alphaf=1,     
    colors=['tab:red',
            'tab:blue',
            'tab:green',
            'tab:purple',
            'tab:orange',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']
        ):  # sourcery skip: extract-duplicate-method
    """Draw a healpix map with fitting results.

        Args:
            sources: use function get_sources() to get the fitting results.
            cat: catalog to draw. such as {"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"]}, first in [1,"s"] is about if add a label?
                "o" is the marker you choose.
                The catalog you can choose is:
                     TeVCat/3FHL/4FGL/PSR/SNR/AGN/QSO/Simbad
        Returns:
            ----------
            >>> fig
    """
    from matplotlib.patches import Ellipse
    fig = mt.hpDraw(region_name, Modelname, map,ra1,dec1,
            radx=rad/np.cos(dec1/180*np.pi),rady=rad,
            colorlabel=colorlabel, contours=contours, save=False, cat=cat, color=color, Drawdiff=Drawdiff
            )
    ax = plt.gca()
    # colors=list(mcolors.TABLEAU_COLORS.keys()) #CSS4_COLORS
    # colors=['tab:red',
    #         'tab:blue',
    #         'tab:green',
    #         'tab:purple',
    #         'tab:orange',
    #         'tab:brown',
    #         'tab:pink',
    #         'tab:olive',
    #         'tab:cyan',
    #         'tab:gray']
    # colors[i]
    colors = MapPalette.colorall
    i=0
    ifasymm=False
    for sc in sources.keys():
        source = sources[sc]
        for par in source.keys():
            if par in ['lon0', 'ra']:
                x = source[par][2]
                xeu = source[par][3]
                xel = source[par][4]     
            elif par in ["lat0","dec"]:
                y = source[par][2]
                yeu = source[par][3]
                yel = source[par][4]
            elif par in ["sigma","rdiff0","radius", "a"]:
                sigma = source[par][2]
                sigmau = source[par][3]
                sigmal = source[par][4]
            elif par in ["e", "elongation"]:
                ifasymm=True
                e = source[par][2]
                eu = source[par][3]
                el = source[par][4]
            elif par in ["theta", "incl"]:
                ifasymm=True
                theta = source[par][2]
                thetau = source[par][3]
                thetal = source[par][4]


        if sources[sc]['type'] == 'extended source' and not ifasymm:
            plt.errorbar(x, y, yerr=(np.abs([yel]), np.abs([yeu])), xerr=(np.abs([xel]), np.abs([xeu])), fmt='o',markersize=2,capsize=1,elinewidth=1,color=colors[i], label=sc)
            error_ellipse = Ellipse((x, y), width=sigma/np.cos(np.radians(y)), height=sigma, edgecolor=colors[i], fill=False,linestyle="-")
            ax.add_artist(error_ellipse)
            error_ellipse = Ellipse((x, y), width=(sigma+sigmau)/np.cos(np.radians(y)), height=sigma+sigmau, edgecolor=colors[i], fill=False,linestyle="--", alpha=0.5)
            ax.add_artist(error_ellipse)
            error_ellipse = Ellipse((x, y), width=(sigma-abs(+sigmal))/np.cos(np.radians(y)), height=sigma-abs(sigmal), edgecolor=colors[i], fill=False,linestyle="--", alpha=0.5)
            ax.add_artist(error_ellipse)
        elif ifasymm:
            plt.errorbar(x, y, yerr=(np.abs([yel]), np.abs([yeu])), xerr=(np.abs([xel]), np.abs([xeu])), fmt='o',markersize=2,capsize=1,elinewidth=1,color=colors[i], label=sc)
            print(x,y,sigma,e,theta)
            Draw_ellipse(x,y,sigma,e,theta,colors[i],"-")
        else:
            plt.errorbar(x, y, yerr=(np.abs([yel]), np.abs([yeu])), xerr=(np.abs([xel]), np.abs([xeu])), fmt='o',markersize=2,capsize=1,elinewidth=1,color=colors[i],label=sc)
        i+=1
        # if i==1:
        #     i+=1

    if ifdrawfits:
        if fitsfile:
            drawfits(fits_file_path=fitsfile, fig=fig, vmin=vmin, vmax=vmax, drawalpha=drawalpha, iffilter=iffilter, cmap=cmap, cutl=cutl, cutu=cutu, filter=filter, alpha=alphaf)
        else:
            drawfits(fig=fig, vmin=vmin, vmax=vmax, drawalpha=drawalpha, iffilter=iffilter, cmap=cmap, cutl=cutl, cutu=cutu, filter=filter, alpha=alphaf)

    if legend:
        plt.legend()
    if save or savename:
        if savename==None:
            plt.savefig(f"../res/{region_name}/{Modelname}/???_sig_llh_model.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/???_sig_llh_model.pdf")
        else:
            plt.savefig(f"../res/{region_name}/{Modelname}/{savename}.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/{savename}.pdf")

    return fig

def gaussian(x,a,mu,sigma):
    return a*np.exp(-((x-mu)/sigma)**2/2)

def getsig1D(S, region_name, Modelname, name):
    bin_y,bin_x,patches=plt.hist(S.compressed(),bins=100)
    plt.close()
    bin_x=np.array(bin_x)
    bin_y=np.array(bin_y)
    fit_range = np.logical_and(bin_x>-5, bin_x<5)
    wdt=(bin_x[1]-bin_x[0])/2.
    try:
        popt, pcov = curve_fit(
            gaussian,
            bin_x[fit_range] + wdt,
            bin_y[fit_range[:-1]],
            bounds=([100, -2, 0], [50000000, 2, 10]),
        )
    except (ValueError, IndexError):
        popt, pcov = curve_fit(
            gaussian,
            bin_x[:100] + wdt,
            bin_y[:100],
            bounds=([100, -2, 0], [50000000, 2, 10]),
        )
    #popt,pcov = curve_fit(gaussian,bin_x[fit_range[0:-1]]+(bin_x[1]-bin_x[0])/2.,bin_y[fit_range[0:-1]],bounds=([100,-2,0],[50000000,2,10]))
    print("************************")
    print(popt)
    print("************************")
    print("max Significance= %.1f"%(max(S.compressed())))

    plt.figure()
    #plt.plot([0.,0.],[1,1e6],'k--',linewidth=0.5)
    plt.plot(
        (bin_x[:100] + bin_x[1:101]) / 2,
        gaussian((bin_x[:100] + bin_x[1:101]) / 2, popt[0], 0, 1),
        '--',
        label='expectation',
    )
    plt.plot((bin_x[:100] + bin_x[1:101]) / 2, bin_y, label="data")
    plt.plot(
        (bin_x[:100] + bin_x[1:101]) / 2,
        gaussian((bin_x[:100] + bin_x[1:101]) / 2, popt[0], popt[1], popt[2]),
        '--',
        label='fit',
    )
    plt.yscale('log')
    plt.xlim(-10,10)
    plt.ylim(1,max(bin_y*2))
    plt.grid(True)
    plt.text(-9.5,max(bin_y),'mean = %f\n width = %f'%(popt[1],popt[2]))
    plt.xlabel(r'Significance($\sigma$)')
    plt.ylabel("entries")
    plt.legend()
    plt.savefig(f"../res/{region_name}/{Modelname}/hist_sig_{name}.pdf")
    plt.savefig(f"../res/{region_name}/{Modelname}/hist_sig_{name}.png",dpi=300)

def getsigmap(region_name, Modelname, mymap,i=0,signif=17,res=False,name="J1908", alpha=None):
    """put in a smooth map and get a sig map.

        Args:
        Returns:
            sigmap: healpix
    """
    if len(mymap) == 1:
        i=0
        imap=mymap[0]
    else:
        imap=mymap[i]

    if res:
        scale=(imap[3]+imap[5])/(imap[6]+imap[8])
        ON=imap[3]*scale
        BK=imap[5]*scale
        name+="_res"
    else:
        scale=(imap[3]+imap[4])/(imap[6]+imap[7])
        ON=imap[3]*scale
        BK=imap[4]*scale

    if alpha is not None:
        alpha = alpha
    else:
        alpha = imap[10]

    if signif==5:
        S=(ON-BK)/np.sqrt(ON+alpha*BK)
    elif signif==9:
        S=(ON-BK)/np.sqrt(ON*alpha+BK)
    elif signif==17:
        S=np.sqrt(2.)*np.sqrt(ON*np.log((1.+alpha)/alpha*ON/(ON+BK/alpha))+BK/alpha*np.log((1.+alpha)*BK/alpha/(ON+BK/alpha)))
        S[ON<BK] *= -1
    else:
        S=(ON-BK)/np.sqrt(BK)
    getsig1D(S, region_name, Modelname, name)
    return S

def write_resmap(region_name, Modelname, WCDA, roi, maptree, ra1, dec1, outname,pta,exta, data_radius, binc="all", ifrunllh=True, detector="WCDA"):
    import os
    """write residual map to skymap root file.

        Args:
            pta=[1,0,1], exta=[0,0]: if you have 3 pt sources and 2 ext sources, and you only want to keep 1st and 3st sources,you do like this.
    """
    print(outname+"_res")
    # outname = "residual_all"

    # root setting
    ## infile
    forg = ROOT.TFile.Open(maptree,'read')
    bininfo = forg.Get("BinInfo")

    # Healpix setting
    colat = np.radians(90-dec1)
    lon = np.radians(ra1)
    vec = hp.ang2vec(colat,lon)
    holepixid = hp.query_disc(1024,vec,np.radians(data_radius))
    pixid=roi.active_pixels(1024)
    npix = hp.nside2npix(1024)

    ptid = len(pta)
    extid = len(exta)
    if binc=="all":
        binc = WCDA._active_planes

    # cut=""
    # kk=0
    # if detector=="WCDA":
    #     for i in range(6):
    #         if str(i) not in binc:
    #             if kk==0:
    #                 cut=cut+f"name!={binc}"
    #             else:
    #                 cut=cut+f"&&name!={binc}"
    #             kk+=1
    # elif detector=="KM2A":
    #     for i in range(14):
    #         if str(i) not in binc:
    #             if kk==0:
    #                 cut=cut+f"name!={binc}"
    #             else:
    #                 cut=cut+f"&&name!={binc}"
    #             kk+=1

    ## outfile
    fout = ROOT.TFile.Open(f"../res/{region_name}/{Modelname}/{outname}.root", 'recreate')
    bininfoout = bininfo.CloneTree()
    fout.Write(f"../res/{region_name}/{Modelname}/{outname}.root", ROOT.TFile.kOverwrite)
    fout.Close()


        
    for bin in binc:
        print('processing at nHit0',bin)
        ## outfile
        fout = ROOT.TFile.Open(f"../res/{region_name}/{Modelname}/{outname}.root", 'UPDATE')
        active_bin = WCDA._maptree._analysis_bins[bin]

        # model = WCDA._get_expectation(active_bin,bin,ptid,extid)
        model = WCDA._get_expectation(active_bin,bin,0,0)
        for i,pt in enumerate(pta):
            if not pt:
                model += WCDA._get_expectation(active_bin,bin,i+1,0)
                if i != 0:
                    model -= WCDA._get_expectation(active_bin,bin,i,0)
        
        for i,ext in enumerate(exta):
            if not ext:
                model += WCDA._get_expectation(active_bin,bin,0,i+1)
                if i != 0:
                    model -= WCDA._get_expectation(active_bin,bin,0,i)

        tdata=forg.Get("nHit%02d"%int(bin)).data
        tbkg=forg.Get("nHit%02d"%int(bin)).bkg

        n10=fout.mkdir('nHit%02d'%int(bin),'nHit%02d'%int(bin))

        dtype1 = [('count', float)]
        dtype2 = [('count', float)]

        toFill_d=np.zeros(npix, dtype = dtype1)
        toFill_m=np.zeros(npix, dtype = dtype2)

        tree1=ROOT.TTree('data','data')
        tree2=ROOT.TTree('bkg','bkg')

        tree1.SetDirectory(n10)
        tree1.SetEntries(npix)
        tree2.SetDirectory(n10)
        tree2.SetEntries(npix)

        #LOOP
        for idx in tqdm(holepixid):
            tdata.GetEntry(idx)
            tbkg.GetEntry(idx)
            toFill_d[idx]=tdata.count
            if idx in pixid:
                roiid=np.argwhere(pixid==idx)[0][0]
                toFill_m[idx]=tbkg.count+model[roiid]
            else:
                toFill_m[idx]=tbkg.count
        rn.array2tree(toFill_d,tree=tree1)
        rn.array2tree(toFill_m,tree=tree2)

        # obj11 = ROOT.TParameter(int)("Nside",1024)
        # obj21 = ROOT.TParameter(int)("Scheme",0)
        # obj12 = ROOT.TParameter(int)("Nside",1024)
        # obj22 = ROOT.TParameter(int)("Scheme",0)

        # tree1.GetUserInfo().Add(obj11)
        # tree1.GetUserInfo().Add(obj21)
        # tree2.GetUserInfo().Add(obj12)
        # tree2.GetUserInfo().Add(obj22)

        fout.Write()
        fout.Close()
    forg.Close()

    os.system(f'./tools/llh_skymap/Add_UserInfo ../res/{region_name}/{Modelname}/{outname}.root {binc[0]} {binc[-1]}')
    if ifrunllh:
        runllhskymap(roi, f"../res/{region_name}/{Modelname}/{outname}.root", ra1, dec1, data_radius, outname, detector=detector, ifres=1)
    return outname+"_res"

def getllhskymap(inname, region_name, ra1, dec1, data_radius, detector="WCDA", ifsave=True, ifdraw=False, drawfullsky=False, tofits=False):
    import glob
    import os
    folder_path = f"{libdir}/tools/llh_skymap/sourcetxt/{detector}_{inname}"
    if not os.path.exists(folder_path):
        pass
    name = folder_path.replace("./","")
    all_files = glob.glob(os.path.join(folder_path, '*'))
    nside=1024
    npix=hp.nside2npix(nside)
    skymap=hp.UNSEEN*np.ones(npix)
    for file in all_files:
        datas = np.load(file, allow_pickle=True)[0]
        for dd in datas:
            if dd != []:
                dd2 = np.array(dd)
                if len(dd2) >0:
                    skymap[dd2[:,0].astype(np.int)]=dd2[:,1]
    skymap=hp.ma(skymap)
    if ifsave:
        hp.write_map(f"../res/{region_name}/{detector}_{inname}.fits.gz", skymap, overwrite=True)
    if ifdraw:
        sources={}
        drawmap(region_name, "Modelname", sources, skymap, ra1, dec1, rad=2*data_radius, contours=[10000],save=0, 
                cat={ "LHAASO": [0, "P"],"TeVCat": [0, "s"],"PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"], "size": 20 ,"color": "grey"}, color="Fermi"
                  )
    if drawfullsky:
        fig = mt.hpDraw("region_name", "Modelname", skymap,0,0,skyrange=(0,360,-20,80),
                    colorlabel="Significance", contours=[1000], save=False, cat={}, color="Milagro", xsize=2048)
    if tofits:
        plt.figure()
        heal2fits(skymap, f"../res/{region_name}/{detector}_{inname}.fits", ra1-data_radius/np.cos(np.radians(dec1)), ra1+data_radius/np.cos(np.radians(dec1)), 0.01/np.cos(np.radians(dec1)), dec1-data_radius, dec1+data_radius, 0.01, ifplot=1, ifnorm=0)
    return skymap
from astropy.io import fits
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from Mycatalog import *
from Mysigmap import *
import MapPalette
from tqdm import tqdm

from Mycoord import *

def settransWCDA(WCDA, ra1, dec1, tansit=None):
    """
        修改天图tansit

        Parameters:

        Returns:
            >>> None
    """
    tobs=0
    if tansit is None:
        import ROOT as rt
        file = rt.TFile("../../data/20210305_20230731_bkgJ2000.root", "READ")
        hside = file.Get("hSide")
        tsecs = hside.GetNbinsX()
        wbinside = hside.GetXaxis().GetBinWidth(1)
        tside0 = hside.GetXaxis().GetBinLowEdge(1)
        for ii in range(tsecs):
            tside = tside0+(ii+0.5)*wbinside
            ha = tside-ra1
            zen, azi = eql2hcs(np.radians(ha), np.radians(dec1))
            if (zen>=0 and zen<=55):
                tobs += hside.GetBinContent(ii+1)
        # tansit = hside.GetBinContent(int(ra1/360*86164))/10
        tansit=tobs/864000
    log.info(f"Set WCDA tansit from: {WCDA._maptree._analysis_bins[str(0)]._n_transits} to {tansit}")
    for i in range(6):           
        WCDA._maptree._analysis_bins[str(i)]._n_transits=tansit

def Drawgascontour(file='../../data/J0248_co_-55--30_all.fits', levels=np.array([0.2,0.3,0.5,0.7,1,1.5,2,3,4])*1e22, vmin=0.2e22, vmax=1e22, oldmethod=0, cmap="Greys"):
    """
        叠加fitscontour

        Parameters:
            file: fits 文件
            levels: contour levels
            norm: 范围

        Returns:
            >>> None
    """ 
    if oldmethod:
        from matplotlib.colors import Normalize
        with fits.open(file) as hdul:
                # 输出文件信息
                qtsj = hdul[0].data
                hdul[0].header
        hdul[0].header
        # 获取坐标信息
        crval1 = hdul[0].header['CRVAL1']
        crval2 = hdul[0].header['CRVAL2']
        cdelt1 = hdul[0].header['CDELT1']
        cdelt2 = hdul[0].header['CDELT2']
        crpix1 = hdul[0].header['CRPIX1']
        crpix2 = hdul[0].header['CRPIX2']

        # CTYPE : 'GLON-AIT'  'GLAT-AIT'  
        # CRVAL : 0.0  0.0  
        # CRPIX : 8.5  -169.5  
        # PC1_1 PC1_2  : 1.0  0.0  
        # PC2_1 PC2_2  : 0.0  1.0  
        # CDELT : 0.090031633151485  0.090031633151485  
        # NAXIS : 1111  731,

        # 计算x轴和y轴的坐标范围
        naxis1, naxis2 = qtsj.shape
        xmin = (1 - crpix1) * cdelt1 + crval1
        xmax = (naxis1 - crpix1) * cdelt1 + crval1
        ymin = (1 - crpix2) * cdelt2 + crval2
        ymax = (naxis2 - crpix2) * cdelt2 + crval2
        glon = np.linspace(xmin,xmax,naxis1)
        glat = np.linspace(ymin,ymax,naxis2)
        Glon,Glat = np.meshgrid(glon,glat)
        galactic_coord = SkyCoord(Glon* u.degree, Glat* u.degree, frame='galactic')
        j2000_coords = galactic_coord.transform_to('fk5')
        Glon,Glat = j2000_coords.ra,j2000_coords.dec
        plt.contour(Glon,Glat,qtsj,5,cmap=cmap,levels=levels,norm=Normalize(vmin=vmin,vmax=vmax),alpha=0.7)
    else:
        from matplotlib.colors import Normalize
        from astropy.io import fits
        from astropy.wcs import WCS
        import numpy as np
        with fits.open(file) as hdul:
                # 输出文件信息
                qtsj = hdul[0].data
                wcs = WCS(hdul[0].header)

        naxis1, naxis2 = qtsj.shape
        # 构建像素坐标数组，例如：(x, y)
        x, y = np.meshgrid(np.arange(naxis2), np.arange(naxis1))

        # 使用wcs计算天球坐标
        ra, dec = wcs.all_pix2world(x, y, 0)

        galactic_coord = SkyCoord(ra* u.degree, dec* u.degree, frame='galactic')
        j2000_coords = galactic_coord.transform_to('fk5')
        Glon,Glat = j2000_coords.ra,j2000_coords.dec
        plt.contour(Glon,Glat,qtsj,5,cmap=cmap,levels=levels,norm=Normalize(vmin=vmin,vmax=vmax),alpha=0.7)

def Draw_reg(file="/data/home/cwy/Science/3MLWCDA/Standard/res/J0057/j0057_new_casob7.reg", iflabel=0):
    """
        画ds9的region文件

        Parameters:
            file: reg 文件
            iflabel: 是否标准label

        Returns:
            >>> None
    """ 
    import pyregion
    ax = plt.gca()
    r = pyregion.open(file)
    mark = r.get_mpl_patches_texts()[1]
    mark = [it for it in mark if str(type(it))=="<class 'matplotlib.lines.Line2D'>"]
    pt=0
    for item in r:
        name = item.name
        coord = item.coord_format
        drawpar = item.attr[1]
        if name=="circle":
            ra = item.params[0].v
            dec = item.params[1].v
            ra, dec = gal2edm(ra, dec)
            size = item.params[2].v
            print(ra, dec, size, drawpar["text"])
            label=None
            if iflabel:
                label=drawpar["text"]
            Draw_ellipse(ra, dec, size, 0, 0, drawpar["color"], alpha=1, linestyle="--", coord = "C", ax=ax, label=label)
            ax.annotate(drawpar["text"], (ra, dec+size+0.2), fontsize=10, color=drawpar["color"])
        elif name=="point":
            ra = item.params[0].v
            dec = item.params[1].v
            ra, dec = gal2edm(ra, dec)
            marker = mark[pt].get_marker()
            label=None
            if iflabel:
                label=drawpar["text"]
            ax.scatter(ra, dec, s=10, c=drawpar["color"], marker=marker)
            pt+=1
        else:
            print(name)
    plt.legend()

def interpimg(hp_map,xmin,xmax,ymin,ymax,xsize):
    """
        从healpix获取差值图片

        Parameters:
            xmin,xmax,ymin,ymax: 图片坐标范围
            xsize: 分辨率

        Returns:
            >>> img array
    """ 
    faspect = abs(xmax - xmin)/abs(ymax-ymin)
    phi   = np.linspace(xmin, xmax, xsize)
    theta = np.linspace(ymin, ymax, int(xsize/faspect))
    Phi, Theta = np.meshgrid(phi, theta)
    rotimg = hp.get_interp_val(hp_map, Phi,Theta,lonlat=True) #,nest=True
    # plt.contourf(Phi,Theta,rotimg)
    # plt.imshow(rotimg, origin="lower",extent=[xmin,xmax,ymin,ymax])
    # plt.colorbar()
    return rotimg

def Draw_diffuse(num = 9, levels=np.array([0.1, 1, 3, 5, 8, 10, 14, 16, 20])*1e-4, ifimg=False, ifGAL=False, iflog=False, ifcolorbar=False, ax=None, sigma=1):
    """
        画区域的银河diffuse模版 countour

        Parameters:

        Returns:
            >>> None
    """ 
    import ROOT
    import root_numpy as rt
    from matplotlib.colors import Normalize

    num=len(levels)

    root_file=ROOT.TFile.Open(("../../data/gll_dust.root"),"read")
    root_th2d=root_file.Get("gll_region")
    X_nbins=root_th2d.GetNbinsX()
    Y_nbins=root_th2d.GetNbinsY()
    X_min=root_th2d.GetXaxis().GetXmin()
    X_max=root_th2d.GetXaxis().GetXmax()
    Y_min=root_th2d.GetYaxis().GetXmin()
    Y_max=root_th2d.GetYaxis().GetXmax()
    X_size=(X_max-X_min)/X_nbins
    Y_size=(Y_max-Y_min)/Y_nbins
    # print(X_min,X_max,X_nbins, X_size)
    # print(Y_min,Y_max,Y_nbins, Y_size)
    data = rt.hist2array(root_th2d).T
    from scipy.ndimage import gaussian_filter
    data = gaussian_filter(data, sigma=sigma)
    if iflog:
        data=np.log(data)
        levels=np.log(levels)
    ra = np.linspace(X_min,X_max,X_nbins)
    dec = np.linspace(Y_min,Y_max,Y_nbins)
    RA, DEC = np.meshgrid(ra, dec)
    if not ifGAL:
        RA, DEC = gal2edm(RA, DEC)
    if ax is None:
        # plt.figure()
        ax = plt.gca()
    if ifimg:
        # plt.imshow(np.log(data),aspect="auto",extent=[X_min,X_max,Y_min,Y_max],origin='lower', alpha=0.7)
        ax.contourf(RA,DEC,data, alpha=0.3)
    lw = [0.1, 0.2, 0.5, 0.7, 1, 1.2, 1.4, 1.6, 1.8]
    ls = [':', ':', '-.', '-.', '--', '-', '-', '-', '-']
    ax.contour(RA,DEC,data,num,cmap="Greys",alpha=0.7, linestyles=ls[-num:],
                    linewidths=lw[-num:], levels=levels) #levels=np.array([0.2,0.3,0.5,0.7,1,1.5,2,3,4])*1e22,norm=Normalize(vmin=0.2e22,vmax=1e22)
    if ifcolorbar:
        ax.colorbar()


def smooth_array(arr):
    zero_indices = np.where(arr == 0)[0]
    for i in zero_indices:
        if i+1 < len(arr)-1:
            arr[i] = np.mean([arr[max(0, i-1)], arr[min(len(arr), i+1)]])
    return arr

def hpDraw(region_name, Modelname, map, ra, dec, coord = 'C', skyrange=None, rad=5, radx=5,rady=2.5,contours=[3,5],colorlabel="Excess",color="Fermi", plotres=False, save=False, cat={"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"], "size":20, "markercolor": "black",  "labelcolor": "black","angle": 60, "catext": 0}, ifDrawgascontour=False, Drawdiff=False, zmin=None, zmax=None, xsize = 2048, plotmol=False, savename="", grid=False, dpi=300, threshold=3):
    """Draw healpixmap.

        Args:
            cat: catalog to draw. such as {"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"]}, first in [1,"s"] is about if add a label?
                "o" is the marker you choose. 
                the catalog you can choose:  TeVCat/3FHL/4FGL/PSR/SNR/AGN/QSO/Simbad
        Returns:
            fig
    """
    map = smooth_array(map)
    if skyrange==None:
        ymax = dec+rady/2
        ymin = dec-rady/2
        xmin = ra-radx/2
        xmax = ra+radx/2
    else:
        xmin, xmax, ymin, ymax = skyrange
        # print(xmin, xmax, ymin, ymax)

    tfig   = plt.figure(num=2)
    rot = (0, 0, 0)
    
    # img = hp.cartview(hp_map,fig=2,lonra=[ra-rad,ra+rad],latra=[dec-rad,dec+rad],return_projected_map=True, rot=rot, coord=coord, xsize=xsize)
    img = interpimg(map, xmin,xmax,ymin,ymax,xsize)
    # img.fillna(1.0,inplace=True)
    img = np.nan_to_num(img)
    plt.close(tfig)

    faspect = abs(xmax - xmin)/abs(ymax-ymin)
    fysize = 4
    figsize = (fysize*faspect+2, fysize+2.75)
    dMin = -5
    dMax = 15
    dMin = np.min(img) if np.min(img) != None else -5
    dMax = np.max(img) if np.max(img) != None else 15
    if zmax !=None:
        dMax=zmax
    if zmin !=None:
        dMin=zmin
    if color == "Milagro":
        textcolor, colormap = MapPalette.setupMilagroColormap(dMin-1, dMax+1, threshold, 1000)
    elif color == "Fermi":
        textcolor, colormap = MapPalette.setupGammaColormap(10000)

    if plotmol:
        plt.figure(dpi=dpi)
        hp.mollview(map, cmap=colormap, min=dMin, max=dMax, title="LHAASO full sky", xsize=2048)
        hp.graticule()
        plt.savefig(f"fullskymol+{savename}.pdf", dpi=dpi)\
        
    fig = plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, origin="lower",extent=[xmin,xmax,ymin,ymax],vmin=dMin,vmax=dMax, cmap=colormap) #

    if grid:
        plt.grid(linestyle="--")
    cbar = plt.colorbar(format='%.2f',orientation="horizontal",shrink=0.6,
                            fraction=0.1,
                            #aspect=25,
                            pad=0.15)

    cbar.set_label(colorlabel)
    if np.max(img)<4:
        tiks = np.concatenate(([np.min(img)],[np.mean(img)],[np.max(img)]))
    elif np.max(img)<6:
        tiks = np.concatenate(([np.min(img)],[np.mean(img)],[3],[np.max(img)]))
    elif np.max(img)<20:
        tiks = np.concatenate(([np.min(img)],[np.mean(img)],[3],[5],[np.max(img)]))
    elif np.max(img)<30:
        tiks = np.concatenate(([np.min(img)],[5],[np.max(img)]))
    else:
        tiks = np.concatenate(([np.min(img)],[np.mean([np.min(img),np.max(img)])],[np.max(img)]))
    if zmax !=None:
        if tiks[tiks>=zmax] != []:
            tiks[-1]=zmax
        else:
            tiks=np.concatenate((tiks,[zmax]))
    if zmin !=None:
        if tiks[tiks<=zmin] != []:
            tiks[0]=zmin
        else:
            tiks= np.concatenate(([zmin],tiks))

        

    cbar.set_ticks(tiks)

    #,cbar.get_ticks()

    contp = plt.contour(img,levels=np.sort(contours),colors='g',linestyles = '-',linewidths = 2,origin='upper',extent=[xmin, xmax, ymax, ymin])
    fmt = {}
    strs=[]
    for i in range(len(contours)):
        strs.append('%d$\sigma$'%(contours[i]))
    for l, s in zip(contp.levels, strs):
        fmt[l] = s

    CLabel = plt.clabel(contp, contp.levels, use_clabeltext=True, rightside_up=True, inline=1, fmt=fmt, fontsize=10)

    for l in CLabel:
        l.set_rotation(180)


    plt.xlabel(r"$\alpha$ [$^\circ$]")
    plt.ylabel(r"$\delta$ [$^\circ$]")

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    if ifDrawgascontour:
        Drawgascontour()
    if Drawdiff:
        Draw_diffuse()
    

    plt.gca().set_aspect(1./np.cos((ymax+ymin)/2*np.pi/180))
    plt.gca().invert_xaxis()
    # plt.scatter(ra, dec, s=20**2,marker="+", facecolor="#000000", color="#000000")
    markerlist=["s","*","o","P","D","v","p","^"]
    if cat != {}:
        if "markercolor" not in cat.keys():
            cat["markercolor"]="black"
        if "labelcolor" not in cat.keys():
            cat["labelcolor"]="black"
        if "size" not in cat.keys():
            cat["size"]=20
        if "angle" not in cat.keys():
            cat["angle"]=-60
        if "catext" not in cat.keys():
            cat["catext"]=0
        for i,catname in enumerate(cat.keys()):
            if (catname != "size") and (catname != "markercolor") and (catname != "labelcolor") and (catname != "angle" and (catname != "catext")):
                Drawcat(xmin,xmax,ymin,ymax,catname,cat[catname][1], cat["markercolor"], cat["labelcolor"], angle=cat["angle"], label=catname, textlabel=cat[catname][0], size=cat["size"], drawext=cat["catext"])

    if save:
        if plotres:
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh_res.png",dpi=dpi)
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh_res.pdf")
        else:
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh.png",dpi=dpi)
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh.pdf")

    return fig

def maskdisk(map, ra1, dec1, radius):
    """
        mask healpix 圆形区域

        Parameters:

        Returns:
            >>> healpix
    """ 
    # 将源的坐标转换为HEALPix像素坐标
    nside=1024
    ipix = hp.ang2pix(nside, ra1, dec1, lonlat=True)

    # 使用query_disc来填充掩模，设置ipix为True
    maskid = hp.query_disc(nside, hp.pix2vec(nside, ipix), radius/180*np.pi)
    map[maskid]=hp.UNSEEN
    map = hp.ma(map)
    map[map==0]=hp.UNSEEN
    map = hp.ma(map)
    return(map)

def maskdiskout(map, ra1, dec1, radius):
    """
        mask healpix 圆形区域以外

        Parameters:

        Returns:
            >>> healpix
    """ 
    # 将源的坐标转换为HEALPix像素坐标
    nside=1024
    ipix = hp.ang2pix(nside, ra1, dec1, lonlat=True)

    # 使用query_disc来填充掩模，设置ipix为True
    pixIdx = list(np.arange(hp.nside2npix(nside)))
    maskid = list(hp.query_disc(nside, hp.pix2vec(nside, ipix), radius/180*np.pi))
    map[np.delete(pixIdx,maskid)]=hp.UNSEEN
    map = hp.ma(map)
    # map[map==0]=hp.UNSEEN
    # map = hp.ma(map)
    return(map)

def maskroi(map, roi):
    """
        mask 不规则 roi 区域以外

        Parameters:

        Returns:
            >>> healpix
    """ 
    nside=1024
    pixIdx = list(np.arange(hp.nside2npix(nside)))
    maskid = roi.active_pixels(nside)
    map[np.delete(pixIdx,maskid)]=hp.UNSEEN
    map=hp.ma(map)
    return(map)

def Draw_lateral_distribution(region_name, Modelname, map, ra, dec, num, width, ifdraw=False, ifsave=True):
    """ Draw_lateral_distribution.

        Args:
            num: num of bins
            width: width of one bin
        Returns:
            >>> np.array([psi, data_ring, errord,  bkg_ring, errorb, model_ring, errorm, excess_ring, res_ring])
    """
    colat_crab = np.radians(90-float(dec))
    lon_crab = np.radians(float(ra))
    vec_crab = hp.ang2vec(colat_crab,lon_crab)
    n = num#number of rings
    w = width#width of rings in degrees
    nside = 1024
    npix=hp.nside2npix(nside)
    pixel_areas = 4 * np.pi / npix
    pixel_areas = 129600 / npix

 
    data_disc = np.zeros(n) #define the excess_disc number in each disc
    data_ring = np.zeros(n) #define the excess_disc number in each ring
    bkg_disc = np.zeros(n) #define the excess_disc number in each disc
    bkg_ring = np.zeros(n) #define the excess_disc number in each ring
    model_disc = np.zeros(n) #define the excess_disc number in each disc
    model_ring = np.zeros(n) #define the excess_disc number in each ring
    excess_ring = np.zeros(n) #define the excess_disc number in each ring
    res_ring = np.zeros(n) #define the excess_disc number in each ring

    npx_disc = np.zeros(n)   #define the number of pixels in each disc
    npx_ring = np.zeros(n) #define the number of pixels in each ring

    disc = list(np.zeros(n))
    for i in tqdm(range(1,n+1), desc="get disc pixnum"):
        disc[i-1] = hp.query_disc(nside,vec_crab,np.radians(w*i))
        npx_disc[i-1] = disc[i-1].shape[0]
        
    npx_ring[0] = npx_disc[0]
    for i in tqdm(range(1,n), desc="get ring pixnum"):
        npx_ring[i] = npx_disc[i]-npx_disc[i-1]

    psi = np.arange(w/2,n*w,w) #horizontal coordinates

    data=map[0]
    bkg=map[1]
    model=map[2]

    for i in tqdm(range(n),desc="compute disk"):
        data_disc[i] = sum(data[disc[i]])
        bkg_disc[i] = sum(bkg[disc[i]])
        model_disc[i] = sum(model[disc[i]])

    data_ring[0] = data_disc[0]
    bkg_ring[0] = bkg_disc[0] 
    model_ring[0] = model_disc[0]
    errord = np.zeros(n) #poissonian error    
    errord[0] = np.sqrt(sum(data[disc[0]]))
    errorb = np.zeros(n) #poissonian error    
    errorb[0] = np.sqrt(sum(bkg[disc[0]]))
    errorm = np.zeros(n) #poissonian error    
    errorm[0] = np.sqrt(sum(model[disc[0]]))
    for i in tqdm(range(1,n),desc="compute ring"):
        data_ring[i] = data_disc[i]-data_disc[i-1]
        errord[i] = np.sqrt(data_ring[i])
        bkg_ring[i] = bkg_disc[i]-bkg_disc[i-1]
        errorb[i] = np.sqrt(bkg_ring[i])
        model_ring[i] = model_disc[i]-model_disc[i-1]
        errorm[i] = np.sqrt(model_ring[i])
    data_ring/=npx_ring
    bkg_ring/=npx_ring
    model_ring/=npx_ring
    excess_ring = data_ring-bkg_ring
    res_ring = data_ring-model_ring
    errord/=npx_ring
    errorb/=npx_ring
    errorm/=npx_ring

    psfdata = np.array([psi, data_ring, errord,  bkg_ring, errorb, model_ring, errorm, excess_ring, res_ring])

    if ifdraw:
        fig1 = plt.figure()
        plt.errorbar(psfdata[0],psfdata[1],psfdata[2],fmt='o', label="data", c="tab:blue")
        plt.errorbar(psfdata[0],psfdata[5],psfdata[6],fmt='o',label="model", c="tab:red")
        plt.errorbar(psfdata[0],psfdata[3],psfdata[4],fmt='o',label="bkg", c="black")
        plt.xlabel(r"$\phi^{\circ}$")
        plt.ylabel(r"$\frac{excess}{N_{pix}}$")
        plt.legend()
        if ifsave:
            plt.savefig(f"../res/{region_name}/{Modelname}/all_profile_{region_name}.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/all_profile_{region_name}.pdf")
        fig2 = plt.figure()
        plt.errorbar(psfdata[0],psfdata[7],psfdata[2],fmt='o',label="excess", c="black")
        plt.errorbar(psfdata[0],psfdata[8],psfdata[2],fmt='o',label="residual", c="tab:red")
        plt.xlabel(r"$\phi^{\circ}$")
        plt.ylabel(r"$\frac{excess}{N_{pix}}$")
        plt.legend()
        if ifsave:
            plt.savefig(f"../res/{region_name}/{Modelname}/eandr_profile_{region_name}.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/eandr_profile_{region_name}.pdf")
    return psfdata, pixel_areas

def Draw_lateral_distribution_deg2(region_name, Modelname, map, ra, dec, num, width, ifdraw=False, ifsave=True):
    """ Draw_lateral_distribution.

        Args:
            num: num of bins
            width: width of one bin
        Returns:
            >>> np.array([psi, data_ring, errord,  bkg_ring, errorb, model_ring, errorm, excess_ring, res_ring])
    """
    colat_crab = np.radians(90-float(dec))
    lon_crab = np.radians(float(ra))
    vec_crab = hp.ang2vec(colat_crab,lon_crab)
    n = num#number of rings
    w = width#width of rings in degrees
    nside = 1024
    npix=hp.nside2npix(nside)
    pixel_areas = 4 * np.pi / npix
    pixel_areas = 129600 / npix

 
    data_disc = np.zeros(n) #define the excess_disc number in each disc
    data_ring = np.zeros(n) #define the excess_disc number in each ring
    bkg_disc = np.zeros(n) #define the excess_disc number in each disc
    bkg_ring = np.zeros(n) #define the excess_disc number in each ring
    model_disc = np.zeros(n) #define the excess_disc number in each disc
    model_ring = np.zeros(n) #define the excess_disc number in each ring
    excess_ring = np.zeros(n) #define the excess_disc number in each ring
    res_ring = np.zeros(n) #define the excess_disc number in each ring

    npx_disc = np.zeros(n)   #define the number of pixels in each disc
    npx_ring = np.zeros(n) #define the number of pixels in each ring

    disc = list(np.zeros(n))
    for i in tqdm(range(1,n+1), desc="get disc pixnum"):
        disc[i-1] = hp.query_disc(nside,vec_crab,np.radians(np.sqrt(w*i)))
        npx_disc[i-1] = disc[i-1].shape[0]

    npx_ring[0] = npx_disc[0]
    for i in tqdm(range(1,n), desc="get ring pixnum"):
        npx_ring[i] = npx_disc[i]-npx_disc[i-1]

    psi = np.arange(w/2,n*w,w) #horizontal coordinates

    data=map[0]
    bkg=map[1]
    model=map[2]

    for i in tqdm(range(n),desc="compute disk"):
        data_disc[i] = sum(data[disc[i]])
        bkg_disc[i] = sum(bkg[disc[i]])
        model_disc[i] = sum(model[disc[i]])

    data_ring[0] = data_disc[0]
    bkg_ring[0] = bkg_disc[0] 
    model_ring[0] = model_disc[0]
    errord = np.zeros(n) #poissonian error    
    errord[0] = np.sqrt(sum(data[disc[0]]))
    errorb = np.zeros(n) #poissonian error    
    errorb[0] = np.sqrt(sum(bkg[disc[0]]))
    errorm = np.zeros(n) #poissonian error    
    errorm[0] = np.sqrt(sum(model[disc[0]]))
    for i in tqdm(range(1,n),desc="compute ring"):
        data_ring[i] = data_disc[i]-data_disc[i-1]
        errord[i] = np.sqrt(data_ring[i])
        bkg_ring[i] = bkg_disc[i]-bkg_disc[i-1]
        errorb[i] = np.sqrt(bkg_ring[i])
        model_ring[i] = model_disc[i]-model_disc[i-1]
        errorm[i] = np.sqrt(model_ring[i])
    data_ring/=npx_ring
    bkg_ring/=npx_ring
    model_ring/=npx_ring
    excess_ring = data_ring-bkg_ring
    res_ring = data_ring-model_ring
    errord/=npx_ring
    errorb/=npx_ring
    errorm/=npx_ring

    psfdata = np.array([psi, data_ring, errord,  bkg_ring, errorb, model_ring, errorm, excess_ring, res_ring])

    if ifdraw:
        fig1 = plt.figure()
        plt.errorbar(psfdata[0],psfdata[1],psfdata[2],fmt='o', label="data", c="tab:blue")
        plt.errorbar(psfdata[0],psfdata[5],psfdata[6],fmt='o',label="model", c="tab:red")
        plt.errorbar(psfdata[0],psfdata[3],psfdata[4],fmt='o',label="bkg", c="black")
        plt.xlabel(r"$\phi^{\circ}$")
        plt.ylabel(r"$\frac{excess}{N_{pix}}$")
        plt.legend()
        if ifsave:
            plt.savefig(f"../res/{region_name}/{Modelname}/all_profile_{region_name}.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/all_profile_{region_name}.pdf")
        fig2 = plt.figure()
        plt.errorbar(psfdata[0],psfdata[7],psfdata[2],fmt='o',label="excess", c="black")
        plt.errorbar(psfdata[0],psfdata[8],psfdata[2],fmt='o',label="residual", c="tab:red")
        plt.xlabel(r"$\phi^{\circ}$")
        plt.ylabel(r"$\frac{excess}{N_{pix}}$")
        plt.legend()
        if ifsave:
            plt.savefig(f"../res/{region_name}/{Modelname}/eandr_profile_{region_name}.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/eandr_profile_{region_name}.pdf")
    return psfdata, pixel_areas

def getmaskedroi(ra1, dec1, data_radius, maskp=[]):
    """
        获取不规则roi

        Parameters:
            maskp: 如: [(85.78, 23.40, 2), (83.63, 22.02, 3)] 及需要mask得中心以及范围序列

        Returns:
            >>> roi
    """ 
    nside=2**10
    numpix = hp.nside2npix(nside)
    roimap = np.ones(numpix, dtype=np.float64)
    pixIdx = np.arange(numpix)
    new_lats = hp.pix2ang(nside, pixIdx)[0] # thetas I need to populate with interpolated theta values
    new_lons = hp.pix2ang(nside, pixIdx)[1] # phis, same
    mask = ((-new_lats + np.pi/2 < -20./180*np.pi) | (-new_lats + np.pi/2 > 80./180*np.pi) | (distance(new_lons/np.pi*180, (np.pi/2-new_lats)/np.pi*180, ra1, dec1)>data_radius))
    for i in range(len(maskp)):
        maskra, maskdec, maskr = maskp[i]
        maskdisk = (distance(new_lons/np.pi*180, (np.pi/2-new_lats)/np.pi*180, maskra, maskdec)<maskr)
        mask = mask | maskdisk
    roimap[mask]=0
    return roimap


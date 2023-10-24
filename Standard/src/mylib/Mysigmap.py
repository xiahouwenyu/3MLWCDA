from tqdm import tqdm
import healpy as hp
import numpy as np

import matplotlib, sys
# sys.path.append(__file__[:-12])
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import Mymap as mt

import ROOT

import copy

import root_numpy as rn

from scipy.optimize import curve_fit

def getmap(WCDA, roi, name="J0248", signif=17, smoothsigma = [0.42, 0.32, 0.25, 0.22, 0.18, 0.15], 
           save = False, 
           binc="all",
           stack=[],
           modelindex=None,
           pta=[], exta=[],
           smooth=False
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
        binc=WCDA._maptree._analysis_bins

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
        for i, weight in enumerate(stack):
            for j in range(10):
                summap[i][j] *= weight
                if j in [6, 7, 8]:
                    summap[i][j] *= weight
        outmap = [np.ma.sum([bin[i] for bin in summap],axis=0) for i in tqdm(range(11))]
        # outmap[-1] = np.ma.sqrt(np.ma.sum([bin[-1]**2*stack[i] for i,bin in enumerate(amap) if i<6],axis=0))
        smooth_sigma=smoothsigma[int(list(WCDA._maptree._analysis_bins.keys())[-1])+1]
        for i,pix in enumerate(tqdm(pixid)):
            alpha[pix]=2*smooth_sigma*1.51/60./np.sin(theta)
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

def drawmap(region_name, Modelname, sources, map, ra1, dec1, rad=6, contours=[3, 5], save=False, savename=None, cat={"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[0,"o"]}):  # sourcery skip: extract-duplicate-method
    """Draw a healpix map with fitting results.

        Args:
            sources: use function get_sources() to get the fitting results.
            cat: catalog to draw. such as {"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"]}, first in [1,"s"] is about if add a label?
                "o" is the marker you choose.
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    from matplotlib.patches import Ellipse
    fig = mt.hpDraw(region_name, Modelname, map,ra1,dec1,
            radx=rad/np.cos(dec1/180*np.pi),rady=rad,
            colorlabel="Significance", contours=contours, save=False, cat=cat
            )
    ax = plt.gca()
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
            elif par in ["sigma","rdiff0","radius"]:
                sigma = source[par][2]
                sigmau = source[par][3]
                sigmal = source[par][4]
        if sources[sc]['type'] == 'extended source':
            plt.errorbar(x, y, yerr=(np.abs([yel]), [yeu]), xerr=(np.abs([xel]), [xeu]), fmt='o',markersize=2,capsize=1,elinewidth=1,color="tab:green", label=sc)
            error_ellipse = Ellipse((x, y), width=sigma/np.cos(np.radians(y)), height=sigma, edgecolor='tab:green', fill=False,linestyle="-")
            ax.add_artist(error_ellipse)
            error_ellipse = Ellipse((x, y), width=(sigma+sigmau)/np.cos(np.radians(y)), height=sigma+sigmau, edgecolor='tab:green', fill=False,linestyle="--", alpha=0.5)
            ax.add_artist(error_ellipse)
            error_ellipse = Ellipse((x, y), width=(sigma-abs(+sigmal))/np.cos(np.radians(y)), height=sigma-abs(sigmal), edgecolor='tab:green', fill=False,linestyle="--", alpha=0.5)
            ax.add_artist(error_ellipse)
        else:
            plt.errorbar(x, y, yerr=(np.abs([yel]), [yeu]), xerr=(np.abs([xel]), [xeu]), fmt='o',markersize=2,capsize=1,elinewidth=1,color="tab:green",label=sc)
    plt.legend()
    if save or savename:
        if savename==None:
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh_model.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh_model.pdf")
        else:
            plt.savefig(f"../res/{region_name}/{Modelname}/{savename}.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/{savename}.pdf")

    return fig

def gaussian(x,a,mu,sigma):
    return a*np.exp(-((x-mu)/sigma)**2/2)

def getsigmap(region_name, Modelname, mymap,i=0,signif=17,res=False,name="J1908"):
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
    return S

def write_resmap(region_name, Modelname, WCDA, roi, maptree, ra1, dec1, outname,pta,exta, binc="all"):
    """write residual map to skymap root file.

        Args:
            pta=[1,0,1], exta=[0,0]: if you have 3 pt sources and 2 ext sources, and you only want to keep 1st and 3st sources,you do like this.
    """
    
    # outname = "residual_all"

    # root setting
    ## infile
    forg = ROOT.TFile.Open(maptree,'read')
    bininfo = forg.Get("BinInfo")

    # Healpix setting
    colat = np.radians(90-dec1)
    lon = np.radians(ra1)
    vec = hp.ang2vec(colat,lon)
    holepixid = hp.query_disc(1024,vec,np.radians(10))
    pixid=roi.active_pixels(1024)
    npix = hp.nside2npix(1024)

    ## outfile
    fout = ROOT.TFile.Open(f"../res/{region_name}/{Modelname}/{outname}.root", 'recreate')
    bininfoout = bininfo.CloneTree()
    fout.Write(f"../res/{region_name}/{Modelname}/{outname}.root", ROOT.TFile.kOverwrite)
    fout.Close()

    ptid = len(pta)
    extid = len(exta)
    if binc=="all":
        binc = WCDA._maptree._analysis_bins

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
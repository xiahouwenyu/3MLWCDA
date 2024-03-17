from threeML import *
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
import copy
from Myfit import *

import ROOT

import numpy as np

import ctypes

import scipy as sp

import matplotlib.pyplot as plt

from Mylightcurve import p2sigma

from Myspec import *

from Mymap import *

def get_upperlimit(jl, par="J0057.spectrum.main.PowerlawM.K", num=200, plot=True, CL=0.95):
    """
        获取参数llh扫描的上限

        Parameters:
        
        Returns:
            >>> 上限, 新minimum
    """ 
    jl.restore_best_fit()
    (
    current_value,
    current_delta,
    current_min,
    current_max,
    ) = jl.minimizer._internal_parameters[par]
    # orgllh = [current_value, jl.minus_log_like_profile(current_value)]
    if current_value<0:
        current_value=1e-24
    trials = np.logspace(-30, np.log10(current_value)+2, num)
    deltaTS=[]
    for trial in trials:
        deltaTS.append(2*(jl.minus_log_like_profile(trial)-jl.minus_log_like_profile(trials[0])))
    deltaTS = np.array(deltaTS)
    TSneed = p2sigma(1-(2*CL-1))**2
    indices = np.where(deltaTS == min(deltaTS))[0]
    newmini = trials[indices]
    if np.any(current_value!=newmini):
        log.info("Find new minimun!!")
        if not isinstance(newmini, np.ndarray):
            current_value=newmini
            deltaTS = deltaTS-deltaTS[indices]
        else:
            current_value=newmini.max()
            newmini=newmini.max()
            deltaTS = deltaTS-deltaTS[indices[0]]
    else:
        newmini = None


    try:
        upper = trials[(deltaTS>=TSneed) & (trials>=current_value)][0]
        sigma1 = trials[deltaTS>=1 & (trials>=current_value)][0]
        sigma2 = trials[deltaTS>=4 & (trials>=current_value)][0]
        sigma3 = trials[deltaTS>=9 & (trials>=current_value)][0]
    except:
        upper=0; sigma1=0; sigma2=0; sigma3=0
    if plot:
        plt.figure()
        try:
            plt.scatter(current_value, trials[indices], marker="*", c="tab:blue", zorder=4, s=100)
        except:
            plt.scatter(current_value, trials[indices[0]], marker="*", c="tab:blue", zorder=4, s=100)
        plt.plot(trials,deltaTS)
        plt.axhline(TSneed,color="black", linestyle="--", label=f"95% upperlimit: {upper:.2e}")
        plt.axvline(upper,color="black", linestyle="--")
        plt.axhline(1,color="tab:green", linestyle="--", label=f"1 sigma: {sigma1:.2e}")
        plt.axhline(4,color="tab:orange", linestyle="--", label=f"2 sigma: {sigma2:.2e}")
        plt.axhline(9,color="tab:red", linestyle="--", label=f"3 sigma: {sigma3:.2e}")
        plt.legend()
        plt.ylabel(r"$\Delta TS$")
        plt.xlabel(par)
        plt.xscale("log")
        plt.show()
    return upper, newmini

def cal_K_WCDA(i,lm,maptree,response,roi,source="J0248", ifgeterror=False, mini="ROOT", ifpowerlawM=False, CL=0.95, nCL=False, threshold=2, iffixtans=False, bondaryrange=100):
    #Only fit the spectrum.K for plotting  points on the spectra
        #prarm1: fixed.spectrum.alpha 
        #param2: fixed.spectrum.belta
        #return: spectrum.K
    # Instance the plugin
    WCDA_1 = HAL("WCDA_1", maptree, response, roi, flat_sky_pixels_size=0.05)
    WCDA_1.psf_integration_method="exact"
    if iffixtans:
        settransWCDA(WCDA_1, roi.ra_dec_center[0], roi.ra_dec_center[1])
    lm2 = copy.deepcopy(lm)
    # fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    fluxUnit = 1e-9
    # Define model
    sources = lm2.sources.keys()
    if ifpowerlawM:
        #遍历源
        oldscs = copy.deepcopy(lm2.sources)
        for ss in sources:
            par = copy.deepcopy(lm2.sources[ss].parameters)
            #PoerlawM() 替换
            if ss==source:
                lm2.remove_source(ss)
                if str(oldscs[ss].source_type) == "point source":
                    lm2.add_source(PointSource(ss, 0, 0, spectral_shape=PowerlawM()))
                elif str(oldscs[ss].source_type) == "extended source":
                    lm2.add_source(ExtendedSource(ss, spatial_shape=oldscs[ss].spatial_shape, spectral_shape=PowerlawM()))
            #遍历参数
            for pa in par.keys():
                newpa=pa
                if ss==source:
                    newpa = pa.replace("Powerlaw","PowerlawM")
                if (".K" not in pa):
                    lm2.sources[ss].parameters[newpa].value = par[pa].value
                    lm2.sources[ss].parameters[newpa].fix = True
                elif (ss != source and ("SpatialTemplate" not in pa)):
                    lm2.sources[ss].parameters[newpa].value = par[pa].value
                    lm2.sources[ss].parameters[newpa].fix = True
                elif ("SpatialTemplate" not in pa):
                    kparname=newpa
                    # print("change bounds!!!!")
                    lm2.sources[ss].parameters[newpa].bounds=np.array((-bondaryrange*par[pa].value*1e9,bondaryrange*par[pa].value*1e9))*fluxUnit
                    lm2.sources[ss].parameters[newpa].delta = 0.1*bondaryrange*par[pa].value
    else:
        #遍历源
        for ss in sources:
            freep = lm2.sources[ss].free_parameters.keys()
            #遍历参数
            for fp in freep:
                if (ss == source and ".K" not in fp) or ss != source:
                    lm2.sources[ss].free_parameters[fp].fix = True
                elif (ss == source and ".K" in fp and lm.sources[ss].components['main'].shape.name=="PowerlawM"):
                    lm2.sources[ss].free_parameters[fp].bounds=np.array((-bondaryrange*lm2.sources[ss].free_parameters[fp].value*1e9,bondaryrange*lm2.sources[ss].free_parameters[fp].value*1e9)) * fluxUnit
                    lm2.sources[ss].free_parameters[fp].delta = 0.1*bondaryrange*lm2.sources[ss].free_parameters[fp].value
                else:
                    kparname=fp

    result2 = fit("nothing","nothing", WCDA_1,lm2,int(i),int(i),mini=mini,savefit=False, ifgeterror=ifgeterror)

    TSflux=result2[0].compute_TS(source,result2[1][1]).values[0][2]
    if np.isnan(TSflux) or TSflux>1e10:
        TSflux=0
    if not nCL:
        if ifpowerlawM:
            lb, ub = result2[0].results.get_equal_tailed_interval(lm2.sources[source].parameters[kparname], cl=2*CL-1)    
            if int(i) >= 10 and TSflux<threshold**2:
                ub, mewmini = get_upperlimit(result2[0], kparname, CL=CL)
                if mewmini is not None:
                    result2[1][0].iloc[0,0] = mewmini
            if (ub-result2[1][0].iloc[0,0])>0 and TSflux<threshold**2:
                result2[1][0].iloc[0,3] = (ub-result2[1][0].iloc[0,0])/1.96
                result2[1][0].iloc[0,2] = (ub-result2[1][0].iloc[0,0])/1.96
                result2[1][0].iloc[0,1] = 0.9*result2[1][0].iloc[0,2]
            # elif TSflux<4:
            #     log.warning("upper limit is lower than value, just use 0 as the value!")
            #     result2[1][0].iloc[0,3] = (ub)/1.96
            #     result2[1][0].iloc[0,2] = (ub)/1.96
            #     result2[1][0].iloc[0,1] = 0.9*result2[1][0].iloc[0,2]
            if result2[1][0].loc[kparname,"value"]<0:
                TSflux=-TSflux
        else:
            lb, ub = result2[0].results.get_equal_tailed_interval(lm2.sources[source].parameters[kparname.replace("PowerlawM","Powerlaw")], cl=2*CL-1)
            if int(i) >= 10  and TSflux<threshold**2:
                ub, mewmini = get_upperlimit(result2[0], kparname.replace("PowerlawM","Powerlaw"), CL=CL)
                if mewmini is not None:
                    result2[1][0].iloc[0,0] = mewmini
            if (ub-result2[1][0].iloc[0,0])>0 and TSflux<threshold**2:
                result2[1][0].iloc[0,3] = (ub-result2[1][0].iloc[0,0])/1.96
                result2[1][0].iloc[0,2] = (ub-result2[1][0].iloc[0,0])/1.96
                result2[1][0].iloc[0,1] = 0.9*result2[1][0].iloc[0,2]
            # elif TSflux<4:
            #     log.warning("upper limit is lower than value, just use 0 as the value!")
            #     result2[1][0].iloc[0,3] = (ub)/1.96
            #     result2[1][0].iloc[0,2] = (ub)/1.96
            #     result2[1][0].iloc[0,1] = 0.9*result2[1][0].iloc[0,2]

    return result2, TSflux

def reweightx(lm,WCDA,i,func = fun_Logparabola,source="J0248"):
    """
        获取拟合能谱下每个bin的能量以及其误差

        Parameters:
        
        Returns:
            >>> x,x_lo,x_hi
    """ 
    piv=3
    par = lm.sources[source].parameters.keys()
    for pp in par:
        if ".K" in pp:
            K = lm.sources[source].parameters[pp].value
        if ".index" in pp:
            func = fun_Powerlaw
            index = lm.sources[source].parameters[pp].value
        if ".alpha" in pp:
            func = fun_Logparabola
            alpha = lm.sources[source].parameters[pp].value
        if ".beta" in pp:
            func = fun_Logparabola
            beta = lm.sources[source].parameters[pp].value
        if ("lat0"in pp) or ("dec" in pp):
            dec = lm.sources[source].parameters[pp].value
    try:
        resp = WCDA._response.get_response_dec_bin(dec) #resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    except:
        resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    sbl=resp[i]
    binlow = sbl.sim_energy_bin_low[0]
    nbins = len(sbl.sim_signal_events_per_bin)
    binhigh = sbl.sim_energy_bin_hi[nbins-1]
    th1=ROOT.TH1D("","",nbins,np.log10(binlow),np.log10(binhigh))
    for j in range(nbins):
        signal = sbl.sim_signal_events_per_bin[j]
        binl = sbl.sim_energy_bin_low[j]
        binu = sbl.sim_energy_bin_hi[j]
        simflux = sbl.sim_differential_photon_fluxes[j]*(binu-binl)
        if func == fun_Logparabola:
            fitflux = sp.integrate.quad(func,binl,binu,args=(K,alpha,beta,piv))[0]
        elif func == fun_Powerlaw:
            fitflux = sp.integrate.quad(func,binl,binu,args=(K,index,piv))[0]
        _flux = signal*fitflux/simflux
        th1.SetBinContent(j+1,_flux)
    x = ctypes.c_double(1.)
    x_lo = ctypes.c_double(1.)
    x_hi = ctypes.c_double(1.)

    y = ctypes.c_double(0.5)
    y_lo = ctypes.c_double(0.17)
    y_hi = ctypes.c_double(0.84)
    th1.GetQuantiles(1,x,y)
    th1.GetQuantiles(1,x_lo,y_lo)
    th1.GetQuantiles(1,x_hi,y_hi)
    return x,x_lo,x_hi

def getexposure(lm,WCDA,i,func = fun_Logparabola,source="J0248"):
    """
        获取每个bin的counts到流强转化比

        Parameters:
        
        Returns:
            >>> ratio
    """ 
    piv=3
    par = lm.sources[source].parameters.keys()
    for pp in par:
        if ".K" in pp:
            K = lm.sources[source].parameters[pp].value
        if ".index" in pp:
            func = fun_Powerlaw
            index = lm.sources[source].parameters[pp].value
        if ".alpha" in pp:
            func = fun_Logparabola
            alpha = lm.sources[source].parameters[pp].value
        if ".beta" in pp:
            func = fun_Logparabola
            beta = lm.sources[source].parameters[pp].value
        if ("lat0"in pp) or ("dec" in pp):
            dec = lm.sources[source].parameters[pp].value
    try:
        resp = WCDA._response.get_response_dec_bin(dec) #resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    except:
        resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    sbl=resp[str(i)]
    binlow = sbl.sim_energy_bin_low[0]
    nbins = len(sbl.sim_signal_events_per_bin)
    binhigh = sbl.sim_energy_bin_hi[nbins-1]
    th1=ROOT.TH1D("","",nbins,np.log10(binlow),np.log10(binhigh))
    for j in range(nbins):
        signal = sbl.sim_signal_events_per_bin[j]
        binl = sbl.sim_energy_bin_low[j]
        binu = sbl.sim_energy_bin_hi[j]
        simflux = sbl.sim_differential_photon_fluxes[j]*(binu-binl)
        if func == fun_Logparabola:
            fitflux = sp.integrate.quad(func,binl,binu,args=(K*1e9,alpha,beta,piv))[0]
        elif func == fun_Powerlaw:
            fitflux = sp.integrate.quad(func,binl,binu,args=(K*1e9,index,piv))[0]
        _flux = signal*fitflux/simflux
        th1.SetBinContent(j+1,_flux)
    def xfunc(x,K,index,piv):
        return x*func(x, K,index,piv)
    # return th1.GetSum()
    return th1.Integral(1,nbins-1)/sp.integrate.quad(xfunc,0.1,20,args=(K*1e9,index,piv))[0]

def reweightxall(WCDA, lm, func = fun_Logparabola,source="J0248"):
    """
        获取整个探测器在拟合能谱下的能量及其误差范围

        Parameters:
        
        Returns:
            >>> x,x_lo,x_hi, th1
    """ 
    piv=3
    par = lm.sources[source].parameters.keys()
    for pp in par:
        if ".K" in pp:
            K = lm.sources[source].parameters[pp].value
        if ".index" in pp:
            func = fun_Powerlaw
            index = lm.sources[source].parameters[pp].value
        if ".alpha" in pp:
            func = fun_Logparabola
            alpha = lm.sources[source].parameters[pp].value
        if ".beta" in pp:
            func = fun_Logparabola
            beta = lm.sources[source].parameters[pp].value
        if ("lat0"in pp) or ("dec" in pp):
            dec = lm.sources[source].parameters[pp].value
    try:
        resp = WCDA._response.get_response_dec_bin(dec) #resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    except:
        resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    sbl=resp["0"]
    binlow = sbl.sim_energy_bin_low[0]
    nbins = len(sbl.sim_signal_events_per_bin)
    binhigh = sbl.sim_energy_bin_hi[nbins-1]
    th1=ROOT.TH1D("","",nbins,np.log10(binlow),np.log10(binhigh))
    for i in WCDA._active_planes:
        sbl=resp[i]
        for j in range(nbins):
            signal = sbl.sim_signal_events_per_bin[j]
            binl = sbl.sim_energy_bin_low[j]
            binc = np.log10(sbl.sim_energy_bin_centers[j])
            binu = sbl.sim_energy_bin_hi[j]
            simflux = sbl.sim_differential_photon_fluxes[j]*(binu-binl)
            if func == fun_Logparabola:
                fitflux = sp.integrate.quad(func,binl,binu,args=(K,alpha,beta,piv))[0]
            elif func == fun_Powerlaw:
                fitflux = sp.integrate.quad(func,binl,binu,args=(K,index,piv))[0]
            _flux = signal*fitflux/simflux
            th1.Fill(binc,_flux)
    x = ctypes.c_double(1.)
    x_lo = ctypes.c_double(1.)
    x_hi = ctypes.c_double(1.)

    y = ctypes.c_double(0.5)
    y_lo = ctypes.c_double(0.17)
    y_hi = ctypes.c_double(0.84)
    th1.GetQuantiles(1,x,y)
    th1.GetQuantiles(1,x_lo,y_lo)
    th1.GetQuantiles(1,x_hi,y_hi)
    return x,x_lo,x_hi, th1

def getdatapoint(Detector, lm, maptree,response,roi, source="J0248", ifgeterror=False, mini="ROOT", ifpowerlawM=False, CL=0.95, piv = 3, nCL=False, threshold=2, iffixtans=False):
    """
        获取某个源的能谱点

        Parameters:
            source: 想获取的源名称
            ifgeterror: 是否获取精确的误差, 似乎有些毛病!
            ifpowerlawM: 是否允许负的拟合范围, 防止卡边界,建议允许
            CL: 上限点取%多少上限?
            piv: 固定的piv energy, 需和之前相同
            nCL: 不使用扫描的上限
            threshold: 上限处理的显著性阈值, 默认2sigma
        
        Returns:
            能谱点举证以及每个bin拟合结果列表: list([jl, results])
            >>> Flux_WCDA, results
    """ 
    Flux_WCDA=np.zeros((len(Detector._active_planes),8), dtype=np.double())
    # piv = result[1][0].values[3][0]/1e9
    par = lm.sources[source].parameters.keys()
    for pp in par:
        if ".K" in pp:
            K = lm.sources[source].parameters[pp].value
        if ".index" in pp:
            func = fun_Powerlaw
            index = lm.sources[source].parameters[pp].value
        if ".alpha" in pp:
            func = fun_Logparabola
            alpha = lm.sources[source].parameters[pp].value
        if ".beta" in pp:
            func = fun_Logparabola
            beta = lm.sources[source].parameters[pp].value
    imin=100
    results = []
    silence_logs()
    for i in tqdm(Detector._active_planes):
        if int(i) <= imin:
            imin = int(i)
        xx = reweightx(lm,Detector, i,source=source,func=func)
        result2, TSflux=cal_K_WCDA(i,lm, maptree,response,roi, source=source, ifgeterror=ifgeterror, mini=mini, ifpowerlawM=ifpowerlawM, CL=CL, nCL=nCL, threshold=threshold, iffixtans=iffixtans)
        # try:
        #     result2, TSflux=cal_K_WCDA(i,lm, maptree,response,roi, source=source, ifgeterror=ifgeterror, mini=mini, ifpowerlawM=ifpowerlawM, CL=CL, nCL=nCL, threshold=threshold)
        # except Exception as e:
        #     log.info(f"{e}")
        #     log.info(f"Point {i} failed to fit, skip it!")
        #     continue
        results.append(result2)
        flux1 = result2[1][0].values[0][0]
        errorl = abs(result2[1][0].values[0][1])
        erroru = result2[1][0].values[0][2]
        error1 = result2[1][0].values[0][3]
        Flux_WCDA[int(i)-imin][0]=np.double(pow(10.,np.double(xx[0])))
        Flux_WCDA[int(i)-imin][1]=np.double(pow(10.,np.double(xx[1])))
        Flux_WCDA[int(i)-imin][2]=np.double(pow(10.,np.double(xx[2])))
        if func == fun_Logparabola:
            Flux_WCDA[int(i)-imin][3]=func(pow(10.,np.double(xx[0])),flux1, alpha,beta,piv)
        elif func == fun_Powerlaw:
            Flux_WCDA[int(i)-imin][3]=func(pow(10.,np.double(xx[0])),flux1, index, piv)
        Flux_WCDA[int(i)-imin][4]=Flux_WCDA[int(i)-imin][3]*(errorl/flux1)
        Flux_WCDA[int(i)-imin][5]=Flux_WCDA[int(i)-imin][3]*(erroru/flux1)
        Flux_WCDA[int(i)-imin][6]=Flux_WCDA[int(i)-imin][3]*(error1/flux1)
        # print(i, flux1, error1, Flux_WCDA[int(i)-imin][3], Flux_WCDA[int(i)-imin][6])
        if TSflux<0:
            TSflux=0
        Flux_WCDA[int(i)-imin][7]=np.sqrt(TSflux)
    activate_logs()
    return Flux_WCDA, results

def Draw_sepctrum_points(region_name, Modelname, Flux_WCDA, label = "Coma_data", color="tab:blue", aserror=False, ifsimpleTS=False, threshold=2, usexerr = False, ncut=True, subplot=None):
    Fluxdata = np.array([Flux_WCDA[:,0], 1e9*Flux_WCDA[:,3]*Flux_WCDA[:,0]**2, 1e9*Flux_WCDA[:,4]*Flux_WCDA[:,0]**2, 1e9*Flux_WCDA[:,5]*Flux_WCDA[:,0]**2,  1e9*Flux_WCDA[:,6]*Flux_WCDA[:,0]**2, Flux_WCDA[:,7], Flux_WCDA[:,1], Flux_WCDA[:,2]])
    """
        从能谱点矩阵画能谱点

        Parameters:
            aserror: 使用非对称误差
            ifsimpleTS: 是否仅仅用误差比来估计显著性
            threshold: 上限显著性阈值
            usexerr: 使用横向误差

            CL: 上限点取%多少上限?
            piv: 固定的piv energy, 需和之前相同
            nCL: 不使用扫描的上限
            threshold: 上限处理的显著性阈值, 默认2sigma
            ncut: 是否不允许负的流强, 强行拉到0
            subplot: 作为子图的ax
        
        Returns:
            >>> None
    """ 
    if subplot is not None:
        ax = subplot
    else:
        ax = plt.gca()

    np.savetxt(f'../res/{region_name}/{Modelname}/Spectrum_{label}.txt', Fluxdata, delimiter='\t', fmt='%e')
    if ncut==True:
        Flux_WCDA[:,3][Flux_WCDA[:,3]<0]=0
        Flux_WCDA[:,3][Flux_WCDA[:,7]<=0]=0

    if ifsimpleTS:
        npd = Flux_WCDA[:,3]/Flux_WCDA[:,6]>=threshold
    else:
        npd = Flux_WCDA[:,7]>=threshold
    if not usexerr:
        if aserror:
            ax.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                yerr=[1e9*Flux_WCDA[:,4][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,5][npd]*Flux_WCDA[:,0][npd]**2],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color)
            
            ax.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2, yerr=[1e9*Flux_WCDA[:,4][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,5][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            ax.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
        else:
            ax.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                        yerr=[1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2],\
                    #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
                    fmt='go',label=label,c=color)
            
            ax.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2, yerr=[1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            ax.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
    else:
        if aserror:
            ax.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                    xerr=[Flux_WCDA[:,0][npd]-Flux_WCDA[:,1][npd], Flux_WCDA[:,2][npd]-Flux_WCDA[:,0][npd]], yerr=[1e9*Flux_WCDA[:,4][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,5][npd]*Flux_WCDA[:,0][npd]**2],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color)
            
            ax.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2, xerr=[Flux_WCDA[:,0][~npd]-Flux_WCDA[:,1][~npd], Flux_WCDA[:,2][~npd]-Flux_WCDA[:,0][~npd]], yerr=[1e9*Flux_WCDA[:,4][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,5][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            ax.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
        else:
            ax.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                        xerr=[Flux_WCDA[:,0][npd]-Flux_WCDA[:,1][npd], Flux_WCDA[:,2][npd]-Flux_WCDA[:,0][npd]], yerr=[1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2],\
                    #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
                    fmt='go',label=label,c=color)
            
            ax.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2, xerr=[Flux_WCDA[:,0][~npd]-Flux_WCDA[:,1][~npd], Flux_WCDA[:,2][~npd]-Flux_WCDA[:,0][~npd]], yerr=[1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            ax.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
        

def Draw_spectrum_fromfile(file="/data/home/cwy/Science/3MLWCDA0.91/Standard/res/J0248/cdiff2D+2pt+freeDGE_0-5/Spectrum_J0248_data.txt", label="", color="red", aserror=False, ifsimpleTS=False, threshold=2, alpha=1, usexerr = False, ncut=False, scale=1, subplot=None):
    """
        从之前Draw_sepctrum_points 保存在res文件夹中的能谱点txt文件画能谱, 参数和Draw_sepctrum_points类似

        Parameters:
        
        Returns:
            >>> None
    """ 
    data = np.loadtxt(file)
    if ncut==True:
        data[1][data[1]<0]=0
        data[1][data[5]<=0]=0

    if subplot is not None:
        ax = subplot
    else:
        ax = plt.gca()

    if ifsimpleTS:
        try: 
            npd = data[1]/data[4]>=threshold
        except:
            npd = data[1]/data[2]>=threshold
    else:
        try: 
            npd = data[5]>=threshold
        except:
            npd = data[1]/data[2]>=threshold

    data[1] = data[1]*scale
    data[2] = data[2]*scale
    data[3] = data[3]*scale
    data[4] = data[4]*scale
    if not usexerr:
        if aserror:
            ax.errorbar(data[0][npd],data[1][npd],
                yerr=[data[2][npd],data[3][npd]],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color, alpha=alpha)
            
            ax.errorbar(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],
                            yerr=[data[2][~npd],data[3][~npd]],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=alpha)
            ax.scatter(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],marker=".",c=color, alpha=alpha)
        else:
            try: 
                ax.errorbar(data[0][npd],data[1][npd],data[4][npd],fmt="go", label=label, color=color, alpha=alpha)
            except:
                ax.errorbar(data[0][npd],data[1][npd],data[2][npd],fmt="go", label=label, color=color, alpha=alpha)
            
            try: 
                ax.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],data[4][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
            except:
                ax.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],data[2][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
                
            ax.scatter(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],marker=".",c=color, alpha=alpha)
    else:
        if aserror:
            ax.errorbar(data[0][npd],data[1][npd],
                xerr=[data[0][npd]-data[6][npd], data[7][npd]-data[0][npd]],
                yerr=[data[2][npd],data[3][npd]],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color, alpha=alpha)
            
            ax.errorbar(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],
                            xerr=[data[0][~npd]-data[6][~npd], data[7][~npd]-data[0][~npd]],
                            yerr=[data[2][~npd],data[3][~npd]],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=alpha)
            ax.scatter(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],marker=".",c=color, alpha=alpha)
        else:
            try: 
                ax.errorbar(data[0][npd],data[1][npd],
                             xerr=[data[0][npd]-data[6][npd], data[7][npd]-data[0][npd]],
                             yerr=data[4][npd],fmt="go", label=label, color=color, alpha=alpha)
            except:
                ax.errorbar(data[0][npd],data[1][npd],
                             xerr=[data[0][npd]-data[6][npd], data[7][npd]-data[0][npd]],
                             yerr=data[2][npd],fmt="go", label=label, color=color, alpha=alpha)
            try: 
                ax.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],
                             xerr=[data[0][~npd]-data[6][~npd], data[7][~npd]-data[0][~npd]],
                             yerr=data[4][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
            except:
                ax.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],
                             xerr=[data[0][~npd]-data[6][~npd], data[7][~npd]-data[0][~npd]],
                             yerr=data[2][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
                
            ax.scatter(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],marker=".",c=color, alpha=alpha)

    ax.set_xscale("log")
    ax.set_yscale("log")
    return data

def drawDig(file='./Coma_detect.csv',size=5, color="tab:blue", label="", fixx=1e-6, fixy=0.624, logx=1, logy=1, xbias=0, ybias=0, upthereshold=None):
    """
        对WebPlotDigitizer-4.6的点画图

        Parameters:
        
        Returns:
            >>> None
    """ 
    # print(file)
    data2 = pd.read_csv(file,sep=',',header=None)
    x2 = fixx*data2.iloc[:,0].values+xbias
    y2 = fixy*data2.iloc[:,1].values+ybias
    id2 = data2.iloc[:,2].values
    x2=x2.reshape([int(len(id2)/size),size])
    y2=y2.reshape([int(len(id2)/size),size])
    if upthereshold is not None:
        cut = np.abs(y2[:,0])/np.abs(y2[:,1]-y2[:,0])<upthereshold
    cut = np.abs(y2[:,1]-y2[:,0])/y2[:,0]<0.1
    # print(cut)
    plt.errorbar(x2[:,0][cut],y2[:,0][cut],0.5*y2[:,0][cut],fmt=".", xerr=np.array([x2[:,0][cut]-x2[:,3][cut], x2[:,4][cut]-x2[:,0][cut]]), color=color, uplims=True, label=label, capsize=3,) #
    plt.errorbar(x2[:,0][~cut],y2[:,0][~cut],y2[:,1][~cut]-y2[:,0][~cut], xerr=np.array([x2[:,0][~cut]-x2[:,3][~cut], x2[:,4][~cut]-x2[:,0][~cut]]), fmt="o", color=color, capsize=3,)
    if logy:
        plt.yscale("log")
    if logx:
        plt.xscale("log")

def drawspechsc(Energy, Flux, Ferr, Fc = 1e-14, label="", colorp="tab:blue", subplot=None, lm=False):
    """
        对hsc的矩阵画图

        Parameters:
        
        Returns:
            >>> None
    """ 
    if subplot is not None:
        ax = subplot
    else:
        ax = plt.gca()
    label2=""
    if lm:
        label2 = " WCDA"
    
    Energy = np.array(Energy)
    Flux = np.array(Flux)
    Ferr = np.array(Ferr)
    color = np.array([1,     1,     1,   1,     1,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,  0])
    color = color[:len(Energy)]
    ax.errorbar(Energy[Ferr!=0][color[Ferr!=0]==1],np.array(Flux[Ferr!=0][color[Ferr!=0]==1])*Fc,np.array(Ferr[Ferr!=0][color[Ferr!=0]==1])*Fc,marker="s",linestyle="none",color=colorp, label=label+label2)
    ax.errorbar(Energy[Ferr==0][color[Ferr==0]==1],np.array(Flux[Ferr==0][color[Ferr==0]==1])*Fc,0.2*np.array(Flux[Ferr==0][color[Ferr==0]==1])*Fc,marker=".",linestyle="none",color=colorp, uplims=True)

    if len(Energy)>6:
        ax.errorbar(Energy[Ferr!=0][color[Ferr!=0]==0],np.array(Flux[Ferr!=0][color[Ferr!=0]==0])*Fc,np.array(Ferr[Ferr!=0][color[Ferr!=0]==0])*Fc,marker="o",linestyle="none",color=colorp.replace("tab:", ""), label=label+" KM2A")
        ax.errorbar(Energy[Ferr==0][color[Ferr==0]==0],np.array(Flux[Ferr==0][color[Ferr==0]==0])*Fc,0.2*np.array(Flux[Ferr==0][color[Ferr==0]==0])*Fc,marker=".",linestyle="none",color=colorp.replace("tab:", ""), uplims=True)

    ax.set_xlabel(r"$E (TeV)$")
    ax.set_ylabel(r"$TeV cm^{-2}s^{-1}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.ylim(1e-17,5e-10)
    ax.legend()

def spec2naima(dir, data0057):
    """
        将读取的能谱点矩阵转化为naima方便复制的形式

        Parameters:
        
        Returns:
            >>> None
    """ 
    dataforlb = np.zeros(data0057.shape)
    dataforlb[0] = data0057[0]
    dataforlb[1] = data0057[6]
    dataforlb[2] = data0057[7]
    dataforlb[3] = data0057[1]
    dataforlb[4] = data0057[2]
    dataforlb[5] = data0057[3]
    dataforlb[6] = data0057[4]
    dataforlb[7] = data0057[5]
    header = "E(TeV) E_68s E_68e flux(TeV/cm**2/s) fluxe fluxel fluxeu TS"
    np.savetxt(dir,dataforlb.T, fmt='%.4e', header=header)
    return dataforlb
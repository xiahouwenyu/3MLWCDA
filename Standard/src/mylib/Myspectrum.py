from threeML import *
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
import copy
from Myfit import *

import ROOT

import numpy as np

import ctypes

import scipy as sp

import matplotlib.pyplot as plt

from Myspec import *

def cal_K_WCDA(i,lm,maptree,response,roi,source="J0248", ifgeterror=False, mini="ROOT", ifpowerlawM=False, CL=0.95, nCL=False):
    #Only fit the spectrum.K for plotting  points on the spectra
        #prarm1: fixed.spectrum.alpha 
        #param2: fixed.spectrum.belta
        #return: spectrum.K
    # Instance the plugin
    WCDA_1 = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
    lm2 = copy.deepcopy(lm)
    fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    # Define model
    sources = lm2.sources.keys()
    if ifpowerlawM:
        oldscs = copy.deepcopy(lm2.sources)
        for ss in sources:
            par = copy.deepcopy(lm2.sources[ss].parameters)
            if ss==source:
                lm2.remove_source(ss)
                if str(oldscs[ss].source_type) == "point source":
                    lm2.add_source(PointSource(ss, 0, 0, spectral_shape=PowerlawM()))
                elif str(oldscs[ss].source_type) == "extended source":
                    lm2.add_source(ExtendedSource(ss, spatial_shape=oldscs[ss].spatial_shape, spectral_shape=PowerlawM()))
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
                    lm2.sources[ss].parameters[newpa].bounds=(-1e-11,1e-11)*fluxUnit
    else:
        for ss in sources:
            freep = lm2.sources[ss].free_parameters.keys()
            for fp in freep:
                if (ss == source and ".K" not in fp) or ss != source:
                    lm2.sources[ss].free_parameters[fp].fix = True
                elif (ss == source and ".K" in fp and lm.sources[ss].components['main'].shape.name=="PowerlawM"):
                    lm2.sources[ss].free_parameters[fp].bounds=(-1e-11,1e-11)
                else:
                    kparname=fp

    result2 = fit("nothing","nothing", WCDA_1,lm2,int(i),int(i),mini=mini,savefit=False, ifgeterror=ifgeterror)

    TSflux=result2[0].compute_TS(source,result2[1][1]).values[0][2]
    if not nCL:
        if ifpowerlawM:
            lb, ub = result2[0].results.get_equal_tailed_interval(lm2.sources[source].parameters[kparname], cl=2*CL-1)
            if (ub-result2[1][0].iloc[0,0])>0:
                result2[1][0].iloc[0,2] = (ub-result2[1][0].iloc[0,0])/1.96
            if result2[1][0].loc[kparname,"value"]<0:
                TSflux=-TSflux
        else:
            lb, ub = result2[0].results.get_equal_tailed_interval(lm2.sources[source].parameters[kparname.replace("PowerlawM","Powerlaw")], cl=2*CL-1)
            if (ub-result2[1][0].iloc[0,0])>0:
                result2[1][0].iloc[0,2] = (ub-result2[1][0].iloc[0,0])/1.96
    return result2, TSflux

def reweightx(lm,WCDA,i,func = fun_Logparabola,source="J0248"):
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

def getdatapoint(Detector, lm, maptree,response,roi, source="J0248", ifgeterror=False, mini="ROOT", ifpowerlawM=False, CL=0.95, piv = 3, nCL=False):
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
    jls = []
    for i in Detector._active_planes:
        if int(i) <= imin:
            imin = int(i)
        xx = reweightx(lm,Detector, i,source=source,func=func)
        result2, TSflux=cal_K_WCDA(i,lm, maptree,response,roi, source=source, ifgeterror=ifgeterror, mini=mini, ifpowerlawM=ifpowerlawM, CL=CL, nCL=False)
        jls.append(result2[0])
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
        if TSflux<0:
            TSflux=0
        Flux_WCDA[int(i)-imin][7]=np.sqrt(TSflux)
    return Flux_WCDA, jls

def Draw_sepctrum_points(region_name, Modelname, Flux_WCDA, label = "Coma_data", color="tab:blue", aserror=False, ifsimpleTS=False, threshold=2, usexerr = False, ncut=True):
    Fluxdata = np.array([Flux_WCDA[:,0], 1e9*Flux_WCDA[:,3]*Flux_WCDA[:,0]**2, 1e9*Flux_WCDA[:,4]*Flux_WCDA[:,0]**2, 1e9*Flux_WCDA[:,5]*Flux_WCDA[:,0]**2,  1e9*Flux_WCDA[:,6]*Flux_WCDA[:,0]**2, Flux_WCDA[:,7], Flux_WCDA[:,1], Flux_WCDA[:,2]])
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
            plt.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                yerr=[1e9*Flux_WCDA[:,4][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,5][npd]*Flux_WCDA[:,0][npd]**2],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color)
            
            plt.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2, yerr=[1e9*Flux_WCDA[:,4][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,5][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            plt.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
        else:
            plt.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                        yerr=[1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2],\
                    #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
                    fmt='go',label=label,c=color)
            
            plt.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2, yerr=[1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            plt.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
    else:
        if aserror:
            plt.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                    xerr=[Flux_WCDA[:,0][npd]-Flux_WCDA[:,1][npd], Flux_WCDA[:,2][npd]-Flux_WCDA[:,0][npd]], yerr=[1e9*Flux_WCDA[:,4][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,5][npd]*Flux_WCDA[:,0][npd]**2],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color)
            
            plt.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2, xerr=[Flux_WCDA[:,0][~npd]-Flux_WCDA[:,1][~npd], Flux_WCDA[:,2][~npd]-Flux_WCDA[:,0][~npd]], yerr=[1e9*Flux_WCDA[:,4][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,5][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            plt.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,5][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
        else:
            plt.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                        xerr=[Flux_WCDA[:,0][npd]-Flux_WCDA[:,1][npd], Flux_WCDA[:,2][npd]-Flux_WCDA[:,0][npd]], yerr=[1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2],\
                    #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
                    fmt='go',label=label,c=color)
            
            plt.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2, xerr=[Flux_WCDA[:,0][~npd]-Flux_WCDA[:,1][~npd], Flux_WCDA[:,2][~npd]-Flux_WCDA[:,0][~npd]], yerr=[1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=1)
            plt.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+1.96*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
        

def Draw_spectrum_fromfile(file="/data/home/cwy/Science/3MLWCDA0.91/Standard/res/J0248/cdiff2D+2pt+freeDGE_0-5/Spectrum_J0248_data.txt", label="", color="red", aserror=False, ifsimpleTS=False, threshold=2, alpha=1, usexerr = False):
    data = np.loadtxt(file)
    data[1][data[1]<0]=0
    data[1][data[5]<=0]=0

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
    if not usexerr:
        if aserror:
            plt.errorbar(data[0][npd],data[1][npd],
                yerr=[data[2][npd],data[3][npd]],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color, alpha=alpha)
            
            plt.errorbar(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],
                            yerr=[data[2][~npd],data[3][~npd]],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=alpha)
            plt.scatter(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],marker=".",c=color, alpha=alpha)
        else:
            try: 
                plt.errorbar(data[0][npd],data[1][npd],data[4][npd],fmt="go", label=label, color=color, alpha=alpha)
            except:
                plt.errorbar(data[0][npd],data[1][npd],data[2][npd],fmt="go", label=label, color=color, alpha=alpha)
            
            try: 
                plt.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],data[4][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
            except:
                plt.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],data[2][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
                
            plt.scatter(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],marker=".",c=color, alpha=alpha)
    else:
        if aserror:
            plt.errorbar(data[0][npd],data[1][npd],
                xerr=[data[0][npd]-data[6][npd], data[7][npd]-data[0][npd]],
                yerr=[data[2][npd],data[3][npd]],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color, alpha=alpha)
            
            plt.errorbar(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],
                            xerr=[data[0][~npd]-data[6][~npd], data[7][~npd]-data[0][~npd]],
                            yerr=[data[2][~npd],data[3][~npd]],
                            uplims=True,
                            marker="None", color=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=2.5, linestyle="None", alpha=alpha)
            plt.scatter(data[0][~npd],data[1][~npd]+1.96*data[3][~npd],marker=".",c=color, alpha=alpha)
        else:
            try: 
                plt.errorbar(data[0][npd],data[1][npd],
                             xerr=[data[0][npd]-data[6][npd], data[7][npd]-data[0][npd]],
                             yerr=data[4][npd],fmt="go", label=label, color=color, alpha=alpha)
            except:
                plt.errorbar(data[0][npd],data[1][npd],
                             xerr=[data[0][npd]-data[6][npd], data[7][npd]-data[0][npd]],
                             yerr=data[2][npd],fmt="go", label=label, color=color, alpha=alpha)
            try: 
                plt.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],
                             xerr=[data[0][~npd]-data[6][~npd], data[7][~npd]-data[0][~npd]],
                             yerr=data[4][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
            except:
                plt.errorbar(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],
                             xerr=[data[0][~npd]-data[6][~npd], data[7][~npd]-data[0][~npd]],
                             yerr=data[2][~npd],
                                uplims=True,
                                marker="None", color=color,
                                markeredgecolor=color, markerfacecolor=color,
                                linewidth=2.5, linestyle="None", alpha=alpha)
                
            plt.scatter(data[0][~npd],data[1][~npd]+1.96*data[2][~npd],marker=".",c=color, alpha=alpha)

    plt.xscale("log")
    plt.yscale("log")
    return data

def drawDig(file='./Coma_detect.csv',size=5, color="tab:blue", label="", fixx=1e-6, fixy=0.624):
    # print(file)
    data2 = pd.read_csv(file,sep=',',header=None)
    x2 = fixx*data2.iloc[:,0].values
    y2 = fixy*data2.iloc[:,1].values
    id2 = data2.iloc[:,2].values
    x2=x2.reshape([int(len(id2)/size),size])
    y2=y2.reshape([int(len(id2)/size),size])
    cut = np.abs(y2[:,1]-y2[:,0])/y2[:,0]<0.1
    # print(cut)
    plt.errorbar(x2[:,0][cut],y2[:,0][cut],0.5*y2[:,0][cut],fmt=".", xerr=np.array([x2[:,0][cut]-x2[:,3][cut], x2[:,4][cut]-x2[:,0][cut]]), color=color, uplims=True, label=label, capsize=3,) #
    plt.errorbar(x2[:,0][~cut],y2[:,0][~cut],y2[:,1][~cut]-y2[:,0][~cut], xerr=np.array([x2[:,0][~cut]-x2[:,3][~cut], x2[:,4][~cut]-x2[:,0][~cut]]), fmt="o", color=color, capsize=3,)
    plt.yscale("log")
    plt.xscale("log")

def drawspechsc(Energy, Flux, Ferr, Fc = 1e-14, label=""):
    Energy = np.array(Energy)
    Flux = np.array(Flux)
    Ferr = np.array(Ferr)
    color = np.array([1,     1,     1,   1,     1,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,  0])
    color = color[:len(Energy)]
    plt.errorbar(Energy[Ferr!=0][color[Ferr!=0]==1],np.array(Flux[Ferr!=0][color[Ferr!=0]==1])*Fc,np.array(Ferr[Ferr!=0][color[Ferr!=0]==1])*Fc,marker="s",linestyle="none",color="tab:blue", label=label+" WCDA")
    plt.errorbar(Energy[Ferr==0][color[Ferr==0]==1],np.array(Flux[Ferr==0][color[Ferr==0]==1])*Fc,0.2*np.array(Flux[Ferr==0][color[Ferr==0]==1])*Fc,marker=".",linestyle="none",color="tab:blue", uplims=True)

    if len(Energy)>6:
        plt.errorbar(Energy[Ferr!=0][color[Ferr!=0]==0],np.array(Flux[Ferr!=0][color[Ferr!=0]==0])*Fc,np.array(Ferr[Ferr!=0][color[Ferr!=0]==0])*Fc,marker="o",linestyle="none",color="cornflowerblue", label=label+" KM2A")
        plt.errorbar(Energy[Ferr==0][color[Ferr==0]==0],np.array(Flux[Ferr==0][color[Ferr==0]==0])*Fc,0.2*np.array(Flux[Ferr==0][color[Ferr==0]==0])*Fc,marker=".",linestyle="none",color="cornflowerblue", uplims=True)

    plt.xlabel(r"$E (TeV)$")
    plt.ylabel(r"$TeV cm^{-2}s^{-1}$")
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-17,5e-10)
    plt.legend()

def spec2naima(dir, data0057):
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
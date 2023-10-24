from threeML import *
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
import copy
from Myfit import *

import ROOT

import numpy as np

import ctypes

import scipy as sp

import matplotlib.pyplot as plt

def cal_K_WCDA(i,lm,maptree,response,roi,source="J0248"):
    #Only fit the spectrum.K for plotting  points on the spectra
        #prarm1: fixed.spectrum.alpha 
        #param2: fixed.spectrum.belta
        #return: spectrum.K
    # Instance the plugin
    WCDA_1 = HAL("WCDA", maptree, response, roi, flat_sky_pixels_size=0.17)
    lm2 = copy.deepcopy(lm)
    # Define model
    sources = lm2.sources.keys()
    for ss in sources:
        freep = lm2.sources[ss].free_parameters.keys()
        for fp in freep:
            if ss == source and ".K" not in fp or ss != source:
                lm2.sources[ss].free_parameters[fp].fix = True
    result2 = fit("nothing","nothing", WCDA_1,lm2,int(i),int(i),mini="ROOT",savefit=False)
    return result2

def reweightx(lm,resp,i,func = fun_Logparabola,source="J0248"):
    sources = lm.sources.keys()
    sbl=resp[i]
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

def getdatapoint(WCDA, lm, maptree,response,roi, source="J0248"):
    Flux_WCDA=np.zeros((len(WCDA._active_planes),7), dtype=np.double())
    resp = WCDA._response.get_response_dec_bin(WCDA._roi.ra_dec_center[1])
    # piv = result[1][0].values[3][0]/1e9
    piv = 3
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
    for i in WCDA._active_planes:
        if int(i) <= imin:
            imin = int(i)
        xx = reweightx(lm,resp,i,source=source,func=func)
        result2=cal_K_WCDA(i,lm, maptree,response,roi, source=source)
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
    return Flux_WCDA

def Draw_sepctrum_points(region_name, Modelname, Flux_WCDA, label = "Coma_data", color="tab:blue"):
    Fluxdata = np.array([Flux_WCDA[:,0], 1e9*Flux_WCDA[:,3]*Flux_WCDA[:,0]**2, 1e9*Flux_WCDA[:,6]*Flux_WCDA[:,0]**2])
    np.savetxt(f'../res/{region_name}/{Modelname}/Spectrum_{label}.txt', Fluxdata, delimiter='\t', fmt='%e')

    npd = Flux_WCDA[:,3]/Flux_WCDA[:,6]>=3
    plt.errorbar(Flux_WCDA[:,0][npd],1e9*Flux_WCDA[:,3][npd]*Flux_WCDA[:,0][npd]**2,\
                yerr=[1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2, 1e9*Flux_WCDA[:,6][npd]*Flux_WCDA[:,0][npd]**2],\
            #  xerr=[Flux_WCDA[:,1],Flux_WCDA[:,2]],\
            fmt='go',label=label,c=color)
    
    plt.errorbar(Flux_WCDA[:,0][~npd], 1e9*(Flux_WCDA[:,3][~npd]+2*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2, yerr=[1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2, 1e9*Flux_WCDA[:,6][~npd]*Flux_WCDA[:,0][~npd]**2],
                    uplims=True,
                    marker="None", color=color,
                    markeredgecolor=color, markerfacecolor=color,
                    linewidth=2.5, linestyle="None", alpha=1)
    plt.scatter(Flux_WCDA[:,0][~npd],1e9*(Flux_WCDA[:,3][~npd]+2*Flux_WCDA[:,6][~npd])*Flux_WCDA[:,0][~npd]**2,marker=".",c=color)
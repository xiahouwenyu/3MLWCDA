#!/usr/bin/python3

from astropy import units as u
from astropy.coordinates import SkyCoord
from ROOT import TFile,TTree
import healpy as hp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import cm
from scipy.optimize import curve_fit
from array import array

from astropy.io import fits as pyfits
from astropy.wcs import wcs
import sys
import argparse

p = argparse.ArgumentParser()
p.add_argument("-i", "--input", dest="infile")
p.add_argument("-o", "--output", dest="outfile")
p.add_argument("-p", "--psf", dest="psf",type=float)
p.add_argument("-n", "--nhitB", dest="nhitB",type=int)

args = p.parse_args()

filein=args.infile
fileout=args.outfile
smooth_sigma=args.psf
nhit=args.nhitB

file1=TFile.Open(filein)
nside=2**10
npix=hp.nside2npix(nside)
pixarea = 4 * np.pi/npix
signal=np.zeros(npix,dtype=np.float64)
background=np.zeros(npix,dtype=np.float64)
for ii in range(nhit,nhit+1): #nhit bin
    print("nHit%.2d"%ii)
    nHit0=file1.GetDirectory("nHit%.2d"%ii)
    T_signal=nHit0.Get("data")
    T_background=nHit0.Get("bkg")
    for iev in range(T_signal.GetEntries()):
        if(iev%1000000==0):
            print(iev,T_signal.count)
        T_signal.GetEntry(iev)
        signal[iev]+=T_signal.count
        T_background.GetEntry(iev)
        background[iev]+=T_background.count

    pixIdx = hp.nside2npix(nside) # number of pixels I can get from this nside 
    pixIdx = np.arange(pixIdx) # pixel index numbers
    new_lats = hp.pix2ang(nside, pixIdx)[0] # thetas I need to populate with interpolated theta values
    new_lons = hp.pix2ang(nside, pixIdx)[1] # phis, same
    mask = ((-new_lats + np.pi/2 < -20./180*np.pi) | (-new_lats + np.pi/2 > 80./180*np.pi)  )  # mask dec > 80 && dec<-20
    
    signal[mask]=hp.UNSEEN
    background[mask]=hp.UNSEEN

    signal=hp.ma(signal)
    background=hp.ma(background)

    #Smoothes the map with a 1-degree FWHM Gaussian (fwhm given in radians)
    signal_smoothed=hp.sphtfunc.smoothing(signal,sigma=np.radians(smooth_sigma))#*pixarea
    background_smoothed=hp.sphtfunc.smoothing(background,sigma=np.radians(smooth_sigma))#*pixarea
    background_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(background,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea
    signal_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(signal,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea
    excess=signal-background
    excess_smoothed = signal_smoothed - background_smoothed
    excess_smoothed[mask]=hp.UNSEEN
    excess_smoothed=hp.ma(excess_smoothed)

    hp.mollview(excess_smoothed,title="Mollview image RING",norm='hist',unit='Excess')
    hp.mollview(excess,title="Mollview image RING",norm='hist',unit='Excess')
    hp.graticule()
    plt.savefig("%s_excess_nHit0%02d_%.2f.pdf"%(filein[0:-5],ii, smooth_sigma))
   
#    hp.mollview(signal_smoothed,title="Mollview image RING",norm='hist',unit='ON')
#    hp.mollview(signal,title="Mollview image RING",norm='hist',unit='ON')
#    hp.graticule()
#    plt.savefig("%s_on_nHit0%02d_%.2f.pdf"%(filein[0:-5],ii, smooth_sigma))

#    hp.mollview(background_smoothed,title="Mollview image RING",norm='hist',unit='OFF')
#    hp.mollview(background,title="Mollview image RING",norm='hist',unit='OFF')
#    hp.graticule()
#    plt.savefig("%s_off_nHit0%02d_%.2f.pdf"%(filein[0:-5],ii, smooth_sigma))


    signal_smoothed[mask]=hp.UNSEEN
    signal_smoothed2[mask]=hp.UNSEEN
    background_smoothed[mask]=hp.UNSEEN
    background_smoothed2[mask]=hp.UNSEEN

    signal_smoothed=hp.ma(signal_smoothed)
    signal_smoothed2=hp.ma(signal_smoothed2)
    background_smoothed=hp.ma(background_smoothed)
    background_smoothed2=hp.ma(background_smoothed2)

    print("sum_on   = %f sum_bk   = %f "%(sum(signal[~mask]),sum(background[~mask])))
    print("sum_ons  = %f sum_bks  = %f "%(sum(signal_smoothed[~mask]),sum(background_smoothed[~mask])))
    print("sum_ons2 = %f sum_bks2 = %f "%(sum(signal_smoothed2[~mask]),sum(background_smoothed2[~mask])))
    
    hp.write_map("%s_nHit0%02d_%.2f.fits.gz"%(filein[0:-5], ii, smooth_sigma),[signal, background, signal_smoothed, background_smoothed, signal_smoothed2, background_smoothed2],overwrite=True)

#!/usr/bin/python3
import healpy as hp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import cm
from scipy.optimize import curve_fit
from array import array
import argparse

print("start1")
p = argparse.ArgumentParser()

p.add_argument("-i", "--input", dest="inmap")
p.add_argument("-o", "--output", dest="outmap")
p.add_argument("-s", "--signif", dest="signif",type=int)
p.add_argument("-p","--psf",dest="psf",type=float)
args = p.parse_args()

print("start")
inmap=args.inmap
outmap=args.outmap
signif=args.signif
psf=args.psf

evt = hp.read_map(inmap, field=0)
evt=np.ma.abs(evt)
bkg = hp.read_map(inmap, field=1)
bkg=np.ma.abs(bkg)
evt_smoothed = hp.read_map(inmap, field=2)
evt_smoothed=np.ma.abs(evt_smoothed)
bkg_smoothed = hp.read_map(inmap, field=3)
bkg_smoothed=np.ma.abs(bkg_smoothed)
evt_smoothed2 = hp.read_map(inmap, field=4)
evt_smoothed2=np.ma.abs(evt_smoothed2)
bkg_smoothed2 = hp.read_map(inmap, field=5)
bkg_smoothed2=np.ma.abs(bkg_smoothed2)
print("readover")
bkg_smoothed2[bkg_smoothed2==0]=0.01
evt_smoothed2[evt_smoothed2==0]=0.01
scale=(evt_smoothed+bkg_smoothed)/(bkg_smoothed2+evt_smoothed2)
scale_evt=(evt_smoothed)/(evt_smoothed2)
scale_bkg=(bkg_smoothed)/(bkg_smoothed2)
scale_evt = scale
scale_bkg = scale
ON=evt_smoothed*scale_evt
BK=bkg_smoothed*scale_bkg
ON[ON==0]=0.01
BK[BK==0]=0.01

nside=2**10
npix=hp.nside2npix(nside)
pixIdx = hp.nside2npix(nside) # number of pixels I can get from this nside 
pixIdx = np.arange(pixIdx) # pixel index numbers
new_lats = hp.pix2ang(nside, pixIdx)[0] # thetas I need to populate with interpolated theta values
new_lons = hp.pix2ang(nside, pixIdx)[1] # phis, same
vec = hp.ang2vec((90-6.28)/180*np.pi,286.92/180*np.pi)
mask = ((-new_lats + np.pi/2 < -20./180*np.pi) | (-new_lats + np.pi/2 > 80./180*np.pi)  )  # mask dec > 80 && dec<-20
#unmask = hp.query_disc(1024,vec=vec , radius=np.radians(6))
PSF = psf
alpha = np.zeros(npix)
for i in range(npix):
    theta, phi = hp.pix2ang(nside, i)
    alpha[i] =2* PSF*1.51/60./np.sin(theta)

if signif==5:
    S=(ON-BK)/np.sqrt(ON+alpha*BK)
elif signif==9:
    S=(ON-BK)/np.sqrt(ON*alpha+BK)
elif signif==17:
    # if ON*np.log((1.+alpha)/alpha*ON/(ON+BK/alpha))+BK/alpha*np.log((1.+alpha)*BK/alpha/(ON+BK/alpha))<0:
    #      S=0
    # else:
    S=np.sqrt(2.)*np.sqrt(ON*np.log((1.+alpha)/alpha*ON/(ON+BK/alpha))+BK/alpha*np.log((1.+alpha)*BK/alpha/(ON+BK/alpha)))
    S[ON<BK] *= -1
else:
    S=(ON-BK)/np.sqrt(BK)

#    unmask =hp.query_disc(1024, vec=vec , radius=np.radians(8))    
#    for pixn in unmask:
#        pixIdx=np.delete(pixIdx,unmask)
#        signal[pixIdx]=hp.UNSEEN
#        background[pixIdx]=hp.UNSEEN
#        S[pixIdx]=hp.UNSEEN
#signal[mask]=hp.UNSEEN
#background[mask]=hp.UNSEEN
#signal=hp.ma(signal)
#background=hp.ma(background)
#pixIdx=np.delete(pixIdx,unmask)
S[mask]=hp.UNSEEN
S = hp.ma(S)



def gaussian(x,a,mu,sigma):
    return a*np.exp(-((x-mu)/sigma)**2/2)

#bin_y,bin_x,patches=plt.hist(S[~mask],bins=100)
#popt,pcov = curve_fit(gaussian,(bin_x[0:100]+bin_x[1:101])/2,bin_y,bounds=([10000,-2,0],[50000000,2,5]))
#popt,pcov = curve_fit(gaussian,(bin_x[0:100]+bin_x[1:101])/2,bin_y,bounds=([10000,-2,0],[50000000,2,5]))

bin_y,bin_x,patches=plt.hist(S,bins=100)
bin_x=np.array(bin_x)
bin_y=np.array(bin_y)
fit_range = np.logical_and(bin_x>-5, bin_x<5)
wdt=(bin_x[1]-bin_x[0])/2.
try:
	popt,pcov = curve_fit(gaussian,bin_x[fit_range]+wdt,bin_y[fit_range[0:-1]],bounds=([100,-2,0],[50000000,2,10]))
except (ValueError, IndexError):
	popt,pcov = curve_fit(gaussian,bin_x[0:100]+wdt,bin_y[0:100],bounds=([100,-2,0],[50000000,2,10]))
#popt,pcov = curve_fit(gaussian,bin_x[fit_range[0:-1]]+(bin_x[1]-bin_x[0])/2.,bin_y[fit_range[0:-1]],bounds=([100,-2,0],[50000000,2,10]))
print("************************")
print(popt)
print("************************")
print("max Significance= %.1f"%(max(S)))

plt.figure()
#plt.plot([0.,0.],[1,1e6],'k--',linewidth=0.5)
plt.plot((bin_x[0:100]+bin_x[1:101])/2,gaussian((bin_x[0:100]+bin_x[1:101])/2,popt[0],0,1),'--',label='expectation')
plt.plot((bin_x[0:100]+bin_x[1:101])/2,bin_y,label="data")
plt.plot((bin_x[0:100]+bin_x[1:101])/2,gaussian((bin_x[0:100]+bin_x[1:101])/2,popt[0],popt[1],popt[2]),'--',label='fit')
plt.yscale('log')
plt.xlim(-10,10)
plt.ylim(1,10000000*2)
plt.grid(True)
plt.text(-9.5,2e5,'mean = %f\n width = %f'%(popt[1],popt[2]))
plt.xlabel(r'Significance($\sigma$)')
plt.ylabel("entries")
plt.legend()
plt.savefig("hist_sig_%s.pdf"%outmap)

hp.write_map("signif_%s.fits.gz"%outmap,S,overwrite=True)

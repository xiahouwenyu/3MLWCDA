import uproot
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import numpy as np
import healpy as hp
from tqdm import tqdm
import sys
import ROOT as rt
import root_numpy as rn

data_name = "/home/lhaaso/caowy/data.root"
file = uproot.open(data_name)
histogram_on = file["all_sky_cube_on;1"]
histogram_bg = file["all_sky_cube_bg;1"]
histogram_off = file["all_sky_cube_off;1"]
header = file["header"].title.split(",")
header_value=file["header"].values()
print(header,header_value)

nside=1024
npix=hp.nside2npix(nside)
dtype1 = [('count', float)]
dtype2 = [('count', float)]
skymaponout=np.zeros(npix) #, dtype = dtype1
skymapoffout=np.zeros(npix) #, dtype = dtype2
pixid = np.arange(npix)
pixarea= 4*np.pi/npix
new_lats = 90-hp.pix2ang(nside, pixid)[0]*180/np.pi # thetas I need to populate with interpolated theta values
new_lons = hp.pix2ang(nside, pixid)[1]*180/np.pi # phis, same

ras = histogram_on.axis(0).centers()
decs = histogram_on.axis(1).centers()

j=int(sys.argv[1])
with open("/home/lhaaso/caowy/LHAASO_skytrans/skytxt/KM2A_sky_on_off_part%i.txt"%(j),"a+") as fs:
    for bin in range(14):
        print('nHit%02d'%int(bin))
        skymap = histogram_on.values()[:,:,bin].T
        skymapbg = histogram_bg.values()[:,:,bin].T
        skymapoff = histogram_off.values()[:,:,bin].T
        fon = interp2d(ras,decs,skymap,kind='linear')
        fbkg = interp2d(ras,decs,skymapbg,kind='linear')
        foff = interp2d(ras,decs,skymapoff,kind='linear')

        for ipix in tqdm(pixid[j*12288:(j+1)*12288]):
            ra_pix , dec_pix = hp.pix2ang(1024,ipix,lonlat=True)
            if (dec_pix<-20.) | (dec_pix>80.):
                skymaponout[ipix]=hp.UNSEEN
                skymapoffout[ipix]=hp.UNSEEN
                continue
            skymaponout[ipix]=np.random.poisson(fon(ra_pix,dec_pix)/(np.radians(0.1)*np.radians(0.1)*np.cos(np.radians(dec_pix)))*pixarea)
            skymapoffout[ipix]=fbkg(ra_pix,dec_pix)/(np.radians(0.1)*np.radians(0.1)*np.cos(np.radians(dec_pix)))*pixarea
            skyoff = foff(ra_pix,dec_pix)/(np.radians(0.1)*np.radians(0.1)*np.cos(np.radians(dec_pix)))*pixarea
            fs.write(str(bin)+" "+str(0)+" "+str(ipix)+" "+str(skymaponout[ipix])+"\n")
            fs.write(str(bin)+" "+str(1)+" "+str(ipix)+" "+str(skymapoffout[ipix])+"\n")
            fs.write(str(bin)+" "+str(2)+" "+str(ipix)+" "+str(skyoff[0])+"\n")
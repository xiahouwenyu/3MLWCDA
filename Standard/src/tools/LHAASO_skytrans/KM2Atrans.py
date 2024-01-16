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
for bin in range(14):
    print('nHit%02d'%int(bin))
    skymap = histogram_on.values()[:,:,bin].T
    skymapbg = histogram_bg.values()[:,:,bin].T
    skymapoff = histogram_off.values()[:,:,bin].T
    fon = interp2d(ras,decs,skymap,kind='linear')
    fbkg = interp2d(ras,decs,skymapbg,kind='linear')
    foff = interp2d(ras,decs,skymapoff,kind='linear')
    
    # for j in range(len(decs)):
    dec=decs[j]
    if (dec>=-25 and dec<=85):
        with open("/home/lhaaso/caowy/LHAASO_skytrans/skytxt/KM2A_sky_on_off_dec%i.txt"%(j),"a+") as fs:
            for i in tqdm(range(len(ras))):
                ra=ras[i]
                pick = (new_lons>ra-0.05) & (new_lons<ra+0.05) & (new_lats>dec-0.05) & (new_lats<dec+0.05)
                pixneed = pixid[pick]
                lens = np.sum(pick)
                if lens:
                    # poissonrand = np.random.poisson(skymap[j][i]/lens, lens)
                    for k in range(lens):
                        ra_pix , dec_pix = hp.pix2ang(1024,pixneed[k],lonlat=True)
                        con = np.random.poisson(fon(ra_pix,dec_pix)/(np.radians(0.1)*np.radians(0.1)*np.cos(np.radians(dec_pix)))*pixarea)
                        coff = fbkg(ra_pix,dec_pix)/(np.radians(0.1)*np.radians(0.1)*np.cos(np.radians(dec_pix)))*pixarea
                        fs.write(str(bin)+" "+str(0)+" "+str(pixneed[k])+" "+str(poissonrand[k])+"\n")
                        fs.write(str(bin)+" "+str(1)+" "+str(pixneed[k])+" "+str(coff[0])+"\n")
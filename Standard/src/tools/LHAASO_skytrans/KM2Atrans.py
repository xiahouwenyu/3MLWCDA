import uproot
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import numpy as np
import healpy as hp
from tqdm import tqdm
import sys
import ROOT as rt
import root_numpy as rn

data_name = "/data/home/cwy/Science/data.root"
file = uproot.open(data_name)
histogram_on = file["all_sky_cube_on;1"]
histogram_off = file["all_sky_cube_bg;1"]
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
skymap = histogram_on.values()[:,:,0].T
# for j in range(len(decs)):
j=int(sys.argv[1])
dec=decs[j]
if (dec>=-25 and dec<=85):
    with open("/data/home/cwy/Science/3MLWCDA/Standard/src/tools/LHAASO_skytrans/skytxt/sky_on%i.txt"%j,"a+") as fs:
        for i in tqdm(range(len(ras))):
            ra=ras[i]
            pick = (new_lons>ra-0.05) & (new_lons<ra+0.05) & (new_lats>dec-0.05) & (new_lats<dec+0.05)
            lens = np.sum(pick)
            if lens:
                poissonrand = np.random.poisson(skymap[i][j]/lens, lens)
                for k in range(lens):
                    fs.write(str(pixid[pick][k])+" "+str(poissonrand[k])+"\n")
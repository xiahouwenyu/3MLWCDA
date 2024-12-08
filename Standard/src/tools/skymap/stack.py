import argparse
import numpy as np
import healpy as hp

p = argparse.ArgumentParser()

p.add_argument("-i", "--input", dest="infile")
p.add_argument("-o", "--output", dest="outfile")
p.add_argument("-w", "--", dest="weight")
args = p.parse_args()

files = (args.infile).split()
weight = (args.weight).split()

weight = []
for w in (args.weight).split():
    weight.append(float(w))
weight = np.asarray(weight)

if len(weight) != len(files):
    exit()

nmaps = len(weight)

nside=2**10
npix=hp.nside2npix(nside)
maps = np.zeros((nmaps, 6, npix))

pixIdx = hp.nside2npix(nside) # number of pixels I can get from this nside 
pixIdx = np.arange(pixIdx) # pixel index numbers
new_lats = hp.pix2ang(nside, pixIdx)[0] # thetas I need to populate with interpolated theta values
new_lons = hp.pix2ang(nside, pixIdx)[1] # phis, same
mask = ((-new_lats + np.pi/2 < -20./180*np.pi) | (-new_lats + np.pi/2 > 80./180*np.pi)  )  # mask dec > 80 && dec<-20

for i, f in enumerate(files):
    for j in range(6):
        maps[i][j] = hp.read_map(f, field=j)
        maps[i][j] *= weight[i]
        if j==4 or j==5:
            maps[i][j] *= weight[i]
    print("sum_ons   = %f sum_ons2   = %f "%(sum(maps[i][2][~mask]),sum(maps[i][4][~mask])))

outmap = sum(maps)

for j in range(6):
    outmap[j][mask]=hp.UNSEEN
    outmap[j]=hp.ma(outmap[j])

print("sum_ons   = %f sum_ons2   = %f "%(sum(outmap[2][~mask]),sum(outmap[4][~mask])))
    
hp.write_map(args.outfile,outmap,overwrite=True)

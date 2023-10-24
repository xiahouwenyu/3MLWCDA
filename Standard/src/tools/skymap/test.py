import healpy as hp
inmap="./20210305_20230731_ihep_goodlist_nHit006_0.29.fits.gz"
evt = hp.read_map(inmap, field=2)
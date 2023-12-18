#!/bin/bash
myname=../../../../data/signif_20210305_20230731_ihep_goodlist_nHit006_0.29.fits.gz.fits.gz
python3.9 plotMercator2.py ${myname} -o fullsky2.pdf --xyrange 1 355 -20 80 --interpolation --milagro -c C --nogrid

# -l 3 --tevcat-labels  -L Significance --interpolation --contours 5 -T "W43" --cat-labels-angle 100 --cat-labels-size 3 --coords C --milagro  #--squareaspect  -M 250 --origin 54 0 100 20  --xyrange -180 180 -20 20 --gamma --dpar 5 --dmer 20
#!/usr/bin/env python
################################################################################
# Plot HEALPix Map in column of FITS file on a Mercator Projection.
# Requires healpy to be installed.
################################################################################

__version__ = "$Id: plotMercator.py 40113 2017-08-09 18:36:18Z criviere $"

try:
    import argparse
    import numpy as np
    import healpy as hp
    import os
    from math import log, sqrt, cos, pi
    import pickle

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon, Wedge
    from astropy.io import ascii
    from astropy.coordinates import SkyCoord,FK5,spherical_to_cartesian,Angle

    import re

    import MapPalette
    
    import csv

    degree = np.pi / 180.
except ImportError as e:
    print(e)
    raise SystemExit

# Try importing TeVCat
try:
    import tevcat_all as TeVCat
    haveTeVCat = True
except ImportError as e:
    haveTeVCat = False
    print(e)

# Try importing the Fermi catalogs
try:
    import FGLCatalog
    import FHLCatalog
    haveFermi = True
except ImportError as e:
    haveFermi = False
    print(e)

# Try importing precession functions
try:
    import precess
    canPrecess = True
except ImportError as e:
    canPrecess = False
    print(e)

# Try importing pyfits.
# Needed to read header of HDU, where there are no maps so healpy can't read. Currently used to check map epoch 
try:
    import pyfits as pf
    havePyfits = True
except ImportError as e:
    havePyfits = False
    print(e)
    
# Try importing the Gumble distribution
try:
    from scipy.stats import gumbel_r
    haveGumbel = True
except ImportError as e:
    haveGumbel = False
    print(e)

#try:
#    import aplpy
#    haveaplpy = True
#except ImportError as e:
#    haveaplpy = False
#    print(e)

def HMS2deg(ra='', dec=''):
    RA, DEC, ra_rs, dec_ds = '', '', 1, 1
    #if dec:
    #D, M, S = [float(i) for i in dec.split()]
    #D = 5.
    dec_M = int(dec)
    #s = (dec - M) * 60
    #S = int(s)
    if dec_M >= 0.:
      dec_D = 5.
      dec_s = (dec - dec_M) * 60
      dec_S = int(dec_s)
    else:
      dec_D = 4.
      dec_M = 60 + dec_M
      dec_s = ((60 + dec) - dec_M) * 60
      dec_S = int(dec_s)
    
    if str(dec_D)[0] == '-':
      dec_ds, dec_D = -1, abs(dec_D)
    dec_deg = float(dec_D) + (float(dec_M)/60.) + (float(dec_S)/3600.)
    DEC = '{0}'.format(dec_deg*dec_ds)
  
    #if ra:
    #H, M, S = [float(i) for i in ra.split()]
    ra_H = 19.
    ra_M = int(ra)
    #print("ra is ")
    #print(ra_M)
    ra_s = (ra - ra_M) * 60
    #print("ra_s is ")
    #print(ra_s)
    ra_S = int(ra_s)
    #print("ra_S is ")
    #print(ra_S)
    if str(ra_H)[0] == '-':
      ra_rs, ra_H = -1, abs(ra_H)
    ra_deg = (float(ra_H)*15.) + (float(ra_M)/4.) + (float(ra_S)/240.)
    #print("ra_deg is ")
    #print(ra_deg)
    RA = '{0}'.format(ra_deg*ra_rs)

    return(RA, DEC)

#    if ra and dec:
#      return (RA, DEC) #tuple
#      #return RA and DEC
#    else:
#      return RA or DEC

def main():
    p = argparse.ArgumentParser(description="Plot map with Mercator projection")
    p.add_argument("fitsfile", nargs="*",
                   help="Input HEALPix FITS file. "
                   "To plot the weighted sum of multiple files, put a list of "
                   "file weight file weight pairs.")
    p.add_argument("-a", "--abthresh", default=None, type=float,
                   help="Apply an absolute (lower/upper) threshold to the map")
    p.add_argument("-c", "--coords", default="C",
                   help="C=equatorial (default), G=galactic")
    p.add_argument("-C", "--column", dest="col", default=0, type=int,
                   help="FITS column in which healpix map resides")
    p.add_argument("--file-diff", dest="diff", default="", type=str,
                   help="FITS File to take the difference from (optional) otherwise uses positional argument")
    p.add_argument("-D", "--coldiff", default=-1, type=int,
                   help="FITS column in which healpix map to be subtracted "
                   "(if any) resides")
    p.add_argument("--mjd", default=None,type=float,
                   help="MJD of the map. Will be plotted in J2000. Supersedes info in header")
    p.add_argument("-l", "--threshold", type=float, default=None,
                   help="Apply a lower threshold to the plot.")
    p.add_argument("--norm-sqdeg", dest="sqdeg", action="store_true",
                   default=False,
                   help="Normalize values to square degree, according to nSides"
                   " (makes sense only certain kinds of maps)")
    p.add_argument("--milagro", action="store_true",
                   help="Use Milagro color scale")
    p.add_argument("--contours", dest="contours", type=float, nargs='*',
                   default = None,
                   help="plot contours. If --contourmap is provided, it will be"
                        " used instead of plotted map. If no contours are"
                        " provided, it will plot confidence intervals"
                        " (1, 2, 3 sigma CI). Else, absolute sigmas are used,"
                        " e.g. --contours 1 2 3 4 [sigma] --contours")
    p.add_argument("--contourscolor", nargs='*',
                   default=None, help="Draw options for contours")
    p.add_argument("--contoursstyle",
                   default=None, help="Draw options for contours")
    p.add_argument("--contourswidth", type=float,
                   default=None, help="Draw options for contours")
    p.add_argument("--gamma", action="store_true",
                   help="Use GeV/TeV gamma-ray color scale")
    p.add_argument("-L", "--label",
                   default=None, help="Color bar label")
    p.add_argument("--cross", action="store_true",
                   help="Plot a cross at the center of the plot")
    p.add_argument("--moon", action="store_true",
                   dest="isMoon", help="is in moon centered coordinates")
    p.add_argument("-m", "--min", type=float, default=None,
                   help="Plot minimum value")
    p.add_argument("-M", "--max", type=float, default=None,
                   help="Plot maximum value")
    p.add_argument("-n", "--ncolors", type=int, default=256,
                   help="Number of contours to use in the colormap")
    p.add_argument("--nocolorbar", dest="colorbar", action="store_false",
                   help="Do not draw color bar.")
    p.add_argument("--nofill", action="store_true",
                   help="Do not draw fill plot with colors. "
                        "Useful for contours.")
    p.add_argument("-o", "--output", default=None,
                   help="Output file name (optional). "
                   "If file name ends in .fits or .fit, save as a healpix map with the pixels outside the plot area set to hp.UNSEEN.")
    p.add_argument("-s", "--scale", type=float, default=1.,
                   help="scale up or down values in map")
    p.add_argument("--sun", action="store_true",
                   dest="isSun", help="is in Sun centered coordinates")
    p.add_argument("--dpar", type=float,default=1.,
                  help="Interval in degrees between parallels")
    p.add_argument("--dmer", type=float, default=1.,
                  help="Interval in degrees between meridians.")
    p.add_argument("--nogrid", action="store_true",
                   help="Do not plot gridlines.")
    p.add_argument("--interpolation", action='store_true',
                   help="Uses bilinear interpolation in data map.")
    p.add_argument("--xsize", type=int, default=1000,
                  help="Number of X pixels, Y will be scaled acordingly.")
    p.add_argument("-t", "--ticks", nargs="+", type=float,
                   help="List of ticks for color bar, space-separated")
    p.add_argument("-T", "--title",
                   help="title of map")
    p.add_argument("--squareaspect", action='store_true',
                   help="Better Mercator X/Y ratio.")
    p.add_argument("--preliminary", action="store_true",
                   help="Add a watermark indicating the plot as preliminary")
    p.add_argument("--onlystats", action="store_true",
                   help="Exits after returning min, max and value at center. "
                   "If map is called 'flux', it will give the error if there "
                   "is a map called 'flux_error'")
    p.add_argument("--contourfile",default=None,
                   help="contour file name")

    # Mutually exclusive option to specify plot xy range OR center + WxH
    argRange = p.add_mutually_exclusive_group(required=False)
    argRange.add_argument("--xyrange", type=float, nargs=4,
                          help="Xmin Xmax Ymin Ymax plot range [degree]")
    argRange.add_argument("--origin", type=float, nargs=4,
                          help="Central (x,y) coords + width + height [degree]")

    # Plot P-Value instead of sigma, expects location and scale of
    # a right-skewed Gumble distribution
    p.add_argument("--gumbel", type=float, nargs=2,
                   help="Plot P-Value instead of significance. "
                   "Arguments: location and scale of right-skewed "
                   "Gumble distribution.")

    # Download/plotting options for Fermi catalog sources
    p.add_argument("--download-fermi", dest="dfermi", default=None,
                   help="Download Fermi FITS data from NASA and exit. Enter version number")
    p.add_argument("--fermicat", default=None,
                   help="Fermi xFGL catalog FITS file name")
    p.add_argument("--fermicat-labels", dest="fermicatLabels",
                  action="store_true",
                  help="Draw Fermi sources with labels.")
  
    # plotting options for Fermi Healpix file
    p.add_argument("--contourmap", default=None,
                   help="Healpix map from which to grab contours.")
    p.add_argument("--contourmapcoord", default="G",
                  help="Coordinate system of contourmap. C=equatorial, G=galactic (default)")

    # Download/plotting options for TeVCat sources
    p.add_argument("--download-tevcat", dest="dtevcat", action="store_true",
                   help="Download data from tevcat.uchicago.edu and exit")
    p.add_argument("--tevcat", default=None,
                   help="Draw TeVCat sources using TeVCat ASCII file")
    p.add_argument("--tevcat-labels", dest="tevcatLabels",
                   action="store_true",
                   help="Draw TeVCat sources with labels.")
    p.add_argument("--cat-labels-angle", dest="catLabelsAngle",
                   default=45, help="Oriantation of catalog labels.")
    p.add_argument("--cat-labels-size", dest="catLabelsSize",
                   default=8, help="Size of catalog labels.")

    # Highlight hotspots listed in an ASCII file
    p.add_argument("--hotspots", default=None,
                   help="Hotspot coordinates in an ASCII file")
    p.add_argument("--hotspot-labels", dest="hotspotLabels",
                   action="store_true",
                   help="Draw hotspots sources with labels.")
    # Plot 2D Gaussian fit result listed in an ASCII file from fitMap.py
    p.add_argument("--gaussfit", default=None,
                   help="2D-Gaussian fit result in an ASCII file")
                   
    # Draw ellipse
    p.add_argument("--ellipse", dest="ellipse", type=float, nargs=5,
                   help="Draw an ellipse to mark a rough contour.")
                   
    # Draw polygon
    p.add_argument("--polygon", dest="polygon", type=float, nargs="+",
                   help="Draw a polygon to mark a rough contour.")
                   
    # For SS433
    p.add_argument("--ss433", dest="ss433", action="store_true")
    
    # For SS433
    p.add_argument("--con433", dest="con_433", action="store_true")
    
    p.add_argument("--circ433", dest="circ_433", action="store_true")
                   
    # Draw polygon input file
    p.add_argument("--polygon_file", dest="polygon_file", default=None, nargs="+",
                   help="Draw a polygon using .csv input (ra dec).")
                   
    # Draw dot
    p.add_argument("--dot", dest="dotthis", type=float, nargs="+",
                   help="Draw number of dots.")
                   
    # Print a list of TS values
    p.add_argument("--TSvalue", dest="TSvalue", action="store_true")

    args = p.parse_args()

    #Sanity checks
    if (args.mjd is not None) and (not canPrecess):
        print("Missing necessary packages to make precession")
        raise SystemExit
    
    if args.nofill:
        # If we don't fill with colors, we don't plot the colorbar either
        args.colorbar = False

    # Download TeVCat
    if args.dtevcat:
        if haveTeVCat:
            print("Fetching data from tevcat.uchicago.edu.")
            tevcat = TeVCat.TeVCat()
            return None
        else:
            print("Sorry, AERIE TeVCat python module is not available.")
            raise SystemExit

    # Downlaod Fermi catalog
    if args.dfermi:
        if haveFermi:
            print ("Fetching 2FGL catalog, version %s" % args.dfermi)
            FGLCatalog.fetch_catalog(args.dfermi)
            return None
        else:
            print ("Sorry, AERIE Fermi python module is not available.")
            raise SystemExit
    
    # Start normal processing
    fitsfile = args.fitsfile
    if fitsfile == []:
        print("Please specify an FITS file to plot.")
        raise SystemExit

    # Fill 2D array
    xmin=-180.
    xmax=180.
    ymax=90
    ymin=-90

    if (args.xyrange):
        xmin, xmax, ymin, ymax = args.xyrange
        xC = (xmin+xmax) / 2.
        yC = (ymin+ymax) / 2.
    elif (args.origin):
        xC, yC, w, h = args.origin
        xmin = xC - 0.5*w
        xmax = xC + 0.5*w
        ymin = yC - 0.5*h
        ymax = yC + 0.5*h
    else:
        print("Using default zoom window. "
              "To customize the zoom window you must specify either "
              "(xyrange) or (origin and width and height).")

    if args.isMoon or args.isSun:
            xmin+=180.
            xmax+=180.
        
    # Move to range expected by healpy
    while xmin < -180:
        xmin += 360
#    while xmin > 180:
#        xmin -= 360
    while xmax < -180:
        xmax += 360
#    while xmax > 180:
#        xmax -= 360

    if xmax < xmin:
        tmax = xmax
        tmin = xmin
        xmin = tmax
        xmax = tmin

    cxmin = xmin
    cxmax = xmax
    frot =0.
    if xmax > 90. and xmin < -90.:
      frot = 180.
      cxmin = xmax - 180.
      cxmax = xmin + 180.
    
    if args.origin:
        while xC>180:
          xC-=360
    
    
    # Read in the skymap and mask out empty pixels
    skymap, skymapHeader = hp.read_map(fitsfile[0], args.col, h=True)
    if len(fitsfile) > 1:
        skymap *= float(fitsfile[1])
    # If fitsfile has more parameters, they should be "mapfile weight" pairs
    for i in range(2, len(fitsfile), 2):
        skymap2 = hp.read_map(fitsfile[i], args.col)
        skymap2 *= float(fitsfile[i+1])
        skymap += skymap2
    # remove naughty values
    skymap[np.isnan(skymap)] = hp.UNSEEN
    skymap *= args.scale
    nside1 = hp.get_nside(skymap)
    npix   = hp.nside2npix(nside1)
    
    if args.coldiff > -1:
        if os.path.isfile(args.diff):
            print("Taking difference with {0}".format(args.diff))
            skymap2 = hp.read_map(args.diff, args.coldiff)
        else:
            print("No extra file provided, using same file as input")
            skymap2 = hp.read_map(fitsfile[0], args.coldiff)
            if len(fitsfile) > 1:
                skymap2 *= float(fitsfile[1])
           
        print("Subtracting column {0} from skymap...".format(args.coldiff))
        skymap -= skymap2
        # If fitsfile has more parameters, they should be "mapfile weight" pairs
        for i in range(2, len(fitsfile), 2):
            skymap2 = hp.read_map(fitsfile[i], args.coldiff)
            skymap2 *= float(fitsfile[i+1])
            skymap -= skymap2


    if (args.gumbel):
        assert haveGumbel
        gumbel_location, gumbel_scale = args.gumbel
        gumbel = gumbel_r(loc=gumbel_location, scale=gumbel_scale)
        skymap = gumbel.logsf(skymap)/log(10)

        def inf_suppressor(x):
            return x if np.isfinite(x) else 0.

        inf_suppressor = np.vectorize(inf_suppressor)
        skymap = inf_suppressor(skymap)

    # Normalize value to square degree
    if args.sqdeg:
        pixsizestr = 4*np.pi / (12*nside1**2)
        str2sqdeg = (180/np.pi)**2
        pixsizesqdeg = pixsizestr * str2sqdeg
        skymap /= pixsizesqdeg

    # I did not find header handler, thats all I came up with...
    toFind = 'TTYPE' + str(args.col+1)
    #print type(skymapHeader), type(skymapHeader[0]), skymapHeader, toFind, dict(skymapHeader)[toFind]
    skymapName = dict(skymapHeader)[toFind]

    #Check if it is flux
    isFlux=False
    hasFluxError=False
    if skymapName=='flux':
        isFlux=True

        keyname=dict((v,n) for n,v in skymapHeader).get("flux_error")

        if keyname is not None:
            if keyname.find('TTYPE')==0 and len(keyname)==6:
                hasFluxError=True
                fluxerrmap = hp.read_map(fitsfile[0], int(keyname[5])-1)

        if not hasFluxError:
            print ("Map called 'flux_error' not present, will not print errors")

    
    isFluxError=False
    if skymapName=="flux_error":
        isFluxError=True
        
    # Find FOV
    pxls = np.arange(skymap.size)
    nZeroPix = pxls[(skymap != 0)]
    pxTh, pxPh = hp.pix2ang(nside1,pxls)
    
    # Mask outside FOV
    values = np.ma.masked_where((pxTh > pxTh[nZeroPix].max())
                                | (pxTh < pxTh[nZeroPix].min())
                                | (skymap == hp.UNSEEN)
                                | (skymap == 1e20) , skymap)

    # Plot the skymap as an image, setting up the color palette on the way
    mpl.rc("font", size=14, family="serif")
    faspect = abs(cxmax - cxmin)/abs(ymax-ymin)
    fysize = 4
    # Set up the figure frame and coordinate rotation angle
    coords = ["C","C"]
    gratcoord = "C"
    if args.coords == "G":
        coords = ["C","G"]
        gratcoord = "G"
        
    if args.mjd is not None:
        rotMap=precess.mjd2J2000ang(mjd=args.mjd,coord=coords,rot=frot)
        rotVertex=precess.mjd2J2000ang(mjd=args.mjd,coord=coords)
    elif not (args.isMoon or args.isSun):
        mjd = False;
        if havePyfits:
            hdulist = pf.open(fitsfile[0])
            header = hdulist[0].header
            if 'EPOCH' in header:
                epoch = header['EPOCH']
                if epoch=='J2000':
                    pass
                elif epoch=='current':
                    if 'STARTMJD' in header and 'STOPMJD' in header:
                        startmjd=header['STARTMJD']
                        stopmjd = header['STOPMJD']
                        if (startmjd>0 and stopmjd>0) or startmjd>stopmjd:
                            mjd = (stopmjd+startmjd)/2
                        else:
                            print ("STARTMJD/STOPMJD are not well-definned, will not attempt to precess!")
                    else:
                        print ("STARTMJD or STOPMJD not present in header, will not attempt to precess!")
                else:
                    print ("Currently EPOCH can be J2000 or current. Will not attempt to precess")
            else:
                print ("Key EPOCH not in header, will not attempt to precess!")
                     
        else:
            print ("Pyfits not available -> Can't check map epoch -> Will not attempt to precess!")
        
        if mjd:
            print ("Current epoch detected in header, will precess from MJD%g to J2000" %mjd)
            rotMap=precess.mjd2J2000ang(mjd=mjd,coord=coords,rot=frot)
            rotVertex=precess.mjd2J2000ang(mjd=mjd,coord=coords)
        else:
            rotMap=frot
            rotVertex=0
    else:
        rotMap=frot
        rotVertex=0
    
    # Get extrema
    angverts = [[xmin,ymin],[xmax,ymin],\
                [xmax,ymax],[xmin,ymax]]
    vertlist = [ ]
    cRot = hp.Rotator(coord=coords,rot=rotVertex)
    for x,y in angverts:
        ctht,cph = np.deg2rad((y,x))
        ctht = 0.5*np.pi  - ctht
        if cph < 0:
            cph = 2.*np.pi + cph
        vertlist.append(cRot.I(hp.ang2vec(ctht,cph)))

    # Get pixels in image
    imgpix = hp.query_polygon(nside1, vertlist, inclusive=True)

    seenpix = imgpix[values[imgpix]>hp.UNSEEN] 

    #if output is fits file: clip input healpy map by setting all pixels outside the plotted area to UNSEEN,
    #then save the result. 
    if args.output and ( args.output.endswith(".fits") or args.output.endswith(".fit") ):
        pix=np.ones(npix, dtype=bool)
        pix[seenpix]=False
        values[pix]=hp.UNSEEN
        print ("Saving clipped healpy map to %s" % args.output)
        hp.write_map(args.output, values, partial=True, column_names=[skymapName])
        exit()

    #Get stats
    precRot = hp.Rotator(coord=coords,rot=rotVertex)

    pixMin,pixMax=seenpix[[np.argmin(values[seenpix]), np.argmax(values[seenpix])]]
    dMin,dMax = values[[pixMin,pixMax]]
    [[thMin,thMax],[phMin,phMax]] = np.rad2deg(precRot(hp.pix2ang(nside1,[pixMin,pixMax])))

    if args.coords == 'C':
        while phMin<0:
            phMin+=360
            
        while phMax<0:
            phMax+=360
    
    th0,ph0 = precRot.I((90.-yC)*degree, xC*degree)
    pix0 = hp.ang2pix(nside1, th0, ph0)
    
    if isFlux:
        if hasFluxError:
            fluxerrMin,fluxerrMax,fluxerr0 = fluxerrmap[[pixMin,pixMax,pix0]]
        else:
            fluxerrMin,fluxerrMax,fluxerr0 = [0,0,0]

        print ("Flux units: TeV^-1 cm^-2 s^-1")
        print ("Coord units: deg")
        print ("Min:                 %11.2e +/- %11.2e (%6.2f, %5.2f)" % (dMin, fluxerrMin,
                                                                         phMin, 90-thMin))
        print ("Max:                 %11.2e +/- %11.2e (%6.2f, %5.2f)" % (dMax, fluxerrMax,
                                                                         phMax, 90-thMax))
        print ("Map value at origin: %11.2e +/- %11.2e" % (skymap[pix0], fluxerr0))
    
    elif isFluxError:
        print("Min:                 %5.4e (%6.2f, %5.2f)" % (dMin, phMin, 90-thMin))
        print("Max:                 %5.4e (%6.2f, %5.2f)" % (dMax, phMax, 90-thMax))
        print("Map value at origin: %5.4e" % skymap[pix0])    
    else:
        print("Min:                 %5.4f (%6.2f, %5.2f)" % (dMin, phMin, 90-thMin))
        print("Max:                 %5.4f (%6.2f, %5.2f)" % (dMax, phMax, 90-thMax))
        print("Map value at origin: %5.4f" % skymap[pix0])
        
    if args.onlystats:
        return 0
    
    figsize = (fysize*faspect+2, fysize+2.75)
    fig   = plt.figure(num=1, figsize=figsize)

    # Set  min/max value of map
    if args.min is not None:
        dMin = args.min
        values[(skymap<dMin) & (values > hp.UNSEEN)] = dMin
    if args.max is not None:
        dMax = args.max
        values[(skymap>dMax) & (values > hp.UNSEEN)] = dMax
    
    textcolor, colormap = MapPalette.setupDefaultColormap(args.ncolors)

    # !Print values of the input coords (origin)
    if args.origin:
        sofatinco = values[pix0]
        print ("Sigma at origin is...")
        print(sofatinco)

    # Use the Fermi/HESS/VERITAS purply-red-yellow color map
    if args.gamma:
        textcolor, colormap = MapPalette.setupGammaColormap(args.ncolors)

    # Use the Milagro color map
    if args.milagro:
        dMin = -5
        dMax = 15
        dMin = args.min if args.min != None else -5
        dMax = args.max if args.max != None else 15
        thresh = args.threshold if args.threshold != None else 2.
        textcolor, colormap = \
            MapPalette.setupMilagroColormap(dMin, dMax, thresh, args.ncolors)
        print("Milagro", dMin, dMax, thresh)
    # Use a thresholded grayscale map with colors for extreme values
    else:
        if args.threshold != None:
            textcolor, colormap = \
                MapPalette.setupThresholdColormap(dMin, dMax, args.threshold,
                                                args.ncolors)
        elif args.abthresh != None:
            textcolor, colormap = \
                MapPalette.setupAbsThresholdColormap(dMin, dMax, args.abthresh,
                                                    args.ncolors)


    if args.interpolation:
        cRot = hp.Rotator(rot=rotMap,coord=coords)
        phi   = np.linspace(np.deg2rad(xmax), np.deg2rad(xmin), args.xsize)
        if xmin < 0 and xmax > 0 and (xmax - xmin) > 180.:
            phi   = np.linspace(np.deg2rad(xmin)+2.*np.pi,np.deg2rad(xmax), args.xsize)
            phi[(phi>2.*np.pi)] -= 2.*np.pi
        theta = 0.5*np.pi  - np.linspace(np.deg2rad(ymin), np.deg2rad(ymax),
                                        int(args.xsize/faspect))
        Phi, Theta = np.meshgrid(phi, theta)
        rTheta,rPhi = cRot.I(Theta.reshape(phi.size*theta.size),\
                           Phi.reshape(phi.size*theta.size))
        rotimg = hp.get_interp_val(values, rTheta.reshape(Phi.shape),\
                                           rPhi.reshape(Theta.shape))
    else:
        tfig   = plt.figure(num=2,figsize=figsize)
        rotimg = hp.cartview(values, fig=2,coord=coords,title="",\
                            cmap=colormap, cbar=False,\
                            lonra=[cxmin,cxmax],latra=[ymin,ymax],rot=rotMap,
                            notext=True, xsize=args.xsize,
                            return_projected_map=True)
        plt.close(tfig)

    
    ax = fig.add_subplot(111)
    ax.set_aspect(1.)

    # if plotting contours
    if args.contours != None:
        if args.contourmap:
            contourmap = args.contourmap
            contourmapcoord = args.contourmapcoord
        else:
            contourmap = fitsfile[0]
            contourmapcoord = 'C'

        contourSkyMap, contourSkyMapHeader = hp.read_map(contourmap, h=True)
        fnside1 = hp.get_nside(contourSkyMap)
        ftoFind = 'TTYPE' + str(1)
        contourSkyMapName = dict(contourSkyMapHeader)[ftoFind]
#<<<<<<< .mine
#        fvalues = contourSkyMap #np.ma.masked_where(contourSkyMap == 0, contourSkyMap)
#=======
#>>>>>>> .r38901
        if args.interpolation:
            cRot = hp.Rotator(rot=rotMap,coord=[contourmapcoord,coords[-1]])
            phi   = np.linspace(np.deg2rad(xmax), np.deg2rad(xmin), args.xsize)
            if xmin < 0 and xmax > 0 and (xmax - xmin) > 180.:
                phi   = np.linspace(np.deg2rad(xmin)+2.*np.pi,np.deg2rad(xmax), args.xsize)
                phi[(phi>2.*np.pi)] -= 2.*np.pi
            theta = 0.5*np.pi - np.linspace(np.deg2rad(ymin), np.deg2rad(ymax),
                                            args.xsize/faspect)
            Phi, Theta = np.meshgrid(phi, theta)
            rTheta,rPhi = cRot.I(Theta.reshape(phi.size*theta.size),\
                                Phi.reshape(phi.size*theta.size))
            frotimg = hp.get_interp_val(contourSkyMap,rTheta.reshape(Phi.shape),\
                                        rPhi.reshape(Theta.shape))
        else:
            tfig   = plt.figure(num=3,figsize=figsize)
            frotimg = hp.cartview(contourSkyMap,
                                  fig=3,
                                  coord=[contourmapcoord, coords[-1]],
                                  title="",
                                  cmap=colormap,
                                  cbar=False,
                                  lonra=[xmin,xmax],
                                  latra=[ymin,ymax],
                                  rot=rotMap,
                                  notext=True,
                                  xsize=1000,
                                  min=dMin, max=dMax,
                                  return_projected_map=True)
            plt.close(tfig)

        if args.contours == []:
            rMaxSig2 = (contourSkyMap[imgpix].max())**2
            if rMaxSig2 < 11.83:
                print ("No spot detected with at least a 3sigma confidence contour")
                contours = [-float("inf")]
            else:
                contours = [sqrt(rMaxSig2-2.30) , sqrt(rMaxSig2-6.18) , sqrt(rMaxSig2-11.83)]
        else:
            contours = args.contours

        ccolor = args.contourscolor or 'g'
        cstyle = args.contoursstyle or '-'
        cwidth = args.contourswidth or 1.
        print ('Contours style: ',ccolor, cstyle, cwidth)

        contp = ax.contour(frotimg,
                            levels=np.sort(contours),
                            colors=ccolor,
                            linestyles = cstyle,
                            linewidths = cwidth,
                            origin='upper',
                            extent=[cxmax, cxmin, ymax, ymin])
        
        ##Extracting Contours
        if args.contourfile:
            print("hello")
            contparray = np.asarray(contp.allsegs)
            print(contparray)
            for n, level in enumerate(contp.allsegs):
                print(n)
                with open(str(args.contourfile+"%i.csv") % n,"w") as resultFile:		
#            with open("/eos/user/t/tangruiyi/sig/sig/4/wcda_cocoon_residual%i_G_Fermi.csv" % n, 'w') as resultFile:
                    wr = csv.writer(resultFile)
                    wr.writerow(["# %i sigma" % contp.levels[n]])
                    wr.writerow(["# Contour No.", "ra", "dec"])
                    for m, poly in enumerate(level):
                        for xy in poly:
                            rowout = xy.tolist()
                            rowout.insert(0, m)
                            rowout[1] += 0 # 360
                            wr.writerow(rowout)

     #   print(contparray[0].max(axis=1))
     #   print(contparray[0].min(axis=1))

        print(contp.allsegs)

        #contour labels
        
        fmt = {}
        strs=[]
        for i in range(len(args.contours)):
            strs.append('%d$\sigma$'%(args.contours[i]))
        for l, s in zip(contp.levels, strs):
            fmt[l] = s

        CLabel = plt.clabel(contp, contp.levels, use_clabeltext=True, rightside_up=True, inline=1, fmt=fmt, fontsize=10)

        for l in CLabel:
            l.set_rotation(180)

    rotimg[(rotimg>dMax)] = dMax
    rotimg[(rotimg<dMin)] = dMin
    if not args.nofill:
        imgp = ax.imshow(rotimg,extent=[cxmax, cxmin, ymax, ymin],\
                         vmin=dMin,vmax=dMax,cmap=colormap)

        imgp.get_cmap().set_under('w',alpha=0.)
        imgp.get_cmap().set_over('w',alpha=0.)

    # Draw ellipse
    if args.ellipse != None:
        x, y, w, h, t = args.ellipse
        #print("hello")
        ellip0  = Ellipse((x-360,y), width=2*w, height=2*h, angle= t, edgecolor='green',facecolor='None')
        ax.add_patch(ellip0)

    # Draw polygon
    if args.polygon != None:
        polygon_list = []
        polygon_list = args.polygon
        #print(polygon_list)
        polygon_array_2D = np.array(polygon_list).reshape(len(polygon_list)/2,2)
#        polygon_array_2D[:,0] -= 360
        #print("hello")
        poly0  = Polygon(polygon_array_2D, closed=True, edgecolor='black',facecolor='None',linestyle="--")
        ax.add_patch(poly0)

    # Draw e1, e2, e3, w1, w2
    if args.ss433:
        w1 = [287.654167-360, 5.036944]
        w2 = [287.416667-360, 5.036944]
        e1 = [288.404167-360, 4.930000]
        e2 = [288.583333-360, 4.906944]
        e3 = [289.016667-360, 4.836944]
        w1_g = [39.603209, -1.952345]
        w2_g = [39.494128, -1.742381]
        e1_g = [39.853438, -2.664511]
        e2_g = [39.915594, -2.833454]
        e3_g = [40.053564, -3.248594]
        
        ss433_centre_ra = 287.9565-360
        ss433_centre_dec = 4.982667
        ss433_centre_ra_g = 39.694031
        ss433_centre_dec_g = -2.244607
        
#        for label, roi in zip(["w1", "w2", "e1", "e2", "e3"], [w1, w2, e1, e2, e3]):
        for label, roi in zip(["w1", "w2", "e1", "e2", "e3"], [w1_g, w2_g, e1_g, e2_g, e3_g]):
            ra, dec = roi
            ax.scatter(ra, dec,color="black",marker="x",zorder=20)
            ax.text(ra-0.1,dec,label,size=10)
#            theta = (90.-dec) * Degree
#            phi = ra * Degree
#            hp.projplot(theta, phi, 'ko', mfc='none')
#            hp.projtext(theta-0.1*Degree, phi+0.1*Degree, label, fontsize=10, color='k')
        ax.scatter(ss433_centre_ra_g, ss433_centre_dec_g, color="red", marker="x", zorder=20)


    # Draw contours of ss433:
    if args.con_433:
        f = ascii.read("/Users/Chang/Documents/HAWC/aerie/trunk/build/SS433-W50/ss433_rosat_goodall_xray.csv")
        ra2000, dec2000 = f['col1'], f['col2']
#        ra1950, dec1950 = f['col1'], f['col2']

        Degree = np.pi/180.

#        x = np.cos(ra1950*Degree) * np.cos(dec1950*Degree)
#        y = np.sin(ra1950*Degree) * np.cos(dec1950*Degree)
#        z = np.sin(dec1950*Degree)

        # apply the precession matrix
#        x2 = 0.999925708 * x - 0.0111789372 * y - 0.0048590035 * z
#        y2 = 0.0111789372 * x + 0.9999375134 * y - 0.0000271626 * z
#        z2 = 0.0048590036 * x - 0.0000271579 * y + 0.9999881946 * z

        # convert the new direction cosines to RA, DEC
#        ra2000 = np.arctan2(y2, x2)/Degree + 360.
#        dec2000 = np.arcsin(z2)/Degree

        lat_list=[]
        lon_list=[]
        
        n=0
        
        for (x,y) in zip(ra2000,dec2000):
        
            sc = SkyCoord(ra=x, dec=y, unit='deg', frame='icrs')
            lat_list.append(sc.galactic.b.degree)
            lon_list.append(sc.galactic.l.degree)
            n+=1
            print(n)
#        ax.plot(lon_list,lat_list,linewidth=0.5,color="black")
        ax.plot(lon_list,lat_list,'k,',markersize=0.3,alpha=0.25)

    if args.circ_433:
        wd2 = Wedge((40.527362, -0.795364), 9.5, 0, 360, fc='k', alpha=0.5, width=7.0, edgecolor='none')#lw=0)
        circle2 = plt.Circle((40.527362, -0.795364), 2.5, color='k', fill=False, lw=0)
        wd1 = Wedge((40.527362, -0.795364), 2.502, 0, 180, fc='k', alpha=0.5, edgecolor='none')#lw=0)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(wd1)
        plt.gcf().gca().add_artist(wd2)

    # Draw polygon with a file
    if args.polygon_file != None:
        polygon_list = []
        for j in args.polygon_file:
            polygon_ra = []
            polygon_dec = []
            deg_ra_list = []
            deg_dec_list = []
            readingfile = ascii.read(j)
            for u in range(len(readingfile)):
                polygon_ra.append(readingfile['col1'][u])
                polygon_dec.append(readingfile['col2'][u])
            for (x,y) in zip(polygon_ra,polygon_dec):
                deg_ra, deg_dec = HMS2deg(ra=x, dec=y)
                float_deg_ra = float(deg_ra)
                float_deg_dec = float(deg_dec)
                float_deg_ra -= 360.
                sc = SkyCoord(ra=float_deg_ra, dec=float_deg_dec, unit='deg', frame=FK5, equinox='J1950.0')
                deg_ra_list.append(sc.transform_to(FK5(equinox='J2000')).ra.degree)
                deg_dec_list.append(sc.transform_to(FK5(equinox='J2000')).dec.degree)
            print(deg_dec_list)
            ax.plot(np.asarray(deg_ra_list)-360.,deg_dec_list,linewidth=2.0,color="black")

    # Draw dots
    if args.dotthis != None:
        polygon_list = args.dotthis
        dot_ra_list=polygon_list[::2]
        dot_ra_list=np.asarray(dot_ra_list)-360
        dot_dec_list=polygon_list[1::2]
        #print(polygon_list)
        #polygon_array_2D = np.array(polygon_list).reshape(len(polygon_list)/2,2)
        #polygon_array_2D[:,0] -= 360
        #print("hello")
        #poly0  = Polygon(polygon_array_2D, closed=True, edgecolor='black',facecolor='None')
        ax.scatter(dot_ra_list,dot_dec_list, color="red")
        #print(dot_ra_list)
        #print(dot_dec_list)

    # Draw grid lines
    xts = np.arange(np.floor(xmin), np.ceil(xmax+args.dmer), args.dmer)[1:-1]
    xtlbs = [ '%g'%(xt+360) if args.coords=='C' and xt<0 else '%g'%xt for xt in xts]
    if xmin < 0. and xmax > 0. and (xmax-xmin) > 180.:
        xts = np.arange(np.floor(cxmin), np.ceil(cxmax+args.dmer), args.dmer)[1:-1]
        xtlbs = [   ]
        for xt in xts:
            cval = xt - 180.
            if xt < 0:
                cval = 180.+xt
            if cval == -180:
                cval = 180
            if args.isMoon or args.isSun:
                cval -= 180.
                if cval < -180.:
                    cval+=360.
            elif args.coords=='C' and cval<0:
                cval+=360
            
            xtlbs.append('%g'%(cval))
    yts = np.arange(np.floor(ymin), np.ceil(ymax+args.dpar), args.dpar)[1:-1]

    if args.nogrid == False:
        ax.grid(color=textcolor)
        ax.xaxis.set_ticks(xts)
        ax.xaxis.set_ticklabels(xtlbs)
        ax.yaxis.set_ticks(yts)

    if args.preliminary:
        plt.text((xmin+xmax)/2., ymin+0.85*(ymax-ymin),
                 "PRELIMINARY",
                 color=textcolor,
                 alpha=0.8,
                 fontdict={"family":"sans-serif", "weight":"bold", "size":28},
                 horizontalalignment='center')

    # If TeVCat data are available, plot them
    if args.tevcat or args.tevcatLabels:
        if haveTeVCat:
            try:
                if args.tevcat:
                    tevcat = TeVCat.TeVCat(args.tevcat)
                elif args.tevcatLabels:
#                    tevcat = TeVCat.TeVCat(args.tevcatLabels)
                     tevcat = TeVCat.TeVCat()
            except IOError as e:
                print(e)
                print("Downloading data from tevcat.uchicago.edu")
                tevcat = TeVCat.TeVCat()
            except:
                print("Why caught here?")
                print("Downloading data from tevcat.uchicago.edu")
                tevcat = TeVCat.TeVCat()
            xa=[]
            ya=[]
            assoca=[]
            cRot = hp.Rotator(coord=["C",coords[-1]])
            tnside=512
            fpix = np.zeros(hp.nside2npix(tnside))
            sources = tevcat.getSources()
            for i in range(0,len(sources)):
                sourceFK5 = sources[i].getFK5()
                ra=sourceFK5.ra.degree/180.*np.pi
                dec=sourceFK5.dec.degree/180.*np.pi
                assoc = sources[i].getCanonicalName()
#                print(sourceFK5.ra.degree,sourceFK5.dec.degree,assoc)
#                cut = (assoc != 'Crab Pulsar')
#                ra = ra[cut]
#                dec = dec[cut]
#                assoc = assoc[cut]
#                cpix = hp.ang2pix(tnside,np.pi*0.5 - dec,ra)
#                slpx = []
#                for sx,px in enumerate(cpix):
#                    fpix[px] += 1
#                    if fpix[px] != 1:
#                        print("%s is a duplicate" % (assoc[sx]))
#                        slpx.append(sx)
                
#                ra = np.delete(ra,slpx)
#                dec = np.delete(dec,slpx)
#                assoc = np.delete(assoc,slpx)
                y, x = cRot(np.pi*0.5 - dec,ra)
                x = np.rad2deg(x) + frot
                if x > 180:
                    x-=360
                y = 90. - np.rad2deg(y)
                if (x < xmin) or (x > xmax) or (y < ymin) or (y>ymax):
                    continue
#                cut = (x > xmin) & (x < xmax) & (y>ymin) & (y<ymax)
#                x = x[cut]
#                y = y[cut]
#                assoc = assoc[cut]
                textcolor = "black"
                if assoc in assoca:
                    continue
                xa.append(x)			
                ya.append(y)			
                assoca.append(assoc)		
            sources_tmp = list(zip(xa,ya,assoca))
            sources_tmp.sort(key=lambda source: source[0])
            ax.scatter(xa,ya, color=textcolor, facecolors="none", marker="s")
#            for r, d, s in zip(xa, ya, assoca):
            i=0
            dt=0
            rt=0
            pre_rt1=0
            pre_rt2=0
            ymid=(ymax+ymin)/2.
            dr=(xmax-xmin)/(len(sources_tmp)/2.+1)/2
            for r, d, s in sources_tmp:
                    print(r, d, s) 
                    
                    if d>ymid:
#                        if np.abs(r-pre_rt1) <dr:
                        if np.abs(r)-np.abs(pre_rt1) <dr:
                            rt=pre_rt1+dr
                        else :
                            rt=r
                        pre_rt1=rt	
                        dt=(ymax+ymid)/2.
                        Rotation=45
                        Va='bottom'
                    else:
#                        if np.abs(r-pre_rt2) <dr:
                        if np.abs(r)-np.abs(pre_rt2) <dr:
                            rt=pre_rt2+dr
                        else :
                            rt=r
                        pre_rt2=rt	
                        dt=(ymid+ymin)/2.
                        Rotation=315
                        Va='top'
                    i+=1
                    ax.text(rt,dt, s+'', color=textcolor,
                            rotation=Rotation, #args.catLabelsAngle,
                            va=Va,
                            fontdict={'family': 'sans-serif',
                                      'size': args.catLabelsSize,
                                      'weight': 'bold'})
                    ax.plot([r,rt],[d,dt],'k--')
        else:
            print("Sorry, TeVCat could not be loaded.")

    # If Fermi data are available, plot them
    if args.fermicat:
        if haveFermi:
            fcat = None
            try:
                fcat = FGLCatalog.FGLCatalog(args.fermicat)
                aflag = fcat.GetAnalysisFlags()
                acut = (aflag == 0)                     # cut analysis errors

                flux = fcat.GetFlux1000()               # 1-100 GeV flux
                dflx = fcat.GetFlux1000Error()          # flux uncertainty
                fcut = dflx/flux < 0.5                  # cut poorly measured srcs

                cuts = np.logical_and(acut, fcut)
                print('Using FGL')
            except:
                try:
                    fcat = FHLCatalog.FHLCatalog(args.fermicat)
                    cuts = (fcat.GetFlux() > 0.)        # Dummy cut
                    print('Using FHL')
                except:
                    print('Fermi catalog could not be loaded!')
        if fcat != None:
            # Don't show TeV associations if plotting from TeVCat
            if args.tevcat or args.tevcatLabels:
                tcfg = fcat.GetTeVCatFlag()
                tcut = (tcfg == "N") | (tcfg == "C")
                cuts = np.logical_and(cuts, tcut)
            ra = fcat.GetRA()[cuts]
            dec = fcat.GetDec()[cuts]
            assoc = fcat.GetSourceName()[cuts]
            catnms = fcat.GetCatalogName()[cuts]
            for i in xrange(len(assoc)):
                if assoc[i] == '':
                    assoc[i] = catnms[i]

            cRot = hp.Rotator(coord=["C",coords[-1]])
            y, x = cRot(np.pi*0.5 - dec,ra)
            x = np.rad2deg(x) + frot
            x[(x>180.)] -= 360.
            y = 90. - np.rad2deg(y)
            cut = (x > xmin) & (x < xmax) & (y>ymin) & (y<ymax)
            x = x[cut]
            ax.scatter(x,y, color=textcolor, facecolors="none", marker="o")

            if args.fermicatLabels:
                for r, d, s in zip(x, y, assoc):
                    ax.text(r, d, s, color=textcolor,
                            rotation=args.catLabelsAngle,
                            fontdict={'family': 'sans-serif',
                                      'size': args.catLabelsSize,
                                      'weight': 'bold'})
        else:
            print("Sorry, the Fermi xFGL catalog could not be loaded.")

    # If a hotspot list is given, plot it
    if args.hotspots:
        fhot = open(args.hotspots, "r")
        ra = []
        dec = []
        assoc = []
        for line in fhot:
            if line.startswith("#"):
                continue
            larr = line.strip().split()
            ra.append(float(larr[1]))
            dec.append(float(larr[2]))
            assoc.append(larr[4])

        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)
        assoc = np.array(assoc)
        cRot = hp.Rotator(coord=["C",coords[-1]])
        y, x = cRot(np.pi*0.5 - dec,ra)
        x = np.rad2deg(x) + frot
        x[(x>180.)] -= 360.
        y = 90. - np.rad2deg(y)
        cut = (x > xmin) & (x < xmax) & (y>ymin) & (y<ymax)
        x = x[cut]
        y = y[cut]
        assoc = assoc[cut]
        ax.scatter(x,y, color=textcolor, facecolors="none", marker="o")
        if args.hotspotLabels:
            for r, d, s in zip(x, y, assoc):
                print(r, d, s)
                ax.text(r,d, s+'   .',
                        color=textcolor,
                        rotation=90,
                        fontdict={'family': 'sans-serif',
                                  'size': 8,
                                  'weight': 'bold'})

    # If a gaussian fit file is given, plot it
    if args.gaussfit:
        gfit = open(args.gaussfit, "r")
        gfit.next()
        gfit.next()
        ra,raErr     = [float(i) for i in gfit.next().strip().split()]
        dec,decErr   = [float(i) for i in gfit.next().strip().split()]
        raW,raWErr   = [float(i) for i in gfit.next().strip().split()]
        decW,decWErr = [float(i) for i in gfit.next().strip().split()]
        gfit.close()

        mrot = -180. if args.isMoon or args.isSun else 0.
      
        cRot = hp.Rotator(coord=["C",coords[-1]])
        y, x = cRot(np.pi*0.5 - np.deg2rad(dec),np.deg2rad(ra+mrot))
        x = np.rad2deg(x) + frot
        x = x-360. if x>180. else x
        y = 90. - np.rad2deg(y)
        ellip0  = Ellipse((x,y), width=2*raW, height=2*decW, edgecolor='black',facecolor='None')
        ax.add_patch(ellip0)
        ax.scatter(x,y, s=20, color='black',facecolors="black",marker="o")

        print(x, y, xmin, xmax, ymin, ymax)
        ax.text(x-1, y+1, 
              "%s=%.02f$\pm$%.02f\n$\Delta\delta$=%.02f$\pm$%.02f\n%s=%.02f$\pm$%.02f\n$\sigma_\delta$=%.02f$\pm$%.02f"
                  % (r"$\Delta\alpha$",ra,raErr,dec,decErr,r"$\sigma_\alpha$",raW,raWErr,decW,decWErr),
              color=textcolor,
              rotation=0,
              fontdict={"size":12})



    # Set up the color bar
    # Setup color tick marks
    if args.colorbar:
        cticks = args.ticks
        if args.ticks == None:
            if (dMax-dMin) > 3:
                clrmin = np.floor(dMin)
                clrmax = np.ceil(dMax)
                ctspc = np.round((clrmax - clrmin)/10.)
                if ctspc == 0.:
                    ctspc = 1.
                if clrmin < 0 and clrmax > 0:
                    cticks = -np.arange(0,-clrmin,ctspc)
                    cticks = np.sort(np.unique(np.append(cticks,np.arange(0,clrmax,ctspc))))
                else:
                    cticks = np.arange(clrmin,clrmax+ctspc,ctspc)
            else:
                cticks = None

        cb = fig.colorbar(imgp, orientation="horizontal",
                          shrink=0.85,
                          fraction=0.1,
                          #aspect=25,
                          pad=0.1,
                          ax=ax,
                          ticks=cticks)

    if args.label:
        skymapName = args.label
    else:
        if re.match("significance", skymapName):
            skymapName=r"significance [$\sigma$]"
    if args.gumbel:
        skymapName="log10(P-Value)"

    if args.colorbar:
        cb.set_label(skymapName)

    xlabel = r"$\alpha$ [$^\circ$]"
    ylabel = r"$\delta$ [$^\circ$]"


    # Set up the color bar and tick axis
    if args.coords == "G":
        xlabel = r"$l$ [$^\circ$]"
        ylabel = r"$b$ [$^\circ$]"
    if args.isMoon or args.isSun:
        xlabel = r"$\Delta\alpha$"
        ylabel = r"$\Delta\delta$"

    # X axis label
    ax.set_xlabel(xlabel, color='k')
    # Y axis label
    ax.set_ylabel(ylabel, color='k')

    # Title
    if args.title != None:
        ax.set_title(r"{0}".format(args.title.replace("\\n","\n")),color='k')

    if args.cross:
        # Indicate the center of the plot with a thick black cross
        ax.scatter(xC, yC, s=20**2,marker="+", facecolor="#000000", color="#000000")

    ax.set_ylim(ymin,ymax)
    ax.set_xlim(cxmax,cxmin)
    if args.squareaspect:
        plt.axes().set_aspect(1./cos((ymax+ymin)/2*pi/180))

    if args.TSvalue:
#        OLD METHOD!
#        np.set_printoptions(threshold='nan')
#        print("TS are:")
#        valueslist = []
##        for i in np.arange(len(values)):
##            if values[seenpix][i] < 0:
##                valueslist.append(-1*values[seenpix][i]**2)
##            else:
##                valueslist.append(values[seenpix][i]**2)
#
##        print(valueslist)
##        print(len(valueslist))
#
#        xmin2,ymin2,xmax2,ymax2 = 38.027362, -3.295364, 43.027362, -0.795364 #for ss433
##        xmin2,ymin2,xmax2,ymax2 = 319.969718, 65.663945, 324.969718, 68.163945
##        xmin2,ymin2,xmax2,ymax2 = 267.702941, 63.450824, 272.702941, 65.950824
##        xmin2,ymin2,xmax2,ymax2 = 115.385877, -56.871604, 120.385877, -53.371604
#
##        xmin2,ymin2,xmax2,ymax2 = 72.140548, -46.337736, 77.140548, -43.837736
##        xmin2,ymin2,xmax2,ymax2 = 55.657384, -31.438959, 60.657384, -28.938959
##        xmin2,ymin2,xmax2,ymax2 = 14.443661, 38.177016, 19.443661, 40.677016
##        xmin2,ymin2,xmax2,ymax2 = 356.630082, 54.373710, 1.630082, 56.873710
##        xmin2,ymin2,xmax2,ymax2 = 238.056998, 50.071864, 243.056998, 52.571864
##        xmin2,ymin2,xmax2,ymax2 = 222.887201, 33.329516, 227.887201, 35.829516
##        xmin2,ymin2,xmax2,ymax2 = 162.557438, -49.889259, 167.557438, -47.389259
##        xmin2,ymin2,xmax2,ymax2 = 44.5, -3.295364, 49.5, -0.795364 #for hao GDE
#        angverts2 = [[xmin2,ymin2],[xmax2,ymin2],\
#                [xmax2,ymax2],[xmin2,ymax2]]
#        vertlist2 = []
#        cRot2 = hp.Rotator(coord=["C","G"],rot=0)
#        for x,y in angverts2:
#            ctht,cph = np.deg2rad((y,x))
#            ctht = 0.5*np.pi  - ctht
#            if cph < 0:
#                cph = 2.*np.pi + cph
#            vertlist2.append(cRot2.I(hp.ang2vec(ctht,cph)))
#
##        disc_cent = [40.527362, -0.795364]
#        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(287.05)) #for ss433
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(200.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(180.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(10.05))
#
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(340.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(320.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(240.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(220.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(160.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(140.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(6.39), np.deg2rad(40.05))
##        disc_center = spherical_to_cartesian(1, np.deg2rad(12.121361), np.deg2rad(290.097444)) #for hao GDE
#        centradius = np.deg2rad(2.5)
#
#        rectangle_roi_pix = hp.query_polygon(nside1, vertlist2, inclusive=True)
#        disc_roi_pix = hp.query_disc(nside1, disc_center, centradius, inclusive=True)
#
#        #find overlap
##        print(set(rectangle_roi_pix).intersection(disc_roi_pix))
##        print(list(values[list(set(rectangle_roi_pix).intersection(disc_roi_pix))]))
#        print(len(set(rectangle_roi_pix).intersection(disc_roi_pix)))
#        print(len(rectangle_roi_pix))
#        print(len(disc_roi_pix))
##        print(set(rectangle_roi_pix).intersection(disc_roi_pix))
#
#        setofpix = set(rectangle_roi_pix).intersection(disc_roi_pix)
#        thepixels_coords = np.rad2deg(precRot(hp.pix2ang(nside1,list(setofpix))))

#        NEW METHOD!
        thevalues = []
        thepixels = []
        numbercount = 0

        deg = np.pi/180.
        ra = 340.05
        dec = 6.39
        radius = 2.5
        disc_roi_pix = hp.query_disc(1024,hp.ang2vec((90-dec)*deg,ra*deg),radius*deg)

        coord = SkyCoord(ra=ra,dec=dec,frame='icrs',unit="deg")
        b0 = coord.transform_to('galactic').b.value
        
        print(len(disc_roi_pix))
        
        for p in disc_roi_pix:
            dec,ra = hp.pix2ang(1024,p)
            ra = ra/deg
            dec = 90. - dec/deg
            coord = SkyCoord(ra=ra,dec=dec,frame='icrs',unit="deg")
            b = coord.transform_to('galactic').b.value
            l = coord.transform_to('galactic').l.value
            if b<=b0:
                thevalues.append(values[p])
                thepixels.append(p)
            
            numbercount = numbercount + 1
            print(numbercount)

        print(len(thepixels))

        thepixels_coords = np.rad2deg(precRot(hp.pix2ang(1024,thepixels)))
        ax.scatter(thepixels_coords[1],90.-np.asarray(thepixels_coords[0]),color='black',facecolors="black",marker="o",alpha=0.1)

        with open("/Users/Chang/Documents/HAWC/aerie/trunk/build/new_maptrees/source_outputs/30month_stuff/test_ss433/Res_1d_hist_340p05.txt", "a") as output:
            output.write(str(thevalues))
            #output.write(str(list(values[list(set(rectangle_roi_pix).intersection(disc_roi_pix))])))

    # Either output the image to a file and quit or plot it in a window
    if args.output:
        if not args.nofill:
            fig.savefig(args.output, dpi=300)
        else:
            fig.savefig(args.output, dpi=300, transparent=True)
    else:
        plt.show()

if __name__ == "__main__":
    main()

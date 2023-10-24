from astropy.io import fits
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
try:
    import tevcat as TeVCat
    haveTeVCat = True
except ImportError as e:
    haveTeVCat = False
    print(e)
import MapPalette
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates import FK5
import threeML
import gammapy as gp
from astroquery.ned import Ned
from astroquery.vizier import Vizier
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.coordinates import Angle

from tqdm import tqdm

try:
    tevcat = TeVCat.TeVCat()
except IOError as e:
    print(e)
    print("Downloading data from tevcat.uchicago.edu")
    tevcat = TeVCat.TeVCat()
except:
    print("Why caught here?")
    print("Downloading data from tevcat.uchicago.edu")
    tevcat = TeVCat.TeVCat()

def Drawgascontour():
        from matplotlib.colors import Normalize
        with fits.open('../../data/J0248_co_-55--30_all.fits') as hdul:
                # 输出文件信息
                qtsj = hdul[0].data
                hdul[0].header
        hdul[0].header
        # 获取坐标信息
        crval1 = hdul[0].header['CRVAL1']
        crval2 = hdul[0].header['CRVAL2']
        cdelt1 = hdul[0].header['CDELT1']
        cdelt2 = hdul[0].header['CDELT2']
        crpix1 = hdul[0].header['CRPIX1']
        crpix2 = hdul[0].header['CRPIX2']

        # 计算x轴和y轴的坐标范围
        naxis1, naxis2 = qtsj.shape
        xmin = (1 - crpix1) * cdelt1 + crval1
        xmax = (naxis1 - crpix1) * cdelt1 + crval1
        ymin = (1 - crpix2) * cdelt2 + crval2
        ymax = (naxis2 - crpix2) * cdelt2 + crval2
        glon = np.linspace(xmin,xmax,naxis1)
        glat = np.linspace(ymin,ymax,naxis2)
        Glon,Glat = np.meshgrid(glon,glat)
        galactic_coord = SkyCoord(Glon* u.degree, Glat* u.degree, frame='galactic')
        j2000_coords = galactic_coord.transform_to('fk5')
        Glon,Glat = j2000_coords.ra,j2000_coords.dec
        plt.contour(Glon,Glat,qtsj,5,cmap="Greys",levels=np.array([0.2,0.3,0.5,0.7,1,1.5,2,3,4])*1e22,norm=Normalize(vmin=0.2e22,vmax=1e22),alpha=0.7)

def GetTeVcat(xmin,xmax,ymin,ymax):
    xa=[]
    ya=[]
    assoca=[]
    sources = tevcat.getSources()
    for i in range(0,len(sources)):
        sourceFK5 = sources[i].getFK5()
        ras=sourceFK5.ra.degree
        decs=sourceFK5.dec.degree
        assoc = sources[i].getCanonicalName()
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetFermicat(xmin,xmax,ymin,ymax,cat="3FHL"):
    # 打开FITS文件
    header = []
    data = []
    if cat == "4FGL":
        file = '../../data/gll_psc_v32.fit'
    elif cat == "3FHL":
        file = '../../data/gll_psch_v13.fit'
    with fits.open(file) as hdul:
        # 输出文件信息
        # hdul.info()

        # 输出每个HDU的信息
        for i, hdu in enumerate(hdul):
            # print(f'HDU {i}:')
            header.append(hdu.header)
            data.append(hdu.data)
    
    xa=[]
    ya=[]
    assoca=[]
    sources = tevcat.getSources()
    for i in range(0,len(data[1])):
        if cat == "3FHL":
            ras=data[1][i][1]
            decs=data[1][i][2]
        elif cat == "4FGL":
            ras=data[1][i][2]
            decs=data[1][i][3]
        assoc = data[1][i][0]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetPSRcat(xmin,xmax,ymin,ymax):
    # 设置max_catalogs参数为100
    Vizier.ROW_LIMIT = -1  # 无限制
    Vizier.MAX_RESULTS = -1 # 无限制
    Vizier.TIMEOUT = 180 # 设置超时时间
    xa=[]
    ya=[]
    assoca=[]
    # 获取ATNF pulsar目录的数据
    atnf_catalog = Vizier.get_catalogs('B/psr')[0]
    for i in range(0,len(atnf_catalog)):
        try:
            ras=float(Angle(atnf_catalog[i][2], unit='hourangle').degree)
            decs=float(Angle(atnf_catalog[i][3], unit='degree').value)
        except:
            continue
        assoc = atnf_catalog[i][0]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetQSOcat(xmin,xmax,ymin,ymax):
    from astroquery.ned import Ned
    Ned.ROW_LIMIT = -1  # 无限制
    Ned.MAX_RESULTS = -1 # 无限制
    Ned.TIMEOUT = 360 # 设置超时时间
    agn_table = Ned.query_region(
        coordinates=SkyCoord((xmin+xmax)/2, (ymin+ymax)/2,unit=(u.deg, u.deg),frame='fk5'),
        radius= (ymax-ymin)/2 * u.degree,
        equinox="J2000.0",
        get_query_payload=False
    )
    xa=[]
    ya=[]
    assoca=[]
    QSO_table = agn_table[agn_table["Type"]=="QSO"]
    for i in range(0,len(QSO_table)):
        try:
            ras=float(Angle(QSO_table[i][2], unit='degree').value)
            decs=float(Angle(QSO_table[i][3], unit='degree').value)
        except:
            continue
        assoc = QSO_table[i][1]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetSimbad(xmin,xmax,ymin,ymax,stype=None,criteria=None):
    from astroquery.simbad import Simbad
    Simbad.ROW_LIMIT = 0  # 无限制
    Simbad.MAX_RESULTS = 0 # 无限制
    Simbad.TIMEOUT = 360 # 设置超时时间
    ra1 = (xmin+xmax)/2
    dec1 = (ymin+ymax)/2
    radx = (xmax-xmin)
    radx*np.cos(dec1/180*np.pi)
    rady = (ymax-ymin)
    fh = ""
    if dec1>0:
        fh = "+"
    if not criteria:
        criteria = f"region(box, {int(ra1/15)} {int(ra1%15/15*60)} {fh}{int(dec1)} {int((dec1-int(dec1))*60)}, {radx}d {rady}d)"
    if stype:
        Simbad_table =  Simbad.query_criteria(criteria, maintype=stype)
    else:
        Simbad_table =  Simbad.query_criteria(criteria)
    xa=[]
    ya=[]
    assoca=[]
    for i in range(0,len(Simbad_table)):
        try:
            ras=float(Angle(Simbad_table[i][1], unit='hourangle').degree)
            decs=float(Angle(Simbad_table[i][2], unit='degree').value)
        except:
            continue
        assoc = Simbad_table[i][0]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        # if "NGC" not in assoc and "M " not in assoc:
        #     continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetSNRcat(xmin,xmax,ymin,ymax):
    # 设置max_catalogs参数为100
    Vizier.ROW_LIMIT = -1  # 无限制
    Vizier.MAX_RESULTS = -1 # 无限制
    Vizier.TIMEOUT = 180 # 设置超时时间
    xa=[]
    ya=[]
    assoca=[]
    # 获取ATNF pulsar目录的数据
    green_catalog = Vizier.get_catalogs('VII/278')[0]
    for i in range(0,len(green_catalog)):
        try:
            ras=float(Angle(green_catalog[i][1], unit='hourangle').degree)
            decs=float(Angle(green_catalog[i][2], unit='degree').value)
        except:
            continue
        assoc = green_catalog[i][0]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetAGNcat(xmin,xmax,ymin,ymax):
    # 设置max_catalogs参数为100
    Vizier.ROW_LIMIT = -1  # 无限制
    Vizier.MAX_RESULTS = -1 # 无限制
    Vizier.TIMEOUT = 180 # 设置超时时间
    xa=[]
    ya=[]
    assoca=[]
    # 获取ATNF pulsar目录的数据
    agn_catalog = Vizier.get_catalogs('J/ApJ/892/105')[0]
    for i in range(0,len(agn_catalog)):
        try:
            ras=float(Angle(agn_catalog[i][2], unit='degree').value)
            decs=float(Angle(agn_catalog[i][3], unit='degree').value)
        except:
            continue
        assoc = agn_catalog[i][1]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)
        ya.append(decs)			
        assoca.append(assoc)
    sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def Drawcat(xmin,xmax,ymin,ymax,cat="TeVCat",mark="s",c="black",angle=45, fontsize=7, label="Cat",textlabel=False, stype=None, criteria=None):
    """Draw catalog.

        Args:
            xmin: min of ra.
            cat: name of the catalog: TeVCat/3FHL/4FGL/PSR/SNR/AGN/QSO/Simbad
            mark: marker
            stype: source type in Simbad
            criteria: Simbad criteria
        Returns:
            fig
    """
    if cat=="TeVCat":
        sources_tmp = GetTeVcat(xmin,xmax,ymin,ymax)
    elif cat=="3FHL":
        sources_tmp = GetFermicat(xmin,xmax,ymin,ymax)
    elif cat=="4FGL":
        sources_tmp = GetFermicat(xmin,xmax,ymin,ymax,"4FGL")
    elif cat=="PSR":
        sources_tmp = GetPSRcat(xmin,xmax,ymin,ymax)
    elif cat=="SNR":
        sources_tmp = GetSNRcat(xmin,xmax,ymin,ymax)
    elif cat=="AGN":
        sources_tmp = GetAGNcat(xmin,xmax,ymin,ymax)
    elif cat=="QSO":
        sources_tmp = GetQSOcat(xmin,xmax,ymin,ymax)
    elif cat=="Simbad":
        if stype:
            sources_tmp = GetSimbad(xmin,xmax,ymin,ymax, stype, criteria)
        else: 
            sources_tmp = GetSimbad(xmin,xmax,ymin,ymax, criteria=criteria)

    ymid = np.mean([ymin,ymax])
    i=0
    dt=0
    rt=0
    pre_rt1=0
    pre_rt2=0
    dr=(xmax-xmin)/2/(len(sources_tmp)/2.+1)/2
    iflabel=1
    counts=1
    for r, d, s in sources_tmp:
            print(cat+": ",counts,r, d, s)
            counts+=1
            
            if d>ymid:
    #                        if np.abs(r-pre_rt1) <dr:
                if np.abs(r)-np.abs(pre_rt1) <dr:
                    rt=pre_rt1+dr
                else :
                    rt=r
                pre_rt1=rt	
                dt=(ymax+ymid)/2.
                Rotation=angle
                Va='bottom'
            else:
    #                        if np.abs(r-pre_rt2) <dr:
                if np.abs(r)-np.abs(pre_rt2) <dr:
                    rt=pre_rt2+dr
                else :
                    rt=r
                pre_rt2=rt	
                dt=(ymid+ymin)/2.
                Rotation=360-angle
                Va='top'
            i+=1
            if textlabel:
                plt.text(rt,dt, s+'', color=c,
                        rotation=Rotation, #catLabelsAngle,
                        va=Va,
                        fontdict={'family': 'sans-serif',
                                    'size': fontsize,
                                    'weight': 'bold'})
                plt.plot([r,rt],[d,dt],'k--',c=c)
            if iflabel==1:
                plt.scatter(r,d, color=c, facecolors="none", 
                marker=mark,label=label)
            else:
                plt.scatter(r,d, color=c, facecolors="none", 
                marker=mark)
            iflabel+=1

def interpimg(hp_map,xmin,xmax,ymin,ymax,xsize):
    faspect = abs(xmax - xmin)/abs(ymax-ymin)
    phi   = np.linspace(xmin, xmax, xsize)
    theta = np.linspace(ymin, ymax, int(xsize/faspect))
    Phi, Theta = np.meshgrid(phi, theta)
    rotimg = hp.get_interp_val(hp_map, Phi,Theta,lonlat=True) #,nest=True
    # plt.contourf(Phi,Theta,rotimg)
    # plt.imshow(rotimg, origin="lower",extent=[xmin,xmax,ymin,ymax])
    # plt.colorbar()
    return rotimg

def hpDraw(region_name, Modelname, map, ra, dec, rad=5, radx=5,rady=2.5,contours=[3,5],colorlabel="Excess",color="Fermi", plotres=False, save=False, cat={"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"]}):
    """Draw healpixmap.

        Args:
            cat: catalog to draw. such as {"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"]}, first in [1,"s"] is about if add a label?
                "o" is the marker you choose.
        Returns:
            fig
    """
    ra_crab, dec_crab =  83.63,22.02
    ra_j0248, dec_j0248 = 42.19,60.35
    # ra,dec = ra_crab,dec_crab
    # ra,dec = ra_j0248, dec_j0248
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='mollweide')
    # rad = 5
    # radx = 5
    # rady = 2.5


    ymax = dec+rady/2
    ymin = dec-rady/2
    xmin = ra-radx/2
    xmax = ra+radx/2

    tfig   = plt.figure(num=2)
    rot = (0, 0, 0)
    coord = 'C'
    xsize = 2048
    # img = hp.cartview(hp_map,fig=2,lonra=[ra-rad,ra+rad],latra=[dec-rad,dec+rad],return_projected_map=True, rot=rot, coord=coord, xsize=xsize)
    img = interpimg(map, xmin,xmax,ymin,ymax,xsize)
    plt.close(tfig)


    fig = plt.figure(dpi=200)
    dMin = -5
    dMax = 15
    dMin = np.min(img) if np.min(img) != None else -5
    dMax = np.max(img) if np.max(img) != None else 15
    if color == "Milagro":
        textcolor, colormap = MapPalette.setupMilagroColormap(dMin-1, dMax+1, 3, 1000)
    elif color == "Fermi":
        textcolor, colormap = MapPalette.setupGammaColormap(10000)
    plt.imshow(img, origin="lower",extent=[xmin,xmax,ymin,ymax], cmap=colormap) #


    plt.grid(linestyle="--")
    cbar = plt.colorbar(format='%.2f',orientation="horizontal",shrink=0.6,
                            fraction=0.1,
                            #aspect=25,
                            pad=0.15)

    cbar.set_label(colorlabel)
    if np.max(img)<4:
        cbar.set_ticks(np.concatenate(([np.min(img)],[np.mean(img)],[np.max(img)])))
    elif np.max(img)<6:
        cbar.set_ticks(np.concatenate(([np.min(img)],[np.mean(img)],[3],[np.max(img)])))
    elif np.max(img)<30:
        cbar.set_ticks(np.concatenate(([np.min(img)],[np.mean(img)],[3],[5],[np.max(img)])))
    elif np.max(img)<40:
        cbar.set_ticks(np.concatenate(([np.min(img)],[5],[np.max(img)])))
    else:
        cbar.set_ticks(np.concatenate(([np.min(img)],[np.mean([np.min(img),np.max(img)])],[np.max(img)])))
    #,cbar.get_ticks()

    contp = plt.contour(img,levels=np.sort(contours),colors='g',linestyles = '-',linewidths = 2,origin='upper',extent=[xmin, xmax, ymax, ymin])
    fmt = {}
    strs=[]
    for i in range(len(contours)):
        strs.append('%d$\sigma$'%(contours[i]))
    for l, s in zip(contp.levels, strs):
        fmt[l] = s

    CLabel = plt.clabel(contp, contp.levels, use_clabeltext=True, rightside_up=True, inline=1, fmt=fmt, fontsize=10)

    for l in CLabel:
        l.set_rotation(180)


    plt.xlabel(r"$\alpha$ [$^\circ$]")
    plt.ylabel(r"$\delta$ [$^\circ$]")

    plt.xlim(ra+radx/2,ra-radx/2)
    plt.ylim(dec-rady/2,dec+rady/2)
    Drawgascontour()

    plt.gca().set_aspect(1./np.cos((ymax+ymin)/2*np.pi/180))
    # plt.scatter(ra, dec, s=20**2,marker="+", facecolor="#000000", color="#000000")
    markerlist=["s","*","o","P","D","v","p","^"]
    for i,catname in enumerate(cat.keys()):
        Drawcat(xmin,xmax,ymin,ymax,catname,cat[catname][1],"black",60,label=catname,textlabel=cat[catname][0])

    if save:
        if plotres:
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh_res.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh_res.pdf")
        else:
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh.png",dpi=300)
            plt.savefig(f"../res/{region_name}/{Modelname}/J0248_sig_llh.pdf")

    return fig

def Draw_lateral_distribution(map, ra, dec, num, width, ifdraw=False):
    colat_crab = np.radians(90-float(dec))
    lon_crab = np.radians(float(ra))
    vec_crab = hp.ang2vec(colat_crab,lon_crab)
    n = num#number of rings
    w = width#width of rings in degrees
    nside = 1024
    npix=hp.nside2npix(nside)
    pixel_areas = 4 * np.pi / npix

 
    data_disc = np.zeros(n) #define the excess_disc number in each disc
    data_ring = np.zeros(n) #define the excess_disc number in each ring
    bkg_disc = np.zeros(n) #define the excess_disc number in each disc
    bkg_ring = np.zeros(n) #define the excess_disc number in each ring
    model_disc = np.zeros(n) #define the excess_disc number in each disc
    model_ring = np.zeros(n) #define the excess_disc number in each ring
    excess_ring = np.zeros(n) #define the excess_disc number in each ring
    res_ring = np.zeros(n) #define the excess_disc number in each ring

    npx_disc = np.zeros(n)   #define the number of pixels in each disc
    npx_ring = np.zeros(n) #define the number of pixels in each ring

    disc = list(np.zeros(n))
    for i in tqdm(range(1,n+1), desc="get disc pixnum"):
        disc[i-1] = hp.query_disc(nside,vec_crab,np.radians(w*i))
        npx_disc[i-1] = disc[i-1].shape[0]
        
    npx_ring[0] = npx_disc[0]
    for i in tqdm(range(1,n), desc="get ring pixnum"):
        npx_ring[i] = npx_disc[i]-npx_disc[i-1]

    psi = np.arange(w/2,n*w,w) #horizontal coordinates

    data=map[0]
    bkg=map[1]
    model=map[2]

    for i in tqdm(range(n),desc="compute disk"):
        data_disc[i] = sum(data[disc[i]])
        bkg_disc[i] = sum(bkg[disc[i]])
        model_disc[i] = sum(model[disc[i]])

    data_ring[0] = data_disc[0]
    bkg_ring[0] = bkg_disc[0] 
    model_ring[0] = model_disc[0]
    errord = np.zeros(n) #poissonian error    
    errord[0] = np.sqrt(sum(data[disc[0]]))
    errorb = np.zeros(n) #poissonian error    
    errorb[0] = np.sqrt(sum(bkg[disc[0]]))
    errorm = np.zeros(n) #poissonian error    
    errorm[0] = np.sqrt(sum(model[disc[0]]))
    for i in tqdm(range(1,n),desc="compute ring"):
        data_ring[i] = data_disc[i]-data_disc[i-1]
        errord[i] = np.sqrt(data_ring[i])
        bkg_ring[i] = bkg_disc[i]-bkg_disc[i-1]
        errorb[i] = np.sqrt(bkg_ring[i])
        model_ring[i] = model_disc[i]-model_disc[i-1]
        errorm[i] = np.sqrt(model_ring[i])
    data_ring/=npx_ring
    bkg_ring/=npx_ring
    model_ring/=npx_ring
    excess_ring = data_ring-bkg_ring
    res_ring = data_ring-model_ring
    errord/=npx_ring
    errorb/=npx_ring
    errorm/=npx_ring

    psfdata = np.array([psi, data_ring, errord,  bkg_ring, errorb, model_ring, errorm, excess_ring, res_ring])

    if ifdraw:
        fig1 = plt.figure()
        plt.errorbar(psfdata[0],psfdata[1],psfdata[2],fmt='o', label="data", c="tab:blue")
        plt.errorbar(psfdata[0],psfdata[5],psfdata[6],fmt='o',label="model", c="tab:red")
        plt.errorbar(psfdata[0],psfdata[3],psfdata[4],fmt='o',label="bkg", c="black")
        plt.xlabel(r"$\phi^{\circ}$")
        plt.ylabel(r"$\frac{excess}{N_{pix}}$")
        plt.legend()
        fig2 = plt.figure()
        plt.errorbar(psfdata[0],psfdata[7],psfdata[2],fmt='o',label="excess", c="black")
        plt.errorbar(psfdata[0],psfdata[8],psfdata[2],fmt='o',label="residual", c="tab:red")
        plt.xlabel(r"$\phi^{\circ}$")
        plt.ylabel(r"$\frac{excess}{N_{pix}}$")
        plt.legend()

    return psfdata
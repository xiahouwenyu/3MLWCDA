import pandas as pd
import numpy as np
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from Mycoord import *
import Mymap

from astropy.coordinates import Angle
from astroquery.vizier import Vizier
import astropy.units as u

from tqdm import tqdm

from astropy.io import fits

try:
    LHAASOCat = pd.read_csv("../../data/LHAASO_Catalog_Table1.csv")
    LHAASOCat=LHAASOCat[LHAASOCat[" Ra"]!=' ']

    data = np.recfromtxt("../../data/RRL_table3.txt")
    data = np.array(data, dtype='U13')
    FASTHIIcat = pd.DataFrame(data[1:], columns=data[0])

    # A large catalogue of molecular clouds with accurate distances within 4 kpc of the Galactic disc
    fits_file = "../../data/J_MNRAS_493_351_table1.dat.fits"
    hdul = fits.open(fits_file)
    MCcat = pd.DataFrame(hdul[1].data) 
except:
    print("not a good position for data")

def getViziercat(name, cut={}):
    Vizier.ROW_LIMIT = -1  # 无限制
    Vizier.MAX_RESULTS = -1 # 无限制
    Vizier.TIMEOUT = 180 # 设置超时时间
    cat = Vizier.get_catalogs(name)[0]
    for k in cut.keys():
        cat = cat[cat[k]<cut[k]]
    cat = cat.to_pandas()
    return cat

def GetAnyCat(xmin,xmax,ymin,ymax, cat, indexname, indexra, indexdec, indexs=None, coor="C", unitx="degree", unity="degree"):
    xa=[]
    ya=[]
    assoca=[]
    sizes = []
    for i in range(0,len(cat)):
        try:
            ras=float(Angle(cat.iloc[i][indexra], unit=unitx).degree) #hourangle
            decs=float(Angle(cat.iloc[i][indexdec], unit=unity).degree)
            if coor=="G":
                ras, decs = gal2edm(ras, decs)
        except:
            continue
        if indexs:
            size = cat.iloc[i][indexs]
            sizes.append(size)
        assoc = cat.iloc[i][indexname]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)	
        ya.append(decs)
        assoca.append(assoc)
    if indexs:
        sources_tmp = list(zip(xa,ya,assoca,sizes))
    else:
        sources_tmp = list(zip(xa,ya,assoca))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def getWR():
    """
        Galactic Wolf-Rayet stars with Gaia DR2 I (Rate+, 2020)
    """
    return getViziercat("J/MNRAS/493/1512")

def GetWR(xmin,xmax,ymin,ymax):
    return GetAnyCat(xmin,xmax,ymin,ymax, getWR(), 4, 5, 6, coor="C", unitx="hourangle")

def getYMC():
    # Milky Way global survey of star clusters. V. (Kharchenko+, 2016) 2MASS
    return getViziercat("J/A+A/585/A101", cut={"logt": np.log10(3e7)})

def GetYMC(xmin,xmax,ymin,ymax):
    # Milky Way global survey of star clusters. V. (Kharchenko+, 2016) 2MASS
    return GetAnyCat(xmin,xmax,ymin,ymax, getYMC(), 1, 4, 5, coor="G")

def getGYMC():
    # Milky Way global survey of star clusters ML GAIA
    return getViziercat("J/A+A/640/A1", cut={"AgeNN": np.log10(3e7)})

def GetGYMC(xmin,xmax,ymin,ymax):
    # Milky Way global survey of star clusters ML GAIA
    return GetAnyCat(xmin,xmax,ymin,ymax, getGYMC(), 0, 1, 2, indexs=3, coor="C")

def GetTeVcat(xmin,xmax,ymin,ymax):
    try:
        import tevcat as TeVCat
        haveTeVCat = True
    except ImportError as e:
        haveTeVCat = False
        print(e)
        
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
    xa=[]
    ya=[]
    assoca=[]
    sizes = []
    sources = tevcat.getSources()
    for i in range(0,len(sources)):
        sourceFK5 = sources[i].getFK5()
        ras=sourceFK5.ra.degree
        decs=sourceFK5.dec.degree
        assoc = sources[i].getCanonicalName()
        size = sources[i].getSize()[0]
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        if assoc in assoca:
            continue
        xa.append(ras)			
        ya.append(decs)			
        assoca.append(assoc)
        sizes.append(size)
    sources_tmp = list(zip(xa,ya,assoca,sizes))
    sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def GetFermicat(xmin,xmax,ymin,ymax,cat="3FHL"):
    # 打开FITS文件
    header = []
    data = []
    if cat == "4FGL":
        file = '../../data/gll_psc_v33.fit'
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

def getSNRcat():
    return getViziercat("VII/284")

def GetSNRcat(xmin,xmax,ymin,ymax):
    # 设置max_catalogs参数为100
    Vizier.ROW_LIMIT = -1  # 无限制
    Vizier.MAX_RESULTS = -1 # 无限制
    Vizier.TIMEOUT = 180 # 设置超时时间
    xa=[]
    ya=[]
    assoca=[]
    # 获取ATNF pulsar目录的数据
    green_catalog = Vizier.get_catalogs('VII/284')[0] #VII/278
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

def GetLHAASOcat(xmin,xmax,ymin,ymax):
    LHAASOCat = pd.read_csv("../../data/LHAASO_Catalog_Table1.csv")
    xa=[]
    ya=[]
    assoca=[]
    sizes=[]
    changeWCDA=False
    for i in range(0,len(LHAASOCat)):
        ras=LHAASOCat.loc[i][2]
        decs=LHAASOCat.loc[i][3]
        det = LHAASOCat.loc[i][1]
        assoc = LHAASOCat.loc[i][0]
        ass = str(LHAASOCat.loc[i][13])
        if ass != " " and ass != "nan":
            assoc+=ass
        if changeWCDA:
            assoc=assoc.replace("LHAASO","WCDA")
            changeWCDA=False

        # if assoc in assoca:
        #     continue
        if "KM2A" in det:
            if ras==" ":
                changeWCDA=True
                continue
            else:
                rakm2a=ras
                deckm2a=decs
                assockm2a=assoc
                continue
        elif "WCDA" in det:
            if ras==" ":
                ras=rakm2a
                decs=deckm2a
                assoc=assockm2a.replace("LHAASO","KM2A")
        ras=float(ras)
        decs=float(decs)
        try:
            size = float(LHAASOCat.loc[i][5])
        except:
            size = 0
        if (ras < xmin) or (ras > xmax) or (decs < ymin) or (decs>ymax):
            continue
        xa.append(ras)
        ya.append(decs)			
        assoca.append(assoc)
        sizes.append(size)
    sources_tmp = list(zip(xa,ya,assoca,sizes))
    # sources_tmp.sort(key=lambda source: source[0])
    return sources_tmp

def Drawcat(xmin,xmax,ymin,ymax,cat="TeVCat",mark="s",c="black",angle=45, fontsize=7, label="Cat",textlabel=False, stype=None, criteria=None, iflabel=1, size=1, drawext=False):
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
    elif cat=="LHAASO":
        sources_tmp = GetLHAASOcat(xmin,xmax,ymin,ymax)
    elif cat=="YMC":
        sources_tmp = GetYMC(xmin,xmax,ymin,ymax) 
    elif cat=="GYMC":
        sources_tmp = GetGYMC(xmin,xmax,ymin,ymax) 
    elif cat=="WR":
        sources_tmp = GetWR(xmin,xmax,ymin,ymax)

    ymid = np.mean([ymin,ymax])
    i=0
    dt=0
    rt=0
    pre_rt1=0
    pre_rt2=0
    dr=(xmax-xmin)/2/(len(sources_tmp)/2.+1)/2
    counts=1
    for source in sources_tmp:
            if len(source)==3:
                r, d, s = source
                print(cat+": ",counts,r, d, s)
                rs=0
            elif len(source)==4:
                r, d, s, rs = source
                print(cat+": ",counts,r, d, s, rs)      
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
                        rotation_mode='anchor',
                        va=Va,
                        fontdict={'family': 'sans-serif',
                                    'size': fontsize,
                                    'weight': 'bold'})
                plt.plot([r,rt],[d,dt],'k--',c=c)
            if iflabel==1:
                plt.scatter(r,d, color=c, facecolors="none", 
                marker=mark,label=label, s=size)
                if rs!=0 and drawext:
                    from matplotlib.patches import Ellipse
                    error_ellipse = Ellipse((r, d), width=(rs)/np.cos(np.radians(d)), height=rs, edgecolor=c, fill=False,linestyle="--", alpha=0.5)
                    ax=plt.gca()
                    ax.add_artist(error_ellipse)
            else:
                plt.scatter(r,d, color=c, facecolors="none", 
                marker=mark, s=size)
                if rs!=0 and drawext:
                    from matplotlib.patches import Ellipse
                    error_ellipse = Ellipse((r, d), width=(rs)/np.cos(np.radians(d)), height=rs, edgecolor=c, fill=False,linestyle="--", alpha=0.5)
                    ax=plt.gca()
                    ax.add_artist(error_ellipse)
            iflabel+=1

def drawcatsimple(LHAASOCat, anycat, colorlhaaso = "tab:cyan",coloranycat="tab:blue", sizecat=500, sizelhaaso=200, catinfo = {"name":"", "RA":"", "DEC":"", "color":"", "size":""}, coor="G", distcut=0.2, bkgmap="../../data/fullsky_WCDA_llh.fits", ifbkg=False, skyrange=(10,80,-2,2), ifcut=True, zmax=30, catname=None, yrange=None):
    #load LHAASO
    LHAASOCat=LHAASOCat[LHAASOCat[" Ra"].values!=' ']
    lhra = LHAASOCat[" Ra"].values; lhra=lhra[lhra!=' ']; lhra=lhra.astype(np.float)
    lhdec = LHAASOCat[" Dec"].values; lhdec=lhdec[lhdec!=' ']; lhdec=lhdec.astype(np.float)
    name = LHAASOCat["Source name"].values
    lhl, lhb = edm2gal(lhra, lhdec)

    #load anycat
    name = anycat[catinfo["name"]].values
    RA = np.array(anycat[catinfo["RA"]].values, dtype="float")
    DEC = np.array(anycat[catinfo["DEC"]].values, dtype="float")
    if "color" in catinfo.keys():
        color = np.array(anycat[catinfo["color"]].values, dtype="float")
    else:
        color = coloranycat
    if "size" in catinfo.keys():
        size = np.array(anycat[catinfo["size"]].values, dtype="float")
    else:
        size = sizecat*np.ones(len(RA))
    l,b=RA,DEC
    if coor=="C":
        l,b = edm2gal(RA, DEC)

    #compute dist
    RA1, RA2 = np.meshgrid(l, lhl)
    DEC1, DEC2 = np.meshgrid(b, lhb)
    distant = skyangle(RA1,DEC1,RA2,DEC2)
    if ifcut:
        indexlhaaso = np.where(distant<distcut)[0]
        indexanycat = np.where(distant<distcut)[1]
    else:
        indexlhaaso = np.where(distant<100000)[0]
        indexanycat = np.where(distant<100000)[1]

    #draw something
    if ifbkg:
        map2, skymapHeader = hp.read_map(bkgmap ,h=True)
        map2 = hp.ma(map2)
        # map2 = my.change_coord(map2,["C","G"])
        Mymap.hpDraw("region_name", "Modelname", map2,0,0,skyrange=skyrange,
            colorlabel="Significance", contours=[1000], save=False, cat={}, color="Fermi", zmax=zmax, xsize=2048)
    else:
        size=size/10
        sizelhaaso=sizelhaaso/10
        plt.figure(figsize=(10,3),dpi=300)
    if not catname:
        catname=catinfo["name"]
    plt.scatter(l[indexanycat], b[indexanycat], s=size[indexanycat], c=color[indexanycat], alpha=0.8,label=catname)
    plt.colorbar(label=catinfo["color"])
    plt.scatter(lhl[indexlhaaso], lhb[indexlhaaso], s=sizelhaaso, c=colorlhaaso,label="LHAASOcat", alpha=0.5)
    plt.xlabel(r"$l^{o}$")
    plt.ylabel(r"$b^{o}$")
    if yrange:
        plt.ylim(yrange[0], yrange[1])
    plt.legend()

    for i in tqdm(range(len(np.where(distant<0.2)[0]))):
        indexlhaaso = np.where(distant<0.2)[0][i]
        indexYMC = np.where(distant<0.2)[1][i]
        # print(indexlhaaso, indexYMC, distant[indexlhaaso][indexYMC], anycat.iloc[indexYMC][catinfo["name"]])
        if i!=0:
            resultsup=results
        results = pd.concat([LHAASOCat.iloc[indexlhaaso],anycat.iloc[indexYMC]])
        results["Dist2LHAASO"] = distant[indexlhaaso][indexYMC]
        if i!=0:
            results = pd.concat([resultsup,results], axis=1)
    return results.T
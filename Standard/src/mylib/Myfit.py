from threeML import *
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
from time import *
from Mymodels import *
import os
import numpy as np
from Myspec import *

import healpy as hp
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm

import root_numpy as rt

from Mymap import *

from Mysigmap import *

from Mycoord import *

try:
    from Mycatalog import LHAASOCat
except:
    pass

from Mylightcurve import p2sigma

log = setup_logger(__name__)
log.propagate = False

deltatime = 3

#####   Model
def setsorce(name,ra,dec,raf=False,decf=False,rab=None,decb=None,
            sigma=None,sf=False,sb=None,radius=None,rf=False,rb=None, sigmar=None, sigmarf=False, sigmarb=None,
            ################################ Spectrum
            k=1.3e-13,kf=False,kb=None,piv=3,pf=True,index=-2.6,indexf=False,indexb=None,alpha=-2.6,alphaf=False,alphab=None,beta=0,betaf=False,betab=None,
            kn=None,
            fitrange=None,
            xc=None, xcf=None, xcb=None,
            ################################ Continuous_injection_diffusion
            rdiff0=None, rdiff0f=False, rdiff0b=None, delta=None, deltaf=False, deltab=None,
            uratio=None, uratiof=False, uratiob=None,                          ##Continuous_injection_diffusion_legacy
            rinj=None, rinjf=True, rinjfb=None, b=None, bf = True, bb=None,                      ##Continuous_injection_diffusion
            incl=None, inclf=True, inclb=None, elongation=None, elongationf=True, elongationb=None,               ##Continuous_injection_diffusion_ellipse
            piv2=1, piv2f=True,

            ################################ Asymm Gaussian on sphere
            a=None, af=False, ab=None, e=None, ef=False, eb=None, theta=None, thetaf=False, thetab=None,

            ################################ Beta
            rc1=None, rc1f=False, rc1b=None, beta1=None, beta1f=False, beta1b=None, rc2=None, rc2f=False, rc2b=None, beta2=None, beta2f=False, beta2b=None, yita=None, yitaf=False, yitab=None,

            ################################ EBL
            redshift=None, ebl_model="franceschini",
            
            spec=None,
            spat=None,
            setdeltabypar=True,
            ratio=None,
            *other,
            **kw):  # sourcery skip: extract-duplicate-method, low-code-quality
    """Create a Sources.

        Args:
            par: Parameters. if sigma is not None, is guassian,or rdiff0 is not None, is Continuous_injection_diffusion,such as this.
            parf: fix it?
            parb: boundary
            spec: Logparabola
            spat: Diffusion/Diffusion2D/Disk/Asymm/Ellipse
        Returns:
            Source
    """
    
    # fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    fluxUnit = 1e-9

    if spec is None:
        if kn is not None:
            spec = PowerlawN() 
        else:
            spec = Powerlaw() 
    
    if spat is None:
        spat=Gaussian_on_sphere()
    elif spat == "Diffusion":
        if uratio != None:
            spat=Continuous_injection_diffusion_legacy()
            spat.piv = piv * 1e9  #* u.TeV
            spat.piv.fix = pf
            spat.piv2 = piv2 * 1e9 #* u.TeV
            spat.piv2.fix = piv2f
        elif rinj != None and b != None:
            spat=Continuous_injection_diffusion()
            spat.piv = piv * 1e9  #* u.TeV
            spat.piv.fix = pf
            spat.piv2 = piv2 * 1e9 #* u.TeV
            spat.piv2.fix = piv2f
        elif incl != None and elongation != None and b != None:
            spat=Continuous_injection_diffusion_ellipse()
            spat.piv = piv * 1e9  #* u.TeV
            spat.piv.fix = pf
            spat.piv2 = piv2 * 1e9 #* u.TeV
            spat.piv2.fix = piv2f
        else:
            raise Exception("Parameters of diffusion model is Incomplete.")
    elif spat == "Diffusion2D":
        if incl != None and elongation != None:
            spat=Continuous_injection_diffusion_ellipse2D()
        else:
            spat=Continuous_injection_diffusion2D()
    elif spat == "Disk":
        spat=Disk_on_sphere()
        if radius != None:
            spat.radius = radius
            spat.radius.fix = rf
            if rb != None:
                spat.radius.bounds = rb
    elif spat == "Beta":
        spat = Beta_function()
    elif spat == "DBeta":
        spat = Double_Beta_function()
    elif spat == "Ring":
        spat = Ring_on_sphere()
        if radius != None:
            spat.radius = radius
            spat.radius.fix = rf
            if rb != None:
                spat.radius.bounds = rb
    elif spat == "Asymm":
        spat=Asymm_Gaussian_on_sphere()
    elif spat == "Ellipse":
        spat=Ellipse_on_sphere()
    else:
        pass

    #  log.info(spat.name)

    if sigma is None and rdiff0 is None and radius is None and a is None and rc1 is None and rc2 is None:
        if redshift is not None:
            eblfunc = EBLattenuation()
            eblfunc.redshift=redshift*u.dimensionless_unscaled
            eblfunc.ebl_model = ebl_model
            if ratio is not None:
                source = PointSource(name,ra,dec,spectral_shape=ratio*spec*eblfunc)
            else:
                source = PointSource(name,ra,dec,spectral_shape=spec*eblfunc)
        else:
            source = PointSource(name,ra,dec,spectral_shape=spec)
        source.position.ra.free=True
        source.position.dec.free=True
        source.position.ra.fix = raf
        source.position.dec.fix = decf
        if rab !=None:
            source.position.ra.bounds=rab
        if decb !=None:
            source.position.dec.bounds=decb
        if fitrange !=None:
            source.position.ra.bounds=(ra-fitrange,ra+fitrange)
            source.position.dec.bounds=(dec-fitrange,dec+fitrange)
    else:
        if redshift is not None:
            eblfunc = EBLattenuation()
            eblfunc.redshift=redshift*u.dimensionless_unscaled
            eblfunc.ebl_model = ebl_model
            source = ExtendedSource(name, spatial_shape=spat, spectral_shape=spec*eblfunc)
        else:
            source = ExtendedSource(name, spatial_shape=spat, spectral_shape=spec)
       


    def setspatParameter(parname,par,parf,parb,unit=""):
        nonlocal spat
        nonlocal spec
        prompt = f"""
if par != None:
    spat.{parname} = par {unit}
    spat.{parname}.fix = parf
if parb != None:
    spat.{parname}.bounds = np.array(parb) {unit}
        """
        exec(prompt)

    def setspecParameter(parname,par,parf,parb,unit=""):
        nonlocal spat
        nonlocal spec
        prompt = f"""
if par != None:
    spec.{parname} = par {unit}
    spec.{parname}.fix = parf
if parb != None:
    spec.{parname}.bounds = np.array(parb) {unit}
        """
        exec(prompt)

    #### set spectral
    spec.K = k * fluxUnit
    spec.K.fix = kf
    if setdeltabypar:
        spec.K.delta = deltatime*k * fluxUnit
    if kn is not None:
        spec.Kn = kn
        spec.Kn.fix = True
    if kb is not None:
        spec.K.bounds = np.array(kb) * fluxUnit

    spec.piv = piv*1e9 #* u.TeV
    spec.piv.fix = pf

    if spec.name == "Log_parabola":
        setspecParameter("alpha",alpha,alphaf,alphab)
        setspecParameter("beta",beta,betaf,betab)
    elif spec.name == "Cutoff_powerlaw":
        xc=xc*1e9
        xcb=(xcb[0]*1e9, xcb[1]*1e9)
        setspecParameter("index",index,indexf,indexb)
        setspecParameter("xc",xc,xcf,xcb)
    elif spec.name == "Powerlaw" or spec.name == "PowerlawM" or spec.name == "PowerlawN":
        setspecParameter("index",index,indexf,indexb)
    #### set spatial

    spat.lon0 = ra
    spat.lat0 = dec
    spat.lon0.fix = raf
    spat.lat0.fix = decf
    if rab !=None:
        spat.lon0.bounds=rab
    if decb !=None:
        spat.lat0.bounds=decb
    if fitrange !=None:
        spat.lon0.bounds=(ra-fitrange,ra+fitrange)
        spat.lat0.bounds=(dec-fitrange,dec+fitrange)

    if sigma != None:
        spat.sigma = sigma #*u.degree
        spat.sigma.fix = sf
    if sb != None:
        spat.sigma.bounds = sb #*u.degree

    
    setspatParameter("rdiff0",rdiff0,rdiff0f,rdiff0b) #,"* u.degree"
    setspatParameter("delta",delta,deltaf,deltab)
    setspatParameter("uratio",uratio,uratiof,uratiob)
    setspatParameter("b",b,bf,bb)
    setspatParameter("incl",incl,inclf,inclb)
    setspatParameter("elongation",elongation,elongationf,elongationb)
    setspatParameter("a",a,af,ab)
    setspatParameter("theta",theta,thetaf,thetab)
    setspatParameter("e",e,ef,eb)
    setspatParameter("rc1",rc1,rc1f,rc1b)
    setspatParameter("rc2",rc2,rc2f,rc2b)
    setspatParameter("beta1",beta1,beta1f,beta1b)
    setspatParameter("beta2",beta2,beta2f,beta2b)
    setspatParameter("yita",yita,yitaf,yitab)
    setspatParameter("sigmar",sigmar,sigmarf,sigmarb)

    return source

def copy_free_parameters(source_model, target_model):
    for param_name, source_param in source_model.free_parameters.items():
        if param_name in target_model.free_parameters:
            target_model.free_parameters[param_name].value = source_param.value
        else:
            print(f"Parameter {param_name} not found in target model")

def getcatModel(ra1, dec1, data_radius, model_radius, detector="WCDA", rtsigma=8, rtflux=15, rtindex=2, rtp=8, fixall=False, roi=None, pf=False, sf=False, kf=False, indexf=False, mpf=True, msf=True, mkf=True, mindexf=True, Kscale=None, releaseall=False, indexb=None, sb=None, kb=None, WCDApiv=3, KM2Apiv=50, setdeltabypar=True, ifext_mt_2=False, releaseroi=None):
    """
        获取LHAASO catalog模型

        Parameters:
            detector: WCDA 还是 KM2A的模型?
            rtsigma: 参数范围是原来模型误差的几倍?
            fixall: 固定所有参数?
            roi: 如果有不规则roi!!!
            pf:  固定位置信息? #, 写得很烂, 后面可以精细化调节想要固定和放开的.
            sf:  固定延展度信息?
            kf:  固定能谱flux?
            indexf: 固定能谱指数?
            mpf: 固定data radius外但是model radius内的位置信息?
            msf: 固定data radius外但是model radius内的延展度信息?
            mkf: 固定data radius外但是model radius内的能谱flux?
            mindexf: 固定data radius外但是model radius内的能谱指数?
            Kscale: 能谱缩放因子
            releaseall: 释放所有ROI内参数?
            indexb: 能谱指数范围 ()
            sb: 延展度范围 ()
            kb: 能谱flux范围 ()

        Returns:
            model
    """ 
    from Mycatalog import LHAASOCat
    activate_logs()
    if ifext_mt_2:
        LHAASOCat=LHAASOCat2
    lm = Model()
    opf = pf; osf=sf; okf=kf; oindexf=indexf
    for i in range(len(LHAASOCat)):
        cc = LHAASOCat.iloc[i][" components"]
        if detector not in cc: continue
        if detector=="WCDA":
            Nc = 1e-13
            piv=WCDApiv
        else:
            Nc = 1e-16
            piv=KM2Apiv
        name = LHAASOCat.iloc[i]["Source name"]
        ras = float(LHAASOCat.iloc[i][" Ra"])
        decs = float(LHAASOCat.iloc[i][" Dec"])
        pe = float(LHAASOCat.iloc[i][" positional error"])
        sigma = float(LHAASOCat.iloc[i][" r39"])
        sigmae = float(LHAASOCat.iloc[i][" r39 error"])
        flux = float(LHAASOCat.iloc[i][" N0"])
        fluxe = float(LHAASOCat.iloc[i][" N0 error"])
        index = float(LHAASOCat.iloc[i][" index"])
        indexe = float(LHAASOCat.iloc[i][" index error"])
        name = name.replace("1LHAASO ","").replace("+","P").replace("-","M").replace("*","").replace(" ","")
        if indexb is not None:
            indexel = indexb[0]
            indexeh = indexb[1]
        else:
            if detector=="WCDA":
                indexel = max(-4,-index-rtindex) #*indexe
                indexeh = min(-1,-index+rtindex)
            else: 
                indexel = max(-5.5,-index-rtindex)
                indexeh = min(-1.5,-index+rtindex)

        if sb is not None:
            sbl = sb[0]
            sbh = sb[1]
        else:
            sbl = sigma-rtsigma*sigmae if sigma-rtsigma*sigmae>0 else 0
            sbh = sigma+rtsigma*sigmae if sigma+rtsigma*sigmae<model_radius else model_radius

        if kb is not None:
            kbl = kb[0]
            kbh = kb[1]
        else:
            if detector=="WCDA":
                kbl = max(1e-15, (flux/rtflux/10)*Nc) #-rtflux*fluxe
                kbh = min(1e-11, (flux*rtflux)*Nc)
            else:
                kbl = max(1e-18, (flux/rtflux/10)*Nc) #-rtflux*fluxe
                kbh = min(1e-14, (flux*rtflux)*Nc)

        if Kscale is not None:
            flux = flux/Kscale
            fluxe = fluxe/Kscale


        doit=False
        if sigma == 0:
            sigma=None
        if roi is None:
            if (distance(ra1,dec1, ras, decs)<data_radius):
                log.info(f"{name} in data_radius: {data_radius} sf:{sf} pf:{pf} kf:{kf} indexf:{indexf}")
                sf = osf 
                pf = opf
                kf = okf
                indexf = oindexf
                doit=True
            elif (distance(ra1,dec1, ras, decs)<=model_radius):
                log.info(f"{name} in model_radius: {model_radius} sf:{sf} pf:{pf} kf:{kf} indexf:{indexf}")
                sf = msf 
                pf = mpf
                kf = mkf
                indexf = mindexf
                doit=True
        else:
            if (distance(ra1,dec1, ras, decs)<data_radius and (hp.ang2pix(1024, ras, decs, lonlat=True) in roi.active_pixels(1024))):
                sf = osf 
                pf = opf
                kf = okf
                indexf = oindexf
                doit=True
                log.info(f"{name} in roi: {data_radius} sf:{sf} pf:{pf} kf:{kf} indexf:{indexf}")
            elif (distance(ra1,dec1, ras, decs)<=model_radius):
                sf = msf 
                pf = mpf
                kf = mkf
                indexf = mindexf
                doit=True
                log.info(f"{name} in model_radius: {model_radius} sf:{sf} pf:{pf} kf:{kf} indexf:{indexf}")

        if kf or indexf:
            if detector=="WCDA":
                piv=3
            else:
                piv=50

        if fixall:
            sf = True
            pf = True
            kf = True
            indexf = True

        if releaseall:
            sf = False
            pf = False
            kf = False
            indexf = False

        if releaseroi is not None:
            if (hp.ang2pix(1024, ras, decs, lonlat=True) in releaseroi.active_pixels(1024)):
                sf = osf
                pf = opf
                kf = okf
                indexf = oindexf
        
        if doit:
            log.info(f"Spec: \n K={flux*Nc:.2e} kb=({kbl:.2e}, {kbh:.2e}) index={-index:.2f} indexb=({indexel:.2f},{indexeh:.2f})")
            if sigma is not None:
                log.info(f"Mor: \n sigma={sigma:.2f} sb=({sbl:.2f},{sbh:.2f}) fitrange={rtp*pe:.2f}")
                prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, sigma={sigma}, sb=({sbl},{sbh}), raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={flux*Nc}, kb=({kbl}, {kbh}), index={-index}, indexb=({indexel},{indexeh}), fitrange={rtp*pe}, kf={kf}, indexf={indexf}, kn={Kscale}, setdeltabypar={setdeltabypar})
lm.add_source({name})
            """
                exec(prompt)
            else:
                log.info(f"Mor: fitrange={rtp*pe:.2f}")
                prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={flux*Nc}, kb=({kbl}, {kbh}), index={-index}, indexb=({indexel},{indexeh}), fitrange={rtp*pe}, kf={kf}, indexf={indexf}, kn={Kscale}, setdeltabypar={setdeltabypar})
lm.add_source({name})
            """
                exec(prompt)
    return lm


def get_modelfromhsc(file, ra1, dec1, data_radius, model_radius, fixall=False, roi=None, releaseall=False, indexb=None, sb=None, kb=None):
    """
        从hsc yaml文件获取模型

        Parameters:
            fixall: 固定所有参数?
            roi: 如果有不规则roi!!!
            pf:  固定位置信息? #, 写得很烂, 后面可以精细化调节想要固定和放开的.
            sf:  固定延展度信息?
            kf:  固定能谱flux?
            indexf: 固定能谱指数?
            mpf: 固定data radius外但是model radius内的位置信息?
            msf: 固定data radius外但是model radius内的延展度信息?
            mkf: 固定data radius外但是model radius内的能谱flux?
            mindexf: 固定data radius外但是model radius内的能谱指数?
            Kscale: 能谱缩放因子
            releaseall: 释放所有ROI内参数?
            indexb: 能谱指数范围 ()
            sb: 延展度范围 ()
            kb: 能谱flux范围 ()

        Returns:
            model
    """ 
    lm = Model()
    import yaml
    config = yaml.load(open(file), Loader=yaml.FullLoader)
    config = dict(config)
    for scid in config.keys():
        scconfig = dict(config[scid])
        name = scconfig['Name'].replace("-", "M").replace("+", "P")
        piv = scconfig['Epiv']
        flux = scconfig['SEDModel']["F0"][0]
        Kb = [scconfig['SEDModel']["F0"][1], scconfig['SEDModel']["F0"][2]]
        kf = scconfig['SEDModel']["F0"][3]
        Nc = scconfig['SEDModel']["F0"][4]
        index =  -scconfig['SEDModel']['alpha'][0]
        Indexb =  [-scconfig['SEDModel']['alpha'][2], -scconfig['SEDModel']['alpha'][1]]
        indexf = scconfig['SEDModel']['alpha'][3]
        ras = scconfig["MorModel"]['ra'][0]
        rab = [scconfig["MorModel"]['ra'][1], scconfig["MorModel"]['ra'][2]]
        pf = scconfig["MorModel"]['ra'][3]
        decs = scconfig["MorModel"]['dec'][0]
        pf = scconfig["MorModel"]['dec'][3]
        decb = [scconfig["MorModel"]['dec'][1], scconfig["MorModel"]['dec'][2]]
        sigma=None
        if scconfig["MorModel"]['type'] == 'Ext_gaus':
            sigma = scconfig["MorModel"]['sigma'][0]
            sf = scconfig["MorModel"]['sigma'][3]
            sigmab = [scconfig["MorModel"]['sigma'][1], scconfig["MorModel"]['sigma'][2]]

        if indexb is not None:
            indexel = indexb[0]
            indexeh = indexb[1]
        else:
            if indexf:
                indexel = None
                indexeh = None
            else:
                indexel = Indexb[0]
                indexeh = Indexb[1]

        if sb is not None:
            sbl = sb[0]
            sbh = sb[1]
        else:
            if sf:
                sbl = None
                sbh = None
            else:
                sbl = sigmab[0]
                sbh = sigmab[1]

        if kb is not None:
            kbl = kb[0]*Nc
            kbh = kb[1]*Nc
        else:
            if kf:
                kbl = None
                kbh = None
            else:
                kbl = Kb[0]*Nc
                kbh = Kb[1]*Nc

        doit=False
        if sigma == 0:
            sigma=None
        if roi is None:
            if (distance(ra1,dec1, ras, decs)<data_radius):
                log.info(f"{name} in data_radius: {data_radius}")
                doit=True
            elif (distance(ra1,dec1, ras, decs)<=model_radius):
                log.info(f"{name} in model_radius: {model_radius}")
                doit=True
                
        else:
            if (distance(ra1,dec1, ras, decs)<data_radius and (hp.ang2pix(1024, ras, decs, lonlat=True) in roi.active_pixels(1024))):
                doit=True
                log.info(f"{name} in roi: {data_radius} sf:{sf} pf:{pf} kf:{kf} indexf:{indexf}")
            elif (distance(ra1,dec1, ras, decs)<=model_radius):
                doit=True
                log.info(f"{name} in model_radius: {model_radius} sf:{sf} pf:{pf} kf:{kf} indexf:{indexf}")

        if fixall:
            sf = True
            pf = True
            kf = True
            indexf = True

        if releaseall:
            sf = False
            pf = False
            kf = False
            indexf = False

        sbs = f"({sbl},{sbh})" if sbl is not None else "None"

        kbs = f"({kbl},{kbh})" if kbl is not None else "None"

        indexbs = f"({indexel},{indexeh})" if indexel is not None else "None"
        
        if doit:
            try:
                log.info(f"Spec: \n K={flux*Nc:.2e} kb=({kbl:.2e}, {kbh:.2e}) index={-index:.2f} indexb=({indexel:.2f},{indexeh:.2f})")
            except:
                pass
            if sigma is not None:
                try:
                    log.info(f"Mor: \n sigma={sigma:.2f} sb=({sbl:.2f},{sbh:.2f})")
                except:
                    pass
                prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, sigma={sigma}, sb={sbs}, raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={flux*Nc}, kb={kbs}, index={-index}, indexb={indexbs}, rab=({rab[0]},{rab[1]}), decb=({decb[0]},{decb[1]}), kf={kf}, indexf={indexf})
lm.add_source({name})
            """
                exec(prompt)
            else:
                log.info(f"Mor: ")
                prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={flux*Nc}, kb={kbs}, index={-index}, indexb={indexbs}, rab=({rab[0]},{rab[1]}), decb=({decb[0]},{decb[1]}), kf={kf}, indexf={indexf})
lm.add_source({name})
            """
                exec(prompt)
    return lm

def model2bayes(model):
    """
        将llh模型设置先验以方便bayes分析

        Parameters:

        Returns:
            model
    """ 
    for param in model.free_parameters.values():

        if param.has_transformation():
            param.set_uninformative_prior(Log_uniform_prior)
        else:
            param.set_uninformative_prior(Uniform_prior)
    return model

def check_bondary(optmodel):
    freepar = optmodel.free_parameters
    ifatlimit = False
    boundpar = []
    for it in freepar.keys():
        if freepar[it].is_normalization and freepar[it].to_dict()["min_value"]>0:
            parv = np.log(freepar[it].to_dict()["value"])
            maxv = np.log(freepar[it].to_dict()["max_value"])
            minv = np.log(freepar[it].to_dict()["min_value"])
        else:
            parv = freepar[it].to_dict()["value"]
            maxv = freepar[it].to_dict()["max_value"]
            minv = freepar[it].to_dict()["min_value"]
        
        if minv is None or maxv is None:
            continue

        if abs((maxv - parv)/(maxv-minv)) < 0.01:
            activate_warnings()
            log.warning(f"Parameter {it} is close to the maximum value: {parv:.2e} < {maxv:.2e}")
            silence_warnings()
            ifatlimit=True
            boundpar.append([it,0])
        if abs((parv - minv)/(maxv-minv)) < 0.01 and not freepar[it].is_normalization:
            activate_warnings()
            log.warning(f"Parameter {it} is close to the minimum value: {parv:.2e} > {minv:.2e}")
            silence_warnings()
            ifatlimit=True
            boundpar.append([it,1])
    return ifatlimit, boundpar
    

def fit(regionname, modelname, Detector,Model,s=None,e=None, mini = "minuit",verbose=False, savefit=True, ifgeterror=False, grids = None, donwtlimit=True, quiet=False, lmini = "minuit"):
    """
        进行拟合

        Parameters:
            Detector: 实例化探测器插件
            s,e: 开始结束bin范围
            mini: minimizer minuit/ROOT/ grid/PAGMO
            verbose: 是否输出拟合过程
            ifgeterror: 是否运行llh扫描获得更准确的误差, 稍微费时间点.
            savefit: 是否保存所有拟合结果到 res/regionname/modelname 文件夹

        Returns:
            >>> [jl,result]
    """ 
    activate_progress_bars()
    if not os.path.exists(f'../res/{regionname}/'):
        os.system(f'mkdir ../res/{regionname}/')
    if not os.path.exists(f'../res/{regionname}/{modelname}/'):
        os.system(f'mkdir ../res/{regionname}/{modelname}/')

    Model.save(f"../res/{regionname}/{modelname}/Model_init.yml", overwrite=True)
    if s is not None and e is not None:
        Detector.set_active_measurements(s,e)
    datalist = DataList(Detector)
    jl = JointLikelihood(Model, datalist, verbose=verbose)
    if mini == "grid" or grids is not None:
        # Create an instance of the GRID minimizer
        grid_minimizer = GlobalMinimization("grid")

        # Create an instance of a local minimizer, which will be used by GRID
        local_minimizer = LocalMinimization(lmini)

        # Define a grid for mu as 10 steps between 2 and 80
        my_grid = grids #{Model.J0248.spatial_shape.lon0: np.linspace(Model.J0248.spatial_shape.lon0.value-2, Model.J0248.spatial_shape.lon0.value+2, 20), Model.J0248.spatial_shape.lat0: np.linspace(Model.J0248.spatial_shape.lat0.value-2, Model.J0248.spatial_shape.lat0.value+2, 10)}

        # Setup the global minimization
        # NOTE: the "callbacks" option is useless in a normal 3ML analysis, it is
        # here only to keep track of the evolution for the plot
        grid_minimizer.setup(
            second_minimization=local_minimizer, grid=my_grid #, callbacks=[get_callback(jl)]
        )

        # Set the minimizer for the JointLikelihood object
        jl.set_minimizer(grid_minimizer)
    elif mini == "PAGMO":
        #Create an instance of the PAGMO minimizer
        pagmo_minimizer = GlobalMinimization("pagmo")

        import pygmo

        my_algorithm = pygmo.algorithm(pygmo.bee_colony(gen=100, limit=50)) #pygmo.bee_colony(gen=20)

        # Create an instance of a local minimizer
        local_minimizer = LocalMinimization(lmini)

        # Setup the global minimization
        pagmo_minimizer.setup(
            second_minimization=local_minimizer,
            algorithm=my_algorithm,
            islands=12,
            population_size=30,
            evolution_cycles=5,
        )

        # Set the minimizer for the JointLikelihood object
        jl.set_minimizer(pagmo_minimizer)
    else:
        jl.set_minimizer(mini)
    result = jl.fit(quiet=quiet)

    ifatb, boundpar = check_bondary(jl.results.optimized_model)
    if donwtlimit:
        if ifatb:
            for it in boundpar:
                ratio=2
                dl = Model.parameters[it[0]].bounds[0]
                ul = Model.parameters[it[0]].bounds[1]
                if any([item in it[0] for item in ["lon0", "lat0", "ra", "dec", "sigma", "index"]]):
                    ratio=1
                    if it[1]==0:
                        Model.parameters[it[0]].bounds = (dl, ul+(ul-dl)*ratio)
                    elif it[1]==1:
                        Model.parameters[it[0]].bounds = (dl-(ul-dl)*ratio, ul)
                else:
                    if Model.parameters[it[0]].is_normalization: #".K" in  boundpar[0]
                        ratio=10
                    if Model.parameters[it[0]].value<0:
                        ratio=1/ratio
                    if it[1]==0:
                        Model.parameters[it[0]].bounds = (dl, ul*ratio)
                    elif it[1]==1:
                        Model.parameters[it[0]].bounds = (dl/ratio, ul)
                log.info(f"Parameter {it[0]} is close to the boundary, extend the boundary to {Model.parameters[it[0]].bounds}.")
            return fit(regionname, modelname, Detector,Model,s,e,mini,verbose, savefit, ifgeterror, grids, donwtlimit)

    if ifgeterror:
        from IPython.display import display
        display(jl.results.get_data_frame())
        result = list(result)
        result[0] = jl.get_errors()

    freepars = []
    fixedpars = []
    for p in Model.parameters:
        try:
            par = Model.parameters[p]
            if par.free:
                freepars.append("%-45s %35.6g ± %2.6g %s" % (p, par.value, result[0]["error"][p], par._unit))
            else:
                fixedpars.append("%-45s %35.6g %s" % (p, par.value, par._unit))
        except:
            continue


    if savefit:
        time1 = strftime("%m-%d-%H", localtime())
        if not os.path.exists(f'../res/{regionname}/'):
            os.system(f'mkdir ../res/{regionname}/')
        if not os.path.exists(f'../res/{regionname}/{modelname}/'):
            os.system(f'mkdir ../res/{regionname}/{modelname}/')
        
        try:
            fig = Detector.display_fit(smoothing_kernel_sigma=0.25, display_colorbar=True)
            fig.savefig(f"../res/{regionname}/{modelname}/fit_result_{s}_{e}.pdf")
        except:
            pass
        Model.save(f"../res/{regionname}/{modelname}/Model.yml", overwrite=True)
        jl.results.write_to(f"../res/{regionname}/{modelname}/Results.fits", overwrite=True)
        jl.results.optimized_model.save(f"../res/{regionname}/{modelname}/Model_opt.yml", overwrite=True)
        with open(f"../res/{regionname}/{modelname}/Results.txt", "w") as f:
            f.write("\nFree parameters:\n")
            for l in freepars:
                f.write("%s\n" % l)
            f.write("\nFixed parameters:\n")
            for l in fixedpars:
                f.write("%s\n" % l)
            f.write("\nStatistical measures:\n")
            f.write(str(result[1].iloc[0])+"\n")
            f.write(str(jl.results.get_statistic_measure_frame().to_dict()))
            
        result[0].to_html(f"../res/{regionname}/{modelname}/Results_detail.html")
        result[0].to_csv(f"../res/{regionname}/{modelname}/Results_detail.csv")
        # new_model_reloaded = load_model("./%s/Model.yml"%(time1))
        # results_reloaded = load_analysis_results("./%s/Results.fits"%(time1))

    return [jl,result]

def get_vari_dis(result, var="J0057.Gaussian_on_sphere.sigma"):
    """
        获取变量采样分布

        Parameters:
            result: 拟合返回的 [jl,result]
            var: 参数名称

        Returns:
            >>> None
    """ 
    rr = result[0].results
    ss = rr.get_variates(var)
    r68 = ss.equal_tail_interval(cl=0.68)
    u95 = ss.equal_tail_interval(cl=2*0.95-1)
    nt,bins,patches=plt.hist(ss.samples, alpha=0.8)
    x = np.arange(r68[0], r68[1], 0.001*bins.std())
    x2 = np.arange(0, u95[1], 0.001*bins.std())
    plt.axvline(r68[0], c="green", alpha=0.8, label=f"r68: {r68[0]:.2e} <--> {r68[1]:.2e}")
    plt.axvline(r68[1], c="green", alpha=0.8)
    plt.fill_between(x,1000*np.ones(len(x)), 0, color="g", alpha=0.3)
    plt.fill_between(x2,1000*np.ones(len(x2)), 0, color="black", alpha=0.3)
    plt.axvline(u95[1], c="black", label=f"upper limit(95%): {u95[1]:.2e}")
    plt.xlabel(var)
    plt.legend()
    plt.ylabel("NSample")
    plt.xlim(left=bins.min()-0.2*bins.std())
    plt.ylim(0,nt.max()+0.2*nt.std())

def jointfit(regionname, modelname, Detector,Model,s=None,e=None,mini = "minuit",verbose=False, savefit=True, ifgeterror=False, grids=None, donwtlimit=True, quiet=False):
    """
        进行联合拟合

        Parameters:
            Detector: 实例化探测器插件列表,如: [WCDA, KM2A]
            s,e: 开始结束bin范围列表, 和探测器同维
            mini: minimizer minuit/ROOT/ grid/PAGMO
            verbose: 是否输出拟合过程
            ifgeterror: 是否运行llh扫描获得更准确的误差, 稍微费时间点.
            savefit: 是否保存所有拟合结果到 res/regionname/modelname 文件夹

        Returns:
            >>> [jl,result]
    """ 
    activate_progress_bars()
    if not os.path.exists(f'../res/{regionname}/'):
        os.system(f'mkdir ../res/{regionname}/')
    if not os.path.exists(f'../res/{regionname}/{modelname}/'):
        os.system(f'mkdir ../res/{regionname}/{modelname}/')

    Model.save(f"../res/{regionname}/{modelname}/Model_init.yml", overwrite=True)
    if s is not None and e is not None:
        for i in range(len(Detector)):
            Detector[i].set_active_measurements(s[i],e[i])
    datalist = DataList(*Detector)
    jl = JointLikelihood(Model, datalist, verbose=verbose)
    if mini == "grid":
        # Create an instance of the GRID minimizer
        grid_minimizer = GlobalMinimization("grid")

        # Create an instance of a local minimizer, which will be used by GRID
        local_minimizer = LocalMinimization("minuit")

        # Define a grid for mu as 10 steps between 2 and 80
        my_grid = grids#{Model.J0248.spatial_shape.lon0: np.linspace(Model.J0248.spatial_shape.lon0.value-2, Model.J0248.spatial_shape.lon0.value+2, 20), Model.J0248.spatial_shape.lat0: np.linspace(Model.J0248.spatial_shape.lat0.value-2, Model.J0248.spatial_shape.lat0.value+2, 10)}

        # Setup the global minimization
        # NOTE: the "callbacks" option is useless in a normal 3ML analysis, it is
        # here only to keep track of the evolution for the plot
        grid_minimizer.setup(
            second_minimization=local_minimizer, grid=my_grid #, callbacks=[get_callback(jl)]
        )

        # Set the minimizer for the JointLikelihood object
        jl.set_minimizer(grid_minimizer)
    elif mini == "PAGMO":
        #Create an instance of the PAGMO minimizer
        pagmo_minimizer = GlobalMinimization("pagmo")

        import pygmo

        my_algorithm = pygmo.algorithm(pygmo.bee_colony(gen=20))

        # Create an instance of a local minimizer
        local_minimizer = LocalMinimization("minuit")

        # Setup the global minimization
        pagmo_minimizer.setup(
            second_minimization=local_minimizer,
            algorithm=my_algorithm,
            islands=10,
            population_size=10,
            evolution_cycles=1,
        )

        # Set the minimizer for the JointLikelihood object
        jl.set_minimizer(pagmo_minimizer)
    else:
        jl.set_minimizer(mini)

    result = jl.fit(quiet=quiet)

    ifatb, boundpar = check_bondary(jl.results.optimized_model)
    if donwtlimit:
        if ifatb:
            for it in boundpar:
                ratio=2
                if any([item in it[0] for item in ["lon0", "lat0", "ra", "dec", "sigma", "index"]]):
                    dl = Model.parameters[it[0]].bounds[0]
                    ul = Model.parameters[it[0]].bounds[1]
                    if it[1]==0:
                        Model.parameters[it[0]].bounds = (dl, ul+(ul-dl)*ratio)
                    elif it[1]==1:
                        Model.parameters[it[0]].bounds = (dl-(ul-dl)*ratio, ul)
                else:
                    if Model.parameters[it[0]].is_normalization: #".K" in  boundpar[0]
                        ratio=10
                    if Model.parameters[it[0]].value<0:
                        ratio=-ratio
                    if it[1]==0:
                        Model.parameters[it[0]].bounds = (dl, ul*ratio)
                    elif it[1]==1:
                        Model.parameters[it[0]].bounds = (dl/ratio, ul) #.bounds[1]
                log.info(f"Parameter {it[0]} is close to the boundary, extend the boundary to {Model.parameters[it[0]].bounds}.")
            return jointfit(regionname, modelname, Detector,Model,s,e,mini,verbose, savefit, ifgeterror, grids, donwtlimit)

    freepars = []
    fixedpars = []
    for p in Model.parameters:
        par = Model.parameters[p]
        if par.free:
            freepars.append("%-45s %35.6g %s" % (p, par.value, par._unit))
        else:
            fixedpars.append("%-45s %35.6g %s" % (p, par.value, par._unit))

    if ifgeterror:
        from IPython.display import display
        display(jl.results.get_data_frame())
        result = list(result)
        result[0] = jl.get_errors()

    if savefit:
        time1 = strftime("%m-%d-%H", localtime())
        if not os.path.exists(f'../res/{regionname}/'):
            os.system(f'mkdir ../res/{regionname}/')
        if not os.path.exists(f'../res/{regionname}/{modelname}/'):
            os.system(f'mkdir ../res/{regionname}/{modelname}/')
        fig=[]
        for i in range(len(Detector)):
            try:
                fig.append(Detector[i].display_fit(smoothing_kernel_sigma=0.25, display_colorbar=True))
                fig[i].savefig(f"../res/{regionname}/{modelname}/fit_result_{s}_{e}.pdf")
            except:
                pass
        Model.save(f"../res/{regionname}/{modelname}/Model.yml", overwrite=True)
        jl.results.write_to(f"../res/{regionname}/{modelname}/Results.fits", overwrite=True)
        jl.results.optimized_model.save(f"../res/{regionname}/{modelname}/Model_opt.yml", overwrite=True)
        with open(f"../res/{regionname}/{modelname}/Results.txt", "w") as f:
            f.write("\nFree parameters:\n")
            for l in freepars:
                f.write("%s\n" % l)
            f.write("\nFixed parameters:\n")
            for l in fixedpars:
                f.write("%s\n" % l)
        result[0].to_html(f"../res/{regionname}/{modelname}/Results_detail.html")
        result[0].to_csv(f"../res/{regionname}/{modelname}/Results_detail.csv")
        # new_model_reloaded = load_model("./%s/Model.yml"%(time1))
        # results_reloaded = load_analysis_results("./%s/Results.fits"%(time1))

    return [jl,result]

def parscan(WCDA, result, par, min=1e-29, max=1e-22, steps=100, log=[False]):
    jjj = result[0]
    rrr=jjj.results

    smresults = jjj.get_contours(par,  min, max, steps, log=log)

    plt.figure()
    CL = 0.95
    plt.plot(smresults[0], 2*(smresults[2]-np.min(smresults[2])))
    deltaTS = 2*(smresults[2]-np.min(smresults[2]))
    trials = smresults[0]
    TSneed = p2sigma(1-(2*CL-1))**2
    indices = np.where(smresults[2] == np.min(smresults[2]))[0]
    newmini = smresults[0][indices]
    try:
        plt.scatter(newmini, 0, marker="*", c="tab:blue", zorder=4, s=100)
    except:
        plt.scatter(newmini, 0, marker="*", c="tab:blue", zorder=4, s=100)

    upper = trials[(deltaTS>=TSneed) & (trials>=newmini)][0]
    sigma1 = trials[(deltaTS>=1) & (trials>=newmini)][0]
    sigma2 = trials[(deltaTS>=4) & (trials>=newmini)][0]
    sigma3 = trials[(deltaTS>=9) & (trials>=newmini)][0]
    plt.axhline(TSneed,color="black", linestyle="--", label=f"95% upperlimit: {upper:.2e}")
    plt.axvline(upper,color="black", linestyle="--")
    plt.axhline(1,color="tab:green", linestyle="--", label=f"1 sigma: {sigma1:.2e}")
    plt.axhline(4,color="tab:orange", linestyle="--", label=f"2 sigma: {sigma2:.2e}")
    plt.axhline(9,color="tab:red", linestyle="--", label=f"3 sigma: {sigma3:.2e}")
    TS = -2*(np.min(smresults[2])-(-WCDA.get_log_like(return_null=True)))
    plt.axhline(TS,color="cyan", linestyle="--", label=f"Model TS: {TS:.2e}")
    if log[0]:
        plt.xscale("log")
    plt.legend()
    plt.ylabel(r"$\Delta TS$")
    # plt.xlabel(par)
    plt.show()
    return upper, sigma1, sigma2, sigma3, TS

def get_profile_likelihood(region_name, Modelname, data, model, par, min=None, max=None, steps=100, log=False, ifplot=False):
    if min is None:
        min = model[par].min_value
    if max is None:
        max = model[par].max_value
    if log:
        mu = np.logspace(np.log10(min), np.log10(max), steps)
    else:
        mu = np.linspace(min, max, steps)

    L = []
    quiet_mode()
    for m in tqdm(mu):
        model[par].value = m
        model[par].fix = True
        result2 = fit(region_name, Modelname, data, model, mini="ROOT", donwtlimit=False, quiet=True)
        L.append(result2[0].current_minimum)
    if ifplot:
        plt.plot(mu, L)
        plt.xlabel(f"{par}")
        plt.ylabel("llh")
        if log:
            plt.xscale("log")
    return mu, L    

def load_modelpath(modelpath):
    # silence_warnings()
    try:
        results = load_analysis_results(modelpath+"/Results.fits")
    except:
        print("No results found")
        results = None

    try:
        lmini = load_model(modelpath+"/Model_init.yml")
    except:
        print("No initial model found")
        lmini = None

    try:
        lmopt = load_model(modelpath+"/Model_opt.yml")
    except:
        print("No optimized model found")
        lmopt = None

    # activate_warnings()
    return results, lmini, lmopt

def getTSall(TSlist, region_name, Modelname, result, WCDA):
    """
        获取TS值

        Parameters:
            TSlist: 想要获取TS得source名称

        Returns:
            >>> 总的TS, TSresults(Dataframe)
    """ 
    TS = {}
    TS_all = WCDA.cal_TS_all()
    log.info(f"TS_all: {TS_all}")
    llh = WCDA.get_log_like()
    log.info(f"llh_all: {llh}")
    for sc in tqdm(TSlist):
        TS[sc]=result[0].compute_TS(sc,result[1][1]).values[0][2]
        log.info(f"TS_{sc}: {TS[sc]}")
    
    TS["TS_all"] = TS_all
    TS["-log(likelihood)"] = -llh
    TSresults = pd.DataFrame([TS])
    TSresults.to_csv(f'../res/{region_name}/{Modelname}/Results.txt', sep='\t', mode='a', index=False)
    TSresults
    return TS, TSresults

def getressimple(WCDA, lm):
    """
        获取简单快速的拟合残差显著性天图,但是显著性y值完全是错的,仅仅看形态分布等

        Parameters:


        Returns:
            残差healpix
    """ 
    WCDA.set_model(lm)
    data=np.zeros(1024*1024*12)
    bkg =np.zeros(1024*1024*12)
    model=np.zeros(1024*1024*12)
    next = lm.get_number_of_extended_sources()
    npt=lm.get_number_of_point_sources()
    for i, plane_id in enumerate(WCDA._active_planes):
        data_analysis_bin = WCDA._maptree[plane_id]
        # try:
        this_model_map_hpx = WCDA._get_model_map(plane_id, npt, next).as_dense()
        # except:
        #     this_model_map_hpx = WCDA._get_model_map(plane_id, npt, next-1).as_dense()
        bkg_subtracted, data_map, background_map = WCDA._get_excess(data_analysis_bin, all_maps=True)
        model += this_model_map_hpx
        bkg   += background_map
        data  += data_map


    data[np.isnan(data)]=hp.UNSEEN
    bkg[np.isnan(bkg)]=hp.UNSEEN
    model[np.isnan(model)]=hp.UNSEEN
    data=hp.ma(data)
    bkg=hp.ma(bkg)
    model=hp.ma(model)
    # resu=data-bkg-model
    on = data
    off = bkg+model
    resu = (on-off)/np.sqrt(on+off)
    resu=hp.sphtfunc.smoothing(resu,sigma=np.radians(0.3))
    return resu

def getresaccuracy(WCDA, lm, signif=17, smooth_sigma=0.3, alpha=3.24e-5):
    """
        获取简单慢速的拟合残差显著性天图,LIMA显著性

        Parameters:


        Returns:
            残差healpix
    """ 
    WCDA.set_model(lm)
    data=np.zeros(1024*1024*12)
    bkg =np.zeros(1024*1024*12)
    model=np.zeros(1024*1024*12)

    next = lm.get_number_of_extended_sources()
    npt=lm.get_number_of_point_sources()
    for i, plane_id in enumerate(WCDA._active_planes):
        data_analysis_bin = WCDA._maptree[plane_id]
        # try:
        this_model_map_hpx = WCDA._get_model_map(plane_id, npt, next).as_dense()
        # except:
        #     this_model_map_hpx = WCDA._get_model_map(plane_id, npt, next-1).as_dense()
        bkg_subtracted, data_map, background_map = WCDA._get_excess(data_analysis_bin, all_maps=True)
        model += this_model_map_hpx
        bkg   += background_map
        data  += data_map
    
    nside=1024
    theta, phi = hp.pix2ang(nside, np.arange(0, 1024*1024*12, 1))
    theta = np.pi/2 - theta
    if alpha is None:
        alpha=2*smooth_sigma*1.51/60./np.sin(theta)


    data[np.isnan(data)]=hp.UNSEEN
    bkg[np.isnan(bkg)]=hp.UNSEEN
    model[np.isnan(model)]=hp.UNSEEN
    data=hp.ma(data)
    bkg=hp.ma(bkg)
    model=hp.ma(model)
    # resu=data-bkg-model
    on = data
    off = bkg+model

    nside=2**10
    npix=hp.nside2npix(nside)
    pixarea = 4 * np.pi/npix
    on1 = hp.sphtfunc.smoothing(on,sigma=np.radians(smooth_sigma))
    on2 = 1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(on,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea
    off1 = hp.sphtfunc.smoothing(off,sigma=np.radians(smooth_sigma))
    off2 = 1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(off,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    
    # resu2=hp.sphtfunc.smoothing(resu,sigma=np.radians(smooth_sigma))
    # resu3=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(resu,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea
    scale=(on1+off1)/(on2+off2)
    ON=on1*scale
    BK=off1*scale
    if signif==5:
        resu=(ON-BK)/np.sqrt(ON+alpha*BK)
    elif signif==9:
        resu=(ON-BK)/np.sqrt(ON*alpha+BK)
    elif signif==17:
        resu=np.sqrt(2.)*np.sqrt(ON*np.log((1.+alpha)/alpha*ON/(ON+BK/alpha))+BK/alpha*np.log((1.+alpha)*BK/alpha/(ON+BK/alpha)))
        resu[ON<BK] *= -1
    else:
        resu=(ON-BK)/np.sqrt(BK)
    # resu = (ON-BK)/np.sqrt(ON)
    return resu

def Search(ra1, dec1, data_radius, model_radius, region_name, WCDA, roi, s, e,  mini = "ROOT", ifDGE=1,freeDGE=1,DGEk=1.8341549e-12,DGEfile="../../data/G25_dust_bkg_template.fits", ifAsymm=False, ifnopt=False, startfromfile=None, startfrommodel=None, fromcatalog=False, cat = { "TeVCat": [0, "s"],"PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"], "4FGL": [0, "d"]}, detector="WCDA", fixcatall=False, extthereshold=9, rtsigma=8, rtflux=15, rtindex=2, rtp=8, ifext_mt_2=True):
    """
        在一个区域搜索新源

        Parameters:
            ifDGE: 是否考虑弥散
            freeDGE: 是否放开弥散
            ifAsymm: 是否使用非对称高斯
            ifnopt: 是否不用点源
            startfrom: 从什么模型开始迭代?
            fromcatalog: 从catalog模型开始迭代?
            cat: 中间图所画的catalog信息
            detector: KM2A还是WCDA!!!!!!
            fixcatall: 是否固定catalog源,如果从catalog开始的话
            extthereshold: 判定延展的阈值

        Returns:
            >>> bestmodel, [jl, result
    """ 
    source=[]
    pts=[]
    exts=[]
    npt=0
    next=0
    TS_all=[0]
    lm=Model()
    lon_array=[]
    lat_array=[]
    Modelname="Original"
    smooth_sigma=0.3
    bestmodel=0
    bestmodelname=0
    
    tDGE=""

    if detector=="WCDA":
        kbs=(1e-15, 1e-11)
        indexbs=(-4, -1)
        kb=(1e-18, 1e-10)
        indexb=(-4.5, -0.5)
    else:
        kbs=(1e-18, 1e-14)
        indexbs=(-5.5, -1.5)
        kb=(1e-18, 1e-14)
        indexb=(-5.5, -0.5)

    if startfromfile is not None:
        lm = load_model(startfromfile)
        exts=[]
        next = lm.get_number_of_extended_sources()
        if 'Diffuse' in lm.sources.keys():
            next-=1
        npt=lm.get_number_of_point_sources()

    if startfrommodel is not None:
        lm = startfrommodel
        exts=[]
        next = lm.get_number_of_extended_sources()
        if 'Diffuse' in lm.sources.keys():
            next-=1
        npt=lm.get_number_of_point_sources()
    
    if fromcatalog:
        lm = getcatModel(ra1, dec1, data_radius, model_radius, fixall=fixcatall, detector=detector,  rtsigma=rtsigma, rtflux=rtflux, rtindex=rtindex, rtp=rtp, ifext_mt_2=ifext_mt_2)
        next = lm.get_number_of_extended_sources()
        if 'Diffuse' in lm.sources.keys():
            next-=1
        npt=lm.get_number_of_point_sources()

    if ifDGE and ('Diffuse' not in lm.sources.keys()):
        if freeDGE:
            tDGE="_DGE_free"
            Diffuse = set_diffusebkg(
                            ra1, dec1, model_radius, model_radius,
                            Kf=False, indexf=False, indexb=indexb,
                            name = region_name, Kb=kb
                            )
        else:
            tDGE="_DGE_fix"
            Diffuse = set_diffusebkg(
                            ra1, dec1, model_radius, model_radius,
                            file=DGEfile,
                            name = region_name
                            )
        lm.add_source(Diffuse)
        exts.append(Diffuse)

    # WCDA.set_model(lm)
    for N_src in range(100):
        # resu = getressimple(WCDA, lm)
        resu = getresaccuracy(WCDA, lm)
        new_source_idx = np.where(resu==np.ma.max(resu))[0][0]
        new_source_lon_lat=hp.pix2ang(1024,new_source_idx,lonlat=True)
        lon_array.append(new_source_lon_lat[0])
        lat_array.append(new_source_lon_lat[1])
        log.info(f"Maxres ra,dec: {lon_array},{lat_array}")
        plt.figure()
        hp.gnomview(resu,norm='',rot=[ra1,dec1],xsize=200,ysize=200,reso=6,title=Modelname)
        plt.scatter(lon_array,lat_array,marker='x',color='red')
        if not os.path.exists(f'../res/{region_name}_iter/'):
            os.system(f'mkdir ../res/{region_name}_iter/')
        plt.savefig(f"../res/{region_name}_iter/{N_src}.png",dpi=300)
        plt.show()

        if not ifnopt:
            npt+=1
            name=f"pt{npt}"
            bestmodelnamec=copy.copy(name)
            pt = setsorce(name,lon_array[N_src],lat_array[N_src], 
                        indexb=indexbs,kb=kbs,
                        fitrange=data_radius)
            lm.add_source(pt)
            bestcache=copy.deepcopy(lm)
            Modelname=f"{npt}pt+{next}ext"+tDGE
            lm.display()
            result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
            TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)
            TS_all.append(TS["TS_all"])
            TS_allpt = TS["TS_all"]
            pts.append(pt)

            sources = get_sources(lm,result)
            sources.pop("Diffuse")
            if detector=="WCDA":
                map2, skymapHeader = hp.read_map("../../data/fullsky_WCDA_20240131_2.6.fits.gz",h=True)
            else:
                map2, skymapHeader = hp.read_map("../../data/fullsky_KM2A_20240131_3.5.fits.gz",h=True)
            map2 = maskroi(map2, roi)
            fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat=cat, color="Fermi")
            plt.show()

        if not ifnopt:
            lm.remove_source(name)
            next+=1; npt-=1
        else:
            next+=1
        name=f"ext{next}"
        Modelname=f"{npt}pt+{next}ext"+tDGE
        if ifAsymm:
            ext = setsorce(name,lon_array[N_src],lat_array[N_src], a=0.1, ae=(0,5), e=0.1, eb=(0,1), theta=10, thetab=(-90,90),
                        indexb=indexbs,kb=kbs,
                        fitrange=data_radius, spat="Asymm")
        else:
            ext = setsorce(name,lon_array[N_src],lat_array[N_src], sigma=0.1, sb=(0,3),
                        indexb=indexbs,kb=kbs,
                        fitrange=data_radius)
        lm.add_source(ext)
        source.append(ext)
        lm.display()
        result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
        TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)

        sources = get_sources(lm,result)
        sources.pop("Diffuse")
        if detector=="WCDA":
            map2, skymapHeader = hp.read_map("../../data/fullsky_WCDA_20240131_2.6.fits.gz",h=True)
        else:
            map2, skymapHeader = hp.read_map("../../data/fullsky_KM2A_20240131_3.5.fits.gz",h=True)
        map2 = maskroi(map2, roi)
        fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat=cat, color="Fermi")
        plt.show()

        if not ifnopt:
            if(TS["TS_all"]-TS_all[-1]>=extthereshold):
                deltaTS = TS["TS_all"]-TS_all[-1]
                log.info(f"Ext is better!! deltaTS={deltaTS:.2f}")
                bestcache=copy.deepcopy(lm)
                bestmodelnamec=copy.copy(name)
                TS_all[-1]=TS["TS_all"]
                exts.append(ext)
                pts.pop()
            else:
                deltaTS = TS["TS_all"]-TS_all[-1]
                log.info(f"pt is better!! deltaTS={deltaTS:.2f}")
                npt+=1
                next-=1
                Modelname=f"{npt}pt+{next}ext"+tDGE
                lm = copy.deepcopy(bestcache)
                WCDA.set_model(lm)
                # lm.remove_source(name)
                # lm.add_source(pts[-1])
                name=f"pt{npt}"
                # result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini="ROOT")
                # TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)
                TS_all[-1]=TS_allpt #TS["TS_all"]
                source[-1]=pt
        else:
            bestcache=copy.deepcopy(lm)
            bestmodelnamec=copy.copy(name)
            TS_all.append(TS["TS_all"])
            exts.append(ext)
        
        plt.show()
        if(N_src==0):
            with open(f'../res/{region_name}_iter/{region_name}_TS.txt', "w") as f:
                f.write("\n")
                f.write("Iter%d TS_total: %f"%(N_src+1,TS_all[-1]) )
                f.write("\n")
        else:
            with open(f'../res/{region_name}_iter/{region_name}_TS.txt', "a") as f:
                f.write("\n")
                f.write("Iter%d TS_total: %f"%(N_src+1,TS_all[-1]) )
                f.write("\n")
                
        if(TS_all[N_src+1]-TS_all[N_src]>25):
            log.info(f"{bestmodelnamec} is better!! deltaTS={TS_all[N_src+1]-TS_all[N_src]:.2f}")
            bestmodelname=bestmodelnamec
            bestmodel=bestcache
        else:
            log.info(f"{bestmodelname} is better!! deltaTS={TS_all[N_src+1]-TS_all[N_src]:.2f}, no need for more!")
            lm.display()
            result = fit(region_name+"_iter", bestmodelname, WCDA, bestcache, s, e,mini="ROOT")
            TS, TSdatafram = getTSall([], region_name+"_iter", bestmodelname, result, WCDA)
            return bestmodel,result

def fun_Logparabola(x,K,alpha,belta,Piv):
    return K*pow(x/Piv,alpha-belta*np.log(x/Piv))

def fun_Powerlaw(x,K,index,piv):
    return K*pow(x/piv,index)

def set_diffusebkg(ra1, dec1, lr=6, br=6, K = None, Kf = False, Kb=None, index =-2.733, indexf = False, file=None, piv=3, name=None, ifreturnratio=False, Kn=None, indexb=None, setdeltabypar=True, kbratio=1000):
    """
        自动生成区域弥散模版

        Parameters:
            lr: 沿着银河的范围半径
            br: 垂直银河的范围
            file: 如果有生成好的模版文件,用它
            name: 模版文件缓存名称

        Returns:
            弥散源
    """ 
    if file == None:
        from astropy.wcs import WCS
        from astropy.io import fits
        if name is None:
            name="Cache"
        root_file=ROOT.TFile.Open(("../../data/gll_dust.root"),"read")
        root_th2d=root_file.Get("gll_region")
        X_nbins=root_th2d.GetNbinsX()
        Y_nbins=root_th2d.GetNbinsY()
        X_min=root_th2d.GetXaxis().GetXmin()
        X_max=root_th2d.GetXaxis().GetXmax()
        Y_min=root_th2d.GetYaxis().GetXmin()
        Y_max=root_th2d.GetYaxis().GetXmax()
        X_size=(X_max-X_min)/X_nbins
        Y_size=(Y_max-Y_min)/Y_nbins
        #  log.info(X_min,X_max,X_nbins, X_size)
        #  log.info(Y_min,Y_max,Y_nbins, Y_size)
        data = rt.hist2array(root_th2d).T
        lranges = lr
        branges = br
        l,b = edm2gal(ra1,dec1)
        # l=int(l); b=int(b)
        ll = np.arange(l-lranges,l+lranges,X_size)
        bb =  np.arange(-branges,branges,Y_size)
        # L,B = np.meshgrid(ll,bb)
        # RA, DEC = gal2edm(L,B)

        lrange=[l-lranges,l+lranges]
        brange=[-branges,branges]

        log.info(f"Set diffuse range: {lrange} {brange}")
        log.info("ra dec coner:")
        log.info(gal2edm(lrange[0], brange[0]))
        log.info(gal2edm(lrange[1], brange[0]))
        log.info(gal2edm(lrange[1], brange[1]))
        log.info(gal2edm(lrange[0], brange[1]))
        dataneed = data[int((brange[0]-Y_min)/Y_size):int((brange[1]-Y_min)/Y_size),int((lrange[0]-X_min)/X_size):int((lrange[1]-X_min)/X_size)]

        s = dataneed.copy()
        for idec,decd in  enumerate(dataneed):
            ddd = brange[0]+idec*Y_size
            for ira,counts in enumerate(decd):
                lll = lrange[0]+ira*X_size
                s[idec,ira] = (np.radians(X_size)*np.radians(Y_size)*np.cos(np.radians(ddd)))
                # s[idec,ira] = (X_size*Y_size*np.cos(np.radians(ddd)))

        A = np.multiply(dataneed,s)
        ss = np.sum(s)
        sa = np.sum(A)



        # fi*si ours
        zsa = 1.3505059134209275e-05
        # si ours
        zss = 0.41946493776343513

        # # fi*si ours
        # zsa = sa
        # # si ours
        # zss = ss

        # fi*si hsc
        hsa = 1.33582226223935e-05
        # si hsc
        hss = 0.18184396950291062

        F0=10.394e-12/(u.TeV*u.cm**2*u.s*u.sr)
        # K=F0*(sa*u.sr)/(ss*u.sr)/((hsa*u.sr)/(hss*u.sr))
        if K is None:
            K = F0*hss*(sa/hsa) #/ss
            Kz = F0*hss*(zsa/hsa) #/ss
            K = K.value

        log.info(f"total sr: {ss}"+"\n"+f"ratio: {ss/2.745913003176557}")
        log.info(f"integration: {sa}"+"\n"+f"ratio: {sa/0.00012671770357488944}")
        log.info(f"set K to: {K}")

        # 定义图像大小
        naxis1 = len(ll)  # 银经
        naxis2 = len(bb)  # 银纬

        # 定义银道坐标范围
        lon_range = lrange  # 银经
        lat_range = brange  # 银纬

        # 创建 WCS 对象
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [naxis1 / 2, naxis2 / 2]  # 中心像素坐标
        wcs.wcs.cdelt = np.array([0.1, 0.1])  # 每个像素的尺寸，单位为度
        wcs.wcs.crval = [l, 0]  # 图像中心的银道坐标，单位为度
        wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']  # 坐标系类型
        # 创建头文件
        header = fits.Header()
        header.update(wcs.to_header())
        header['OBJECT'] = 'Test Image'
        header['BUNIT'] = 'Jy/beam'

        # 创建 HDU
        hdu = fits.PrimaryHDU(data=dataneed/sa, header=header)

        # 保存为 FITS 文件
        file = f'../../data/{name}_dust_bkg_template.fits'
        log.info(f"diffuse file path: {file}")
        hdu.writeto(file, overwrite=True)
    # fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    fluxUnit = 1e-9

    if Kn is not None:
        Diffusespec = PowerlawN()
        Diffuseshape = SpatialTemplate_2D(fits_file=file)
        Diffuse = ExtendedSource("Diffuse",spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
        kk=float(K/Kn)
        Kb=np.array(Kb)/float(Kn)
        Diffusespec.K = kk * fluxUnit
        Diffusespec.K.fix=Kf
        if setdeltabypar:
            Diffusespec.K.delta = deltatime*kk * fluxUnit

        if Kb is not None:
            Diffusespec.K.bounds=np.array(Kb) * fluxUnit
        else:
            Diffusespec.K.bounds=np.array((kk/kbratio,kbratio*kk)) * fluxUnit
        Diffusespec.Kn = Kn
        Diffusespec.Kn.fix = True
    else:
        Diffusespec = Powerlaw()
        Diffuseshape = SpatialTemplate_2D(fits_file=file)
        Diffuse = ExtendedSource("Diffuse",spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
        Diffusespec.K = K * fluxUnit
        Diffusespec.K.fix=Kf
        if setdeltabypar:
            Diffusespec.K.delta = deltatime*K * fluxUnit
        if Kb is not None:
            Diffusespec.K.bounds=np.array(Kb) * fluxUnit
        else:
            Diffusespec.K.bounds=np.array((K/kbratio,kbratio*K)) * fluxUnit



    Diffusespec.piv = piv * u.TeV
    Diffusespec.piv.fix=True

    Diffusespec.index = index
    Diffusespec.index.fix = indexf
    if indexb is not None:
        Diffusespec.index.bounds = indexb
    else:
        Diffusespec.index.bounds = (-4,-1)
    Diffuseshape.K = 1/u.deg**2
    if ifreturnratio:
        return Diffuse, [sa/0.00012671770357488944, ss, ss/2.745913003176557],
    else:
        return Diffuse

def set_diffusemodel(name, fits_file, K = 7.3776826e-13, Kf = False, Kb=None, index =-2.733, indexf = False, piv=3, setdeltabypar=True, kbratio=1000, ratio=None, spec = Powerlaw()):
    """
        读取fits的形态模版

        Parameters:
            fits_file: 模版fits文件,正常格式就行,但需要归一化到每sr

        Returns:
            弥散源
    """ 
    # fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    fluxUnit = 1e-9
    Diffuseshape = SpatialTemplate_2D(fits_file=fits_file)
    Diffusespec = spec
    if ratio is not None:
        Diffuse = ExtendedSource(name, spatial_shape=Diffuseshape,spectral_shape=ratio*Diffusespec)
    else:
        Diffuse = ExtendedSource(name, spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
    Diffusespec.K = K * fluxUnit
    Diffusespec.K.fix=Kf

    if setdeltabypar:
        Diffusespec.K.delta = 1*K * fluxUnit

    if Kb:
        Diffusespec.K.bounds=np.array(Kb) * fluxUnit
    else:
        Diffusespec.K.bounds=np.array((K/kbratio,kbratio*K)) * fluxUnit

    Diffusespec.piv = piv * u.TeV
    Diffusespec.piv.fix=True

    Diffusespec.index = index
    Diffusespec.index.fix = indexf
    Diffusespec.index.bounds = (-4,-1)
    Diffuseshape.K = 1/u.deg**2
    return Diffuse


def get_sources(lm,result=None):
    """Get info of Sources.

        Args:
        Returns:
            Sources info
    """
    sources = {}
    for name,sc in lm.sources.items():
        source = {}
        for p in sc.parameters:
            source['type'] = str(sc.source_type)
            try:
                source['shape'] = sc.spatial_shape.name
            except:
                source['shape'] = "Point source"
            par = sc.parameters[p]
            if par.free:
                if result is not None:
                    puv = result[1][0].loc[p,"positive_error"]
                    plv = result[1][0].loc[p,"negative_error"]
                else:
                    puv=0
                    plv=0
                source[p.split('.')[-1]] = (1,par,par.value,puv,plv)
            else:
                source[p.split('.')[-1]] = (0,par,par.value,0,0)
            sources[name] = source
    return sources
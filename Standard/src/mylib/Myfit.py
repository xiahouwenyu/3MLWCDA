from threeML import *
try:
    from hawc_hal import HAL, HealpixConeROI, HealpixMapROI
except:
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
# import root_numpy as rt
from Mymap import *
from Mysigmap import *
from Mycoord import *
try:
    from Mycatalog import LHAASOCat
except:
    pass
from scipy.optimize import minimize
from Mylightcurve import p2sigma

log = setup_logger(__name__)
log.propagate = False

deltatimek = 0.5  # default delta time for the parameters

#####   Model
def setsorce(name,ra,dec,raf=False,decf=False,rab=None,decb=None,
            sigma=None,sf=False,sb=None,radius=None,rf=False,rb=None, sigmar=None, sigmarf=False, sigmarb=None,
            ################################ Spectrum
            k=None,kf=False,kb=None,piv=3,pf=True,index=-2.6,indexf=False,indexb=None,alpha=-2.6,alphaf=False,alphab=None,beta=0,betaf=False,betab=None,
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

            ################################ Other parameters
            deltatimek = deltatimek,  deltap=0.05, deltas=0.05, deltaspec=0.05, # delta time for the parameters
            
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
    if k is None:
        k = fun_Logparabola(x=piv, K=2.5e-13, alpha=-2.7, belta=0.15, Piv=3)

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
            spat.radius.delta = deltas
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
            spat.radius.delta = deltas
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
        source.position.ra = ra #* u.deg
        # source.position.ra.free=True
        # source.position.dec = dec
        # source.position.dec.free=True
        source.position.ra.fix = raf
        source.position.dec.fix = decf
        source.position.ra.delta = deltap
        source.position.dec.delta = deltap
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
       


    def setspatParameter(parname,par,parf,parb,unit="",delta=None):
        nonlocal spat
        nonlocal spec
        prompt = f"""
if par != None:
    spat.{parname} = par {unit}
    spat.{parname}.fix = parf
    if delta != None:
        spat.{parname}.delta = delta {unit}
if parb != None:
    spat.{parname}.bounds = np.array(parb) {unit}

        """
        exec(prompt)

    def setspecParameter(parname,par,parf,parb,unit="",delta=None):
        nonlocal spat
        nonlocal spec
        prompt = f"""
if par != None:
    spec.{parname} = par {unit}
    spec.{parname}.fix = parf
    if delta != None:
        spec.{parname}.delta = delta {unit}
if parb != None:
    spec.{parname}.bounds = np.array(parb) {unit}
        """
        exec(prompt)

    #### set spectral
    spec.K = k * fluxUnit
    spec.K.fix = kf
    if setdeltabypar:
        spec.K.delta = deltatimek*k * fluxUnit
    if kn is not None:
        spec.Kn = kn
        spec.Kn.fix = True
    if kb is not None:
        spec.K.bounds = np.array(kb) * fluxUnit

    spec.piv = piv*1e9 #* u.TeV
    spec.piv.fix = pf

    if spec.name == "Log_parabola":
        setspecParameter("alpha",alpha,alphaf,alphab,delta=deltaspec)
        setspecParameter("beta",beta,betaf,betab,delta=deltaspec)
    elif spec.name == "Cutoff_powerlaw" or spec.name == "Cutoff_powerlawM":
        xc=xc*1e9
        xcb=(xcb[0]*1e9, xcb[1]*1e9)
        setspecParameter("index",index,indexf,indexb,delta=deltaspec)
        setspecParameter("xc",xc,xcf,xcb,1e9)
    elif spec.name == "Powerlaw" or spec.name == "PowerlawM" or spec.name == "PowerlawN":
        setspecParameter("index",index,indexf,indexb,delta=deltaspec)
    #### set spatial

    spat.lon0 = ra
    spat.lat0 = dec
    spat.lon0.delta = deltap #*u.degree
    spat.lat0.delta = deltap #*u.degree
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
        spat.sigma.delta = deltas #*u.degree
    if sb != None:
        spat.sigma.bounds = sb #*u.degree

    
    setspatParameter("rdiff0",rdiff0,rdiff0f,rdiff0b, delta=deltas) #,"* u.degree"
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
            try:
                target_model.free_parameters[param_name].value = source_param.value
            except Exception as e:
                if "minimum" in str(e):
                    target_model.free_parameters[param_name].bound[0] = source_param.value - 0.1*source_param.value
                    target_model.free_parameters[param_name].value = source_param.value
                elif "maximum" in str(e):
                    target_model.free_parameters[param_name].bound[1] = source_param.value + 0.1*source_param.value
                    target_model.free_parameters[param_name].value = source_param.value
        else:
            print(f"Parameter {param_name} not found in target model")

def change_spectrum(lm, ss, spec=Log_parabola(), piv=None):
    if spec.name == "Log_parabola":
        spec.alpha = lm.sources[ss.name].spectrum.main.Powerlaw.index.value
        spec.alpha.bounds = lm.sources[ss.name].spectrum.main.Powerlaw.index.bounds
        spec.K = lm.sources[ss.name].spectrum.main.Powerlaw(piv*1e9)
        spec.K.bounds = [spec.K.value/20, spec.K.value*20]
        spec.beta = np.random.uniform(0, 0.3)
        spec.beta.bounds = (0, 1)
        if piv is not None:
            spec.piv = piv*1e9
            spec.piv.fix = True
    elif spec.name == "Cutoff_powerlaw":
        spec.index = lm.sources[ss.name].spectrum.main.Powerlaw.index.value
        spec.index.bounds = (lm.sources[ss.name].spectrum.main.Powerlaw.index.bounds[0]-2, lm.sources[ss.name].spectrum.main.Powerlaw.index.bounds[1]+4)
        spec.xc = piv*1e9
        spec.xc.bounds = (0.1*1e9, 500*1e9)
        spec.K = lm.sources[ss.name].spectrum.main.Powerlaw(piv*1e9)
        spec.K.bounds = [spec.K.value/100, spec.K.value*100]
        if piv is not None:
            spec.piv = piv*1e9
            spec.piv.fix = True
    try:
        source = ExtendedSource(ss.name, spatial_shape=ss.spatial_shape, spectral_shape=spec)
    except:
        source = PointSource(ss.name, ra=ss.position.ra.value, dec=ss.position.dec.value, spectral_shape=spec)
    lm.remove_source(ss.name)
    lm.add_source(source)

def fit_logparabola_from_powerlaws(
    A1, gamma1, E1,
    A2, gamma2, E2,
    pivot=20.0
):
    """
    根据两个powerlaw点（flux + slope）同时拟合一个最优logparabola
    
    参数:
    - A1, gamma1, E1: WCDA点的归一化、谱指数、能量（TeV）
    - A2, gamma2, E2: KM2A点的归一化、谱指数、能量（TeV）
    - pivot: LogParabola的pivot能量（TeV）
    
    返回:
    - A0, alpha, beta: LogParabola参数
    """

    logE1 = np.log(E1 / pivot)
    logE2 = np.log(E2 / pivot)

    def loss(params):
        logA0, alpha, beta = params
        # flux at E1 and E2
        logphi1_pred = logA0 - alpha * logE1 - beta * logE1**2
        logphi2_pred = logA0 - alpha * logE2 - beta * logE2**2

        # slope at E1 and E2
        gamma1_pred = alpha + 2 * beta * logE1
        gamma2_pred = alpha + 2 * beta * logE2

        # true values
        logphi1_true = np.log(A1)
        logphi2_true = np.log(A2)

        # residuals
        r1 = logphi1_pred - logphi1_true
        r2 = logphi2_pred - logphi2_true
        r3 = gamma1_pred - gamma1
        r4 = gamma2_pred - gamma2

        return r1**2 + r2**2 + r3**2 + r4**2  # total squared error

    # 初始猜测
    logA0_guess = np.log((A1 + A2)/2)
    alpha_guess = (gamma1 + gamma2) / 2
    beta_guess = (gamma2 - gamma1) / (2 * (logE2 - logE1))
    
    result = minimize(loss, [logA0_guess, alpha_guess, beta_guess])

    logA0, alpha, beta = result.x
    A0 = np.exp(logA0)
    return A0, alpha, beta

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
        samesource = False
        cc = LHAASOCat.iloc[i][" components"]
        if detector=="WCDA":
            if detector not in cc: continue
            Nc = 1e-13
            piv=WCDApiv
        elif detector=="KM2A":
            if detector not in cc: continue
            Nc = 1e-16
            piv=KM2Apiv
        elif detector=="jf":
            Nc1 = 1e-13
            Nc2 = 1e-16
            piv=20
        name = LHAASOCat.iloc[i]["Source name"]
        # lastname = LHAASOCat.iloc[i-1]["Source name"]
        # nextname = LHAASOCat.iloc[i+1]["Source name"]
        if i != len(LHAASOCat)-1 and name.replace("1LHAASO ","").replace("+","P").replace("-","M").replace("*","").replace(" ","") == LHAASOCat.iloc[i+1]["Source name"].replace("1LHAASO ","").replace("+","P").replace("-","M").replace("*","").replace(" ","") and detector=="jf":
            continue
        if i!=0 and name.replace("1LHAASO ","").replace("+","P").replace("-","M").replace("*","").replace(" ","") == LHAASOCat.iloc[i-1]["Source name"].replace("1LHAASO ","").replace("+","P").replace("-","M").replace("*","").replace(" ","") and detector=="jf":
            samesource = True
            ras = (float(LHAASOCat.iloc[i][" Ra"])+float(LHAASOCat.iloc[i-1][" Ra"]))/2
            decs = (float(LHAASOCat.iloc[i][" Dec"])+float(LHAASOCat.iloc[i-1][" Dec"]))/2
            pe = abs(float(LHAASOCat.iloc[i][" Ra"])-float(LHAASOCat.iloc[i-1][" Ra"]))+float(LHAASOCat.iloc[i][" positional error"])
            sigma = (float(LHAASOCat.iloc[i][" r39"])+float(LHAASOCat.iloc[i-1][" r39"]))/2
            sigmae = abs(float(LHAASOCat.iloc[i][" r39 error"])-float(LHAASOCat.iloc[i-1][" r39 error"]))+float(LHAASOCat.iloc[i][" r39 error"])
            flux1 = float(LHAASOCat.iloc[i][" N0"])
            flux2 = float(LHAASOCat.iloc[i-1][" N0"])
            fluxe = float(LHAASOCat.iloc[i][" N0 error"])
            index1 = float(LHAASOCat.iloc[i][" index"])
            index2 = float(LHAASOCat.iloc[i-1][" index"])
            indexe = float(LHAASOCat.iloc[i][" index error"])
        else:
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
            elif detector=="KM2A":
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

        # if kf or indexf:
        #     if detector=="WCDA":
        #         piv=3
        #     else:
        #         piv=50

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
            if detector=="jf":
                if samesource:
                    A0, alpha, beta = fit_logparabola_from_powerlaws(
                        flux1*Nc1, index1, 3,
                        flux2*Nc2, index2, 50,
                        pivot=20.0
                    )
                    alpha = -alpha
                else:
                    if "WCDA" in cc:
                        A0 = fun_Powerlaw(20,flux*Nc1,-index,3)
                    elif "KM2A" in cc:
                        A0 = fun_Powerlaw(20,flux*Nc2,-index,50)
                    alpha = -index
                    beta = 0
                kbl = max(1e-18, (A0/rtflux/10)) #-rtflux*fluxe
                kbh = min(1e-12, (A0*rtflux))
                indexel = max(-8,alpha-rtindex)
                indexeh = min(10,alpha+rtindex)
                log.info(f"Spec: \n K={A0:.2e} kb=({kbl:.2e}, {kbh:.2e}) index={alpha:.2f} indexb=({indexel:.2f},{indexeh:.2f})")
                if sigma is not None:
                    log.info(f"Mor: \n sigma={sigma:.2f} sb=({sbl:.2f},{sbh:.2f}) fitrange={rtp*pe:.2f}")
                    prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, sigma={sigma}, sb=({sbl},{sbh}), raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={A0}, kb=({kbl}, {kbh}), alpha={alpha}, alphab=({indexel},{indexeh}), beta={beta}, betab=(0,3), fitrange={rtp*pe}, kf={kf}, alphaf={indexf}, kn={Kscale}, setdeltabypar={setdeltabypar}, spec=Log_parabola())
lm.add_source({name})
                    """
                    exec(prompt)
                else:
                    log.info(f"Mor: fitrange={rtp*pe:.2f}")
                    prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={A0}, kb=({kbl}, {kbh}), alpha={alpha}, alphab=({indexel},{indexeh}), beta={beta}, betab=(0,3), fitrange={rtp*pe}, kf={kf}, alphaf={indexf}, kn={Kscale}, setdeltabypar={setdeltabypar}, spec=Log_parabola())
lm.add_source({name})
                    """
                    exec(prompt)
            else:
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





def save_lhaaso_model(lhaaso_model, filepath):
    """保存LHAASO模型为YAML格式，使用紧凑数组表示法"""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedSeq
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = None
    
    # 将普通字典和列表转换为ruamel.yaml的特殊对象
    def convert_to_ruamel(data):
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                new_dict[k] = convert_to_ruamel(v)
            return new_dict
        elif isinstance(data, list):
            new_list = CommentedSeq(convert_to_ruamel(x) for x in data)
            if all(isinstance(x, (int, float, str)) for x in data):
                new_list.fa.set_flow_style()  # 只有在这里设置流式风格
            return new_list
        else:
            return data
    
    # 转换整个数据结构
    ruamel_model = convert_to_ruamel(lhaaso_model)
    
    # 保存到文件
    with open(filepath, 'w') as f:
        yaml.dump(ruamel_model, f)



def split_sci_num(num):
    """分解科学计数法数字为系数和指数，确保系数在 [1, 10) 范围内"""
    from decimal import Decimal, getcontext
    
    # 设置高精度
    getcontext().prec = 20
    
    d = Decimal(str(num))
    if d == 0:
        return 0.0, 0
    
    # 获取标准科学计数法的系数和指数
    coeff = float(d.scaleb(-d.adjusted()).to_eng_string())
    exp = d.adjusted()
    
    # 手动调整到 [1, 10) 范围（防止极端情况）
    while abs(coeff) >= 10.0:
        coeff /= 10.0
        exp += 1
    while abs(coeff) < 1.0:
        coeff *= 10.0
        exp -= 1
    
    return coeff, exp

def convert_3ml_to_lhaaso(three_ml_model, region_name, Modelname, piv=50, save=True):
    """
    将3ML模型转换为与示例完全匹配的LHAASO格式
    
    参数:
        three_ml_model (dict): 3ML模型字典
        
    返回:
        dict: 与示例格式一致的LHAASO模型字典
    """
    three_ml_model = three_ml_model.to_dict()
    lhaaso_model = {
        "DGE": {
            "Active": 1,
            "ConvoPSF": 1,
            "Template0": {
                "Name": "dust",
                "Tempfile": "/home/lhaaso/hushicong/Standard_prog_lib/Source_Analysis/Space_energy_Joint_fitting/v0.65_combine_diff/Template/gll_dust_process_Eqm.root",  # 需要替换为实际路径
                "TempHist": ["hTemp_ana"],
                "Epiv": piv,
                "SEDModel": {
                    "type": "PL",
                    "F0": [250, 0.00, 500.00, 0, 1.e-17],
                    "alpha": [2.74, 2.00, 4.00, 0]
                }
            }
        },
        "SRC": {
            "UseCatalog": 0,
            "Active": 1,
            "Epiv": piv,
            "ParStatus": {
                "Position": 0,
                "F0": 0,
                "Index": 0,
                "MorPar": 0
            }
        }
    }

    # 处理点源和高斯源
    src_index = 0
    for src_name, src_data in three_ml_model.items():
        if src_name in ['WCDA_bkg_renorm', 'KM2A_bkg_renorm', 'Diffuse']:
            continue
            
        # 初始化源模型
        src_key = f"Src{src_index}"
        src_model = {
            "Name": src_name,
            "Epiv": piv,
            "SEDModel": {},
            "MorModel": {}
        }
        
        # 处理谱模型
        spectrum_data = src_data['spectrum']['main']
        spectrum_type = list(spectrum_data.keys())[0]
        
        
        if spectrum_type == 'Powerlaw':
            pl_params = spectrum_data[spectrum_type]
            coeff, exp = split_sci_num(pl_params['K']['value'] * 1e9)
            coeff1, exp1 = split_sci_num(pl_params['K']['min_value'] * 1e9)
            coeff1 = coeff1*10**exp1/10**exp
            coeff2, exp2 = split_sci_num(pl_params['K']['max_value'] * 1e9)
            coeff2 = coeff2*10**exp2/10**exp
            src_model["SEDModel"] = {
                "type": "PL",
                "F0": [
                    coeff,  # 当前值
                    coeff1,  # 最小值
                    coeff2,  # 最大值
                    int(not pl_params['K']['free']),  # 是否可调
                    10**exp  # 步长 f"{:.1e}
                ],
                "alpha": [
                    -pl_params['index']['value'],  # 当前值
                    -pl_params['index']['max_value'],  # 最小值
                    -pl_params['index']['min_value'],  # 最大值
                    int(not pl_params['index']['free'])  # 是否可调
                ]
            }
            
        elif spectrum_type == 'Log_parabola':
            lp_params = spectrum_data[spectrum_type]
            coeff, exp = split_sci_num(lp_params['K']['value'] * 1e9)
            coeff1, exp1 = split_sci_num(lp_params['K']['min_value'] * 1e9)
            coeff1 = coeff1*10**exp1/10**exp
            coeff2, exp2 = split_sci_num(lp_params['K']['max_value'] * 1e9)
            coeff2 = coeff2*10**exp2/10**exp
            src_model["SEDModel"] = {
                "type": "LP",
                "F0": [
                    coeff,  # 当前值
                    coeff1,  # 最小值
                    coeff2,  # 最大值
                    int(not lp_params['K']['free']),  # 是否可调
                    10**exp  # 步长 f"{:.1e}"
                ],
                "alpha": [
                    -lp_params['alpha']['value'],  # 当前值
                    -lp_params['alpha']['max_value'],  # 最小值
                    -lp_params['alpha']['min_value'],  # 最大值
                    int(not lp_params['alpha']['free'])  # 是否可调
                ],
                "beta": [
                    lp_params['beta']['value'],  # 当前值
                    lp_params['beta']['min_value'],  # 最小值
                    lp_params['beta']['max_value'],  # 最大值
                    int(not lp_params['beta']['free'])  # 是否可调
                ]
            }
        
        # 处理空间模型
        if 'position' in src_data:
            pos = src_data['position']
            src_model["MorModel"] = {
                "type": "Point",
                "ra": [
                    pos['ra']['value'],  # 当前值
                    pos['ra']['min_value'],  # 最小值
                    pos['ra']['max_value'],  # 最大值
                    int(not pos['ra']['free'])  # 是否可调
                ],
                "dec": [
                    pos['dec']['value'],  # 当前值
                    pos['dec']['min_value'],  # 最小值
                    pos['dec']['max_value'],  # 最大值
                    int(not pos['dec']['free'])  # 是否可调
                ]
            }
            
        elif 'Gaussian_on_sphere' in src_data:
            gauss = src_data['Gaussian_on_sphere']
            src_model["MorModel"] = {
                "type": "Ext_gaus",
                "ra": [
                    gauss['lon0']['value'],  # 当前值
                    gauss['lon0']['min_value'],  # 最小值
                    gauss['lon0']['max_value'],  # 最大值
                    int(not gauss['lon0']['free'])  # 是否可调
                ],
                "dec": [
                    gauss['lat0']['value'],  # 当前值
                    gauss['lat0']['min_value'],  # 最小值
                    gauss['lat0']['max_value'],  # 最大值
                    int(not gauss['lat0']['free'])  # 是否可调
                ],
                "sigma": [
                    gauss['sigma']['value'],  # 当前值
                    gauss['sigma']['min_value'],  # 最小值
                    gauss['sigma']['max_value'],  # 最大值
                    int(not gauss['sigma']['free'])  # 是否可调
                ]
            }


        # 添加到SRC部分
        lhaaso_model["SRC"][src_key] = src_model
        src_index += 1
    
    if save:
        # 保存到文件
        save_lhaaso_model(lhaaso_model, f"../res/{region_name}/{Modelname}/Model_hsc.yaml")
        # import yamA
        # with open(f"../res/{region_name}/{Modelname}/Model_hsc.yaml", "w") as f:
        #     yaml.dump(lhaaso_model, f, default_flow_style=False, sort_keys=False, indent=2)

    return lhaaso_model


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
        Nc = float(scconfig['SEDModel']["F0"][4])
        index =  -scconfig['SEDModel']['alpha'][0]
        Indexb =  [-scconfig['SEDModel']['alpha'][2], -scconfig['SEDModel']['alpha'][1]]
        indexf = scconfig['SEDModel']['alpha'][3]
        if scconfig['SEDModel']['type'] == 'LP':
            beta = scconfig['SEDModel']['beta'][0]
            betab = [scconfig['SEDModel']['beta'][1], scconfig['SEDModel']['beta'][2]]
            betaf = scconfig['SEDModel']['beta'][3]
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
            if scconfig["MorModel"]['type'] == 'Ext_gaus'and sf:
                sbl = None
                sbh = None
            elif scconfig["MorModel"]['type'] == 'Ext_gaus':
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

        

        kbs = f"({kbl},{kbh})" if kbl is not None else "None"
        indexbs = f"({indexel},{indexeh})" if indexel is not None else "None"
        
        if doit:
            try:
                log.info(f"Spec: \n K={flux*Nc:.2e} kb=({kbl:.2e}, {kbh:.2e}) index={-index:.2f} indexb=({indexel:.2f},{indexeh:.2f})")
            except:
                pass
            if scconfig['SEDModel']['type'] == 'LP':
                if sigma is not None:
                    sbs = f"({sbl},{sbh})" if sbl is not None else "None"
                    try:
                        log.info(f"Mor: \n sigma={sigma:.2f} sb=({sbl:.2f},{sbh:.2f})")
                    except:
                        pass
                    prompt = f"""{name} = setsorce("{name}", {ras}, {decs}, sigma={sigma}, sb={sbs}, raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={flux*Nc}, kb={kbs}, alpha={-index}, alphab={indexbs}, rab=({rab[0]},{rab[1]}), decb=({decb[0]},{decb[1]}), kf={kf}, alphaf={indexf}, beta={beta}, betab=({betab[0]},{betab[1]}), betaf={betaf}, spec=Log_parabola())
lm.add_source({name})
                    """
                    exec(prompt)
                else:
                    log.info(f"Mor: ")
                    prompt = f"""{name} = setsorce("{name}", {ras}, {decs}, raf={pf}, decf={pf}, piv={piv},
        k={flux*Nc}, kb={kbs}, alpha={-index}, alphab={indexbs}, rab=({rab[0]},{rab[1]}), decb=({decb[0]},{decb[1]}), kf={kf}, alphaf={indexf}, beta={beta}, betab=({betab[0]},{betab[1]}), betaf={betaf}, spec=Log_parabola())
lm.add_source({name})
                    """
                    exec(prompt)

            else:
                if sigma is not None:
                    sbs = f"({sbl},{sbh})" if sbl is not None else "None"
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
        try:
            par = Model.parameters[p]
            if par.free:
                freepars.append("%-45s %35.6g ± %2.6g %s" % (p, par.value, result[0]["error"][p], par._unit))
            else:
                fixedpars.append("%-45s %35.6g %s" % (p, par.value, par._unit))
        except:
            continue

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


def create_high_dim_matrix(array_list):
    # 获取输入数组的个数（维度数）和每个数组的长度（每维的大小）
    dimensions = [len(arr) for arr in array_list]
    num_arrays = len(array_list)
    
    # 创建一个高维矩阵，初始可以用任意值填充，这里用 None
    matrix = np.full(dimensions, None, dtype=object)
    
    # 遍历矩阵的每个位置，填充对应的元素组合
    for idx in np.ndindex(*dimensions):
        # 对于当前坐标 idx，取出每个数组对应位置的元素组成列表
        element = [array_list[i][idx[i]] for i in range(num_arrays)]
        matrix[idx] = element
    
    return matrix

def reconstruct_high_dim_array(flat_indices, flat_values, original_shape):
    # 创建一个与原始形状相同的数组，初始填充任意值（这里用 0）
    reconstructed_array = np.zeros(original_shape, dtype=flat_values.dtype)
    
    # 将 flat_values 填入对应的 flat_indices 位置
    for idx, val in zip(flat_indices, flat_values):
        reconstructed_array[tuple(idx)] = val
    
    return reconstructed_array

from itertools import combinations

def custom_corner_plot(high_dim_array, axis_values, labels=None):
    # 获取数组的维度和形状
    ndim = high_dim_array.ndim
    shape = high_dim_array.shape
    
    # 检查 axis_values 的长度是否匹配维度数
    if len(axis_values) != ndim:
        raise ValueError("axis_values 的长度必须与 high_dim_array 的维度数匹配")
    
    # 检查每个维度的坐标轴值长度是否匹配形状
    for i, ax in enumerate(axis_values):
        if len(ax) != shape[i]:
            raise ValueError(f"第 {i} 维的 axis_values 长度 ({len(ax)}) 与形状 ({shape[i]}) 不匹配")
    
    # 如果没有提供标签，自动生成
    if labels is None:
        labels = [f"Dim{i+1}" for i in range(ndim)]
    
    # 找到整个数组的最小值及其索引
    min_value = np.min(high_dim_array)
    min_index = np.unravel_index(np.argmin(high_dim_array), shape)
    min_params = {labels[i]: axis_values[i][min_index[i]] for i in range(ndim)}
    
    # 打印最小值和对应的参数
    print(f"高维数组的最小值: {min_value}")
    print("对应的参数:")
    iii = 0
    minvas = {}
    for label, value in min_params.items():
        print(f"  {label}: {value}")
        minvas[iii] = value
        iii += 1
    
    # 创建一个 n x n 的子图网格
    fig, axes = plt.subplots(ndim, ndim, figsize=(ndim * 3, ndim * 3))
    
    # 对角线和下三角的所有二维组合
    for i, j in combinations(range(ndim), 2):
        # 固定第 i 和第 j 维，其他维度取最小值
        min_axes = tuple(k for k in range(ndim) if k != i and k != j)
        projection = np.min(high_dim_array, axis=min_axes)
        
        # 根据 i, j 的顺序调整投影数据和坐标轴
        # if i < j:
        proj_data = projection
        x_idx, y_idx = i, j
        # else:
        #     proj_data = projection.T
        #     x_idx, y_idx = j, i
        
        # 下三角：绘制二维 contour
        ax = axes[y_idx, x_idx]
        X, Y = np.meshgrid(axis_values[x_idx], axis_values[y_idx])
        try:
            proj_min_idx = np.unravel_index(np.argmin(proj_data), proj_data.shape)
            min_x = axis_values[x_idx][proj_min_idx[1]]
            if minvas[x_idx] == min_x:
                ax.contourf(X, Y, proj_data)
                contour = ax.contour(X, Y, proj_data)
            else:
                proj_data = proj_data.T
                proj_min_idx = np.unravel_index(np.argmin(proj_data), proj_data.shape)
                min_x = axis_values[x_idx][proj_min_idx[1]]                
                ax.pcolormesh(axis_values[x_idx], axis_values[y_idx], proj_data)
                # ax.contourf(X, Y, proj_data)    
                # contour = ax.contour(X, Y, proj_data)            
                
        except Exception as e:
            print(e)
            proj_data = proj_data.T
            ax.pcolormesh(axis_values[x_idx], axis_values[y_idx], proj_data)
            # contour = ax.contour(X, Y, proj_data)
            # ax.contourf(X, Y, proj_data)    
            proj_min_idx = np.unravel_index(np.argmin(proj_data), proj_data.shape)
        # ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel(labels[x_idx])
        ax.set_ylabel(labels[y_idx])
        
       
        min_x = axis_values[x_idx][proj_min_idx[1]]  # 注意：contour 的 x 是第 1 个索引
        min_y = axis_values[y_idx][proj_min_idx[0]]  # y 是第 0 个索引
        ax.scatter([min_x], [min_y], marker='*', color='red', s=100, label='Min')
    
    # 对角线：绘制一维分布，使用 axis_values 作为 x 轴
    for i in range(ndim):
        ax = axes[i, i]
        other_axes = tuple(k for k in range(ndim) if k != i)
        proj_1d = np.min(high_dim_array, axis=other_axes)
        ax.plot(axis_values[i], proj_1d)
        ax.set_xlabel(labels[i])
        
        # 找到一维投影的最小值位置并标注星标
        min_idx_1d = np.argmin(proj_1d)
        min_x_1d = axis_values[i][min_idx_1d]
        min_y_1d = proj_1d[min_idx_1d]
        ax.scatter([min_x_1d], [min_y_1d], marker='*', color='red', s=100, label='Min')
    
    # 上三角留空
    for i in range(ndim):
        for j in range(i + 1, ndim):
            axes[i, j].axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

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

def hDparscan(pars, nums, GRBlike, altr_hypo):
    trails = []
    for i,par in enumerate(pars):
        pmin = altr_hypo.parameters[par].min_value
        pmax = altr_hypo.parameters[par].max_value
        num = 100
        trail = np.linspace(pmin, pmax, nums[i])
        trails.append(trail)
    trails2 = create_high_dim_matrix(trails)
    # new_array = np.zeros(original_shape)

    flat_indices = np.array(list(np.ndindex(trails2.shape)))
    flat_values = trails2.ravel()
    result = np.zeros((len(flat_values)))

    for i,values in enumerate(tqdm(flat_values)):
        for j,par in enumerate(pars):
            altr_hypo.parameters[par].value = values[j]
        GRBlike.set_model(altr_hypo)
        llh = GRBlike.get_log_like()
        result[i] = llh

    result = reconstruct_high_dim_array(flat_indices, result, trails2.shape)
    pars = [it.split(".")[-1] for it in pars]
    custom_corner_plot(-result,trails,pars)
    return result, trails

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

def getTSall(TSlist, region_name, Modelname, result, WCDAs):
    """
        获取TS值

        Parameters:
            TSlist: 想要获取TS得source名称

        Returns:
            >>> 总的TS, TSresults(Dataframe)
    """ 
    if not isinstance(WCDAs, list):
        WCDAs = [WCDAs]
    TS = {}
    TS["TS_all"] = 0
    TS["-log(likelihood)"] = 0
    for sc in tqdm(TSlist):
        TS[sc]=0
    for WCDA in WCDAs:
        
        TS_all = WCDA.cal_TS_all()
        log.info(f"TS_all: {TS_all}")
        llh = WCDA.get_log_like()
        log.info(f"llh_all: {llh}")
        for sc in tqdm(TSlist):
            TS[sc]+=result[0].compute_TS(sc,result[1][1]).values[0][2]
            log.info(f"TS_{sc}: {TS[sc]}")
        
        TS["TS_all"] += TS_all
        TS["-log(likelihood)"] += -llh
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

def getresaccuracy(WCDA, lm, signif=17, smooth_sigma=0.3, alpha=3.24e-5, savepath=None, plot=False):
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
    if savepath[-1] != '/':
        savepath += '/'
    if savepath is not None:
        hp.write_map(savepath+"resufastmap.fits", resu, overwrite=True)

    new_source_idx = np.where(resu==np.ma.max(resu))[0][0]
    new_source_lon_lat=hp.pix2ang(1024,new_source_idx,lonlat=True)
    print(new_source_lon_lat)

    if plot:
        plt.figure()
        hp.gnomview(resu,norm='',rot=[new_source_lon_lat[0],new_source_lon_lat[1]],xsize=200,ysize=200,reso=6,title=savepath)
        plt.scatter(new_source_lon_lat[0],new_source_lon_lat[1],marker='x',color='red', label=f"{new_source_lon_lat}")
        plt.legend()
        plt.show()
        plt.savefig(f"{savepath}WCDA_res_init.png",dpi=300)
    return resu, new_source_lon_lat


def _smooth_and_scale_maps(on, off, smooth_sigma):
    """
    根据 getresaccuracy 中的方法对 ON/OFF 图进行平滑和缩放。
    """
    nside = hp.get_nside(on)
    npix = hp.nside2npix(nside)
    pixarea = 4 * np.pi / npix

    on = on.filled(hp.UNSEEN) if isinstance(on, np.ma.MaskedArray) else on
    off = off.filled(hp.UNSEEN) if isinstance(off, np.ma.MaskedArray) else off
    on[np.isnan(on)] = hp.UNSEEN
    off[np.isnan(off)] = hp.UNSEEN

    on1 = hp.sphtfunc.smoothing(on, sigma=np.radians(smooth_sigma))
    off1 = hp.sphtfunc.smoothing(off, sigma=np.radians(smooth_sigma))

    on2 = 1./(4.*np.pi*np.radians(smooth_sigma)**2) * (hp.sphtfunc.smoothing(on, sigma=np.radians(smooth_sigma/np.sqrt(2)))) * pixarea
    off2 = 1./(4.*np.pi*np.radians(smooth_sigma)**2) * (hp.sphtfunc.smoothing(off, sigma=np.radians(smooth_sigma/np.sqrt(2)))) * pixarea

    scale = (on1 + off1) / (on2 + off2 + 1e-20)
    
    ON = on1 * scale
    BK = off1 * scale
    
    return hp.ma(ON), hp.ma(BK)

def compute_significance_mapfast(on, bk, signif=17, alpha=3.24e-5):
    """
    从 ON 和 BK 图计算显著性图 (Li & Ma)。
    此函数可以正确处理 alpha 为标量或数组的情况。
    """
    on[np.isnan(on)] = hp.UNSEEN
    bk[np.isnan(bk)] = hp.UNSEEN
    on = hp.ma(on)
    bk = hp.ma(bk)

    epsilon = 1e-20
    on_safe = np.maximum(on, 0)
    bk_safe = np.maximum(bk, 0)

    if signif == 5:
        sig = (on - bk) / np.sqrt(on_safe + alpha * bk_safe + epsilon)
    elif signif == 9:
        sig = (on - bk) / np.sqrt(on_safe * alpha + bk_safe + epsilon)
    elif signif == 17:
        term1_ratio = (1. + alpha) / alpha * on_safe / (on_safe + bk_safe / alpha + epsilon)
        term2_ratio = (1. + alpha) * bk_safe / alpha / (on_safe + bk_safe / alpha + epsilon)
        
        log_term1 = on_safe * np.log(np.maximum(term1_ratio, epsilon))
        log_term2 = bk_safe / alpha * np.log(np.maximum(term2_ratio, epsilon))
        
        sig = np.sqrt(2 * np.maximum(log_term1 + log_term2, 0))
        sig[on < bk] *= -1
    else:
        sig = (on - bk) / np.sqrt(bk_safe + epsilon)
        
    sig.set_fill_value(0.0)
    sig[np.isnan(sig)] = 0.0
    return sig

WCDA_r68 = np.array([0.874, 0.657, 0.497, 0.396, 0.328, 0.259, 0.19])
KM2A_r68 = np.array([0.874, 0.657, 0.497, 0.43, 0.43, 0.36, 0.30, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.13])  # KM2A的r68值与WCDA相同
WCDA_r32 = WCDA_r68/1.51
KM2A_r32 = KM2A_r68/1.51
def get_residual_significance_mapfast(
    WCDA,
    lm=None,
    signif=17,
    smooth_sigma=WCDA_r32,  #0.3 KM2A_r68
    alpha=3.24e-5,  # <--- 修正1: 默认值设为None，以触发动态计算
    combine='sum',
    active_sources=None,
    plot=False,
    savepath=None
):
    """
    计算显著性图，支持加权合并、多bin显著性、以及自定义模型残差。
    *已修正：加入了与 getresaccuracy 对齐的平滑和缩放计算方法*

    combine: 决定是否对多能段进行合并
        - 'none'：返回每个能段独立显著性图 (各自平滑)
        - 'sum'：直接求和ON和BK再算显著性 (先求和后平滑，与getresaccuracy逻辑最一致)
        - 'weighted'：按(S/B)加权合并后再算显著性 (先加权平均后平滑)

    active_sources: (pta, exta)，控制哪些源被计入模型
    """
    nside = 1024
   
    # --- 修正2: 动态 ALPHA 计算逻辑 ---
    # 仅当alpha未被用户指定时，才进行动态计算，完美复现getresaccuracy的行为
    if alpha is None:
        # 这个计算需要像素的坐标theta，与getresaccuracy完全一致
        print("Alpha is None. Calculating dynamically based on pixel position...")
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) # theta是colatitude
        # 严格复现getresaccuracy中的坐标转换
        theta_lat = np.pi/2 - theta # 转换为latitude
        # 避免在赤道附近(theta_lat=0)或极点附近出现问题
        sin_theta_lat = np.sin(theta_lat)
        # 防止除以零
        sin_theta_lat[sin_theta_lat == 0] = 1e-9 
        
        # 动态计算alpha数组
        final_alpha = 2 * 0.3 * 1.51 / 60. / sin_theta_lat
    else:
        # 如果用户传入了alpha值(例如标量3.24e-5)，则使用该值
        print(f"Using user-provided alpha: {alpha}")
        final_alpha = alpha
    # --- 修正结束 ---

    if lm:
        WCDA.set_model(lm)
    
    on_all = []
    bk_all = []

    npt = lm.get_number_of_point_sources() if lm else 0
    next = lm.get_number_of_extended_sources() if lm else 0

    for i, bin_id in enumerate(WCDA._active_planes):
        dmap, bkmap = WCDA._get_excess(WCDA._maptree[bin_id], all_maps=True)[1:]

        model_map_val = 0.0
        if lm:
            # 模型计算部分保持不变
            if active_sources:
                pta, exta = active_sources
                model_map = WCDA._get_model_map(bin_id, bin_id, 0, 0).as_dense()
                for idx, keep in enumerate(pta):
                    if not keep:
                        model_map += WCDA._get_model_map(bin_id, bin_id, idx+1, 0).as_dense()
                        if idx != 0:
                            model_map -= WCDA._get_model_map(bin_id, bin_id, idx, 0).as_dense()
                for idx, keep in enumerate(exta):
                    if not keep:
                        model_map += WCDA._get_model_map(bin_id, bin_id, 0, idx+1).as_dense()
                        if idx != 0:
                            model_map -= WCDA._get_model_map(bin_id, bin_id, 0, idx).as_dense()
                model_map_val = model_map
            else:
                model_map_val = WCDA._get_model_map(bin_id, npt, next).as_dense()
        
        on = dmap
        bk = bkmap + model_map_val
        on[np.isnan(on)]=hp.UNSEEN
        bk[np.isnan(bk)]=hp.UNSEEN
        on=hp.ma(on)
        bk=hp.ma(bk)
        on_all.append(on)
        bk_all.append(bk)

    if isinstance(smooth_sigma, np.ndarray):
        planes = [int(it) for it in WCDA._active_planes]
        smooth_sigma = smooth_sigma[planes]

    resu_all = []
    if combine == 'sum':
        on_total = np.ma.sum(on_all, axis=0)
        bk_total = np.ma.sum(bk_all, axis=0)
        if isinstance(smooth_sigma, np.ndarray):
            smooth_sigma = np.sqrt(np.mean(smooth_sigma**2))
        ON_smooth, BK_smooth = _smooth_and_scale_maps(on_total, bk_total, smooth_sigma)
        # --- 修正3: 传递正确的alpha ---
        resu = compute_significance_mapfast(ON_smooth, BK_smooth, signif=signif, alpha=final_alpha)
        resu_all = resu
    # (其他combine模式的逻辑保持不变，但同样会受益于正确的alpha处理)
    elif combine == 'weighted':
        weights = [((np.ma.sum(on) - np.ma.sum(bk)) / (np.ma.sum(bk))) for on, bk in zip(on_all, bk_all)]
        weights = np.array(weights/np.sum(weights))
        if isinstance(smooth_sigma, np.ndarray):
            smooth_sigma = np.sqrt(np.mean(weights**2*smooth_sigma**2)/np.sum(weights)**2)
            print(smooth_sigma)
        # weighted_on = np.ma.average(np.array(on_all), axis=0, weights=weights)*len(weights)
        # weighted_bk = np.ma.average(np.array(bk_all), axis=0, weights=weights)*len(weights)
        # ON_smooth, BK_smooth = _smooth_and_scale_maps(weighted_on, weighted_bk, smooth_sigma)
        # resu = compute_significance_mapfast(ON_smooth, BK_smooth, signif=signif, alpha=final_alpha)
        # resu_all = resu
        on_all = np.ma.array(on_all)
        bk_all = np.ma.array(bk_all)

        net_signal_stack = on_all - bk_all
        sum_weighted_net_signal = np.ma.sum(weights[:, np.newaxis] * net_signal_stack, axis=0)

        # 计算 V_w = Σ [ w_i² * (N_on_i + α * BKG_i) ]
        variance_term_stack = on_all + alpha * bk_all
        sum_weighted_variance = np.ma.sum((weights**2)[:, np.newaxis] * variance_term_stack, axis=0)

        # --- 步骤2: 求解方程组得到等效的 ON 和 OFF 计数图 ---
        epsilon = 1e-30  # 一个非常小的数，防止除以零
        denominator = alpha**2 + alpha
        
        # N_off_w = (V_w - S_net) / (α² + α)
        # off_w 是等效的 OFF 区域计数，它将作为 'bk' 参数传入下一函数
        off_w = (sum_weighted_variance - sum_weighted_net_signal) / (denominator + epsilon)
        
        # N_on_w = S_net + α * N_off_w
        on_w = sum_weighted_net_signal + alpha * off_w
        
        # 物理上，计数值不能为负
        on_w = np.ma.maximum(on_w, 0)
        off_w = np.ma.maximum(off_w, 0)
        bk_w = alpha*off_w
        on_w.fill_value = hp.UNSEEN
        bk_w.fill_value = hp.UNSEEN
        on_w=hp.ma(on_w)
        bk_w=hp.ma(bk_w)
        # print("已计算出等效的加权 ON/OFF 计数图，正在计算最终显著性...")
        
        # --- 步骤3: 使用等效计数图计算最终的显著性 ---
        # 注意：我们将计算出的 `off_w` 作为 `bk` 参数传入
        ON_smooth, BK_smooth = _smooth_and_scale_maps(on_w, bk_w, smooth_sigma)
        resu = compute_significance_mapfast(ON_smooth, BK_smooth, signif=signif, alpha=alpha)
        resu_all = resu
    elif combine == 'none':
        for on, bk in zip(on_all, bk_all):
            ON_smooth, BK_smooth = _smooth_and_scale_maps(on, bk, smooth_sigma)
            sig = compute_significance_mapfast(ON_smooth, BK_smooth, signif=signif, alpha=final_alpha)
            resu_all.append(sig)

    # (文件保存和绘图逻辑保持不变)
    # --- 文件保存和绘图逻辑保持不变 ---
    if savepath:
        if savepath[-1] != '/':
            savepath += '/'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        # healpy.write_map可以接受list of maps
        hp.write_map(savepath+"fastmap.fits", resu_all, overwrite=True)

    if plot:
        resu_to_plot = resu_all
        if isinstance(resu_all, np.ndarray):
            n = len(resu_all)
            if n == 0:
                print("No maps to plot.")
                return resu_all
            
            # 如果是列表，默认只画第一张或者合并后的图（取决于combine模式）
            # 如果想画所有，可以用下面的循环
            print(f"Plotting results for {len(resu_all)} bins.")
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))
            plt.figure(figsize=(ncols * 5, nrows * 4))
            for i, m in enumerate(resu_all):
                plt.subplot(nrows, ncols, i + 1)
                hp.mollview(m, title=f"bin {i}", sub=(nrows, ncols, i + 1))
            plt.tight_layout()
            plt.show()
        else:
            # 适用于 combine='sum' 或 'weighted'
            resu = resu_all
            new_source_idx = np.where(resu == np.ma.max(resu))[0][0]
            new_source_lon_lat = hp.pix2ang(1024, new_source_idx, lonlat=True)
            
            plt.figure(figsize=(8, 8))
            hp.gnomview(resu, norm='', rot=[new_source_lon_lat[0], new_source_lon_lat[1]], xsize=200, ysize=200, reso=6,
                        title=(savepath or "Residual Significance Map"))
            # 标记最亮点的代码可以加在这里
            # hp.projtext(new_source_lon_lat[0], new_source_lon_lat[1], 'X', lonlat=True, color='red')
            print(f"Hottest spot at (lon, lat): {new_source_lon_lat}")
            plt.show()

    return resu_all


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
    TS_all=[]
    lm=Model()
    lon_array=[]
    lat_array=[]
    Modelname="Original"
    smooth_sigma=0.3
    bestmodelname="Original"
    
    tDGE=""

    if detector=="WCDA":
        kbs=(1e-15, 1e-11)
        indexbs=(-4, -1)
        kb=(1e-18, 1e-10)
        indexb=(-4.5, -0.5)
    else:
        kbs=(1e-18, 1e-13)
        indexbs=(-5.5, -1.5)
        kb=(1e-18, 1e-13)
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
        if detector=="WCDA":
            piv=3
        else:
            piv=50
        if freeDGE:
            tDGE="_DGE_free"
            Diffuse = set_diffusebkg(
                            ra1, dec1, model_radius, model_radius,
                            Kf=False, indexf=False, indexb=indexb,
                            name = region_name, Kb=kb, piv=piv
                            )
        else:
            tDGE="_DGE_fix"
            Diffuse = set_diffusebkg(
                            ra1, dec1, model_radius, model_radius,
                            file=DGEfile, piv=piv,
                            name = region_name
                            )
        lm.add_source(Diffuse)
        exts.append(Diffuse)

    sources = get_sources(lm)
    if detector=="WCDA":
        map2, skymapHeader = hp.read_map("../../data/fullsky_WCDA_20240131_2.6.fits.gz",h=True)
    else:
        map2, skymapHeader = hp.read_map("../../data/fullsky_KM2A_20240131_3.5.fits.gz",h=True)
    map2 = maskroi(map2, roi)
    fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat=cat, color="Fermi", savename="Oorg")
    plt.show()

    bestmodel=copy.deepcopy(lm)
    bestresult = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
    bestresultc = copy.deepcopy(bestresult)
    TS, TSdatafram = getTSall([], region_name+"_iter", Modelname, bestresult, WCDA)
    TSorg = TS["TS_all"]
    TS_all.append(TS["TS_all"])
    sources = get_sources(lm,bestresult)
    sources.pop("Diffuse")
    if detector=="WCDA":
        map2, skymapHeader = hp.read_map("../../data/fullsky_WCDA_20240131_2.6.fits.gz",h=True)
    else:
        map2, skymapHeader = hp.read_map("../../data/fullsky_KM2A_20240131_3.5.fits.gz",h=True)
    map2 = maskroi(map2, roi)
    fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat=cat, color="Fermi")
    plt.show()

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
        plt.savefig(f"../res/{region_name}_iter/{Modelname}_{N_src}.png",dpi=300)
        plt.show()

        if not ifnopt:
            npt+=1
            name=f"pt{npt}"
            bestmodelnamec=copy.copy(Modelname)
            pt = setsorce(name,lon_array[N_src],lat_array[N_src], 
                        indexb=indexbs,kb=kbs,
                        fitrange=data_radius)
            lm.add_source(pt)
            bestcache=copy.deepcopy(lm)
            Modelname=f"{npt}pt+{next}ext"+tDGE
            lm.display()
            ptresult = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
            TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, ptresult, WCDA)

            # if TS["TS_all"]-TSorg<25:
            #     log.info("worst than original, stop!")
            #     return bestmodel,bestresult
            
            TS_all.append(TS["TS_all"])
            TS_allpt = TS["TS_all"]

            sources = get_sources(lm,ptresult)
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
            ext = setsorce(name,lon_array[N_src],lat_array[N_src], sigma=0.1, sb=(0,5),
                        indexb=indexbs,kb=kbs,
                        fitrange=data_radius)
        lm.add_source(ext)
        source.append(ext)
        lm.display()
        result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
        TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)
        if TS["TS_all"]-TSorg<25:
            log.info("worst than original, stop!")
            return bestmodel,bestresult

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
                bestresultc = copy.deepcopy(result)
                bestcache=copy.deepcopy(lm)
                bestmodelnamec=copy.copy(Modelname)
                TS_all[-1]=TS["TS_all"]
                exts.append(ext)
            else:
                deltaTS = TS["TS_all"]-TS_all[-1]
                log.info(f"pt is better!! deltaTS={deltaTS:.2f}")
                npt+=1
                next-=1
                Modelname=f"{npt}pt+{next}ext"+tDGE
                bestresultc = copy.deepcopy(ptresult)
                lm = copy.deepcopy(bestcache)
                WCDA.set_model(lm)
                pts.append(pt)
                # lm.remove_source(name)
                # lm.add_source(pts[-1])
                name=f"pt{npt}"
                # result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini="ROOT")
                # TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)
                TS_all[-1]=TS_allpt #TS["TS_all"]
                source[-1]=pt
        else:
            bestcache=copy.deepcopy(lm)
            bestresultc = copy.deepcopy(result)
            bestmodelnamec=copy.copy(Modelname)
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
            bestmodelname= copy.copy(bestmodelnamec)
            bestresult = copy.deepcopy(bestresultc)
            bestmodel= copy.deepcopy(bestcache)
        else:
            log.info(f"{bestmodelname} is better!! deltaTS={TS_all[N_src+1]-TS_all[N_src]:.2f}, no need for more!")
            bestmodel.display()
            sources = get_sources(bestmodel,bestresult)
            sources.pop("Diffuse")
            if detector=="WCDA":
                map2, skymapHeader = hp.read_map("../../data/fullsky_WCDA_20240131_2.6.fits.gz",h=True)
            else:
                map2, skymapHeader = hp.read_map("../../data/fullsky_KM2A_20240131_3.5.fits.gz",h=True)
            map2 = maskroi(map2, roi)
            fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat=cat, color="Fermi")
            plt.show()
            log.info(f"Best model is {bestmodelname}")
            return bestmodel,bestresult

def fun_Logparabola(x,K,alpha,belta,Piv):
    return K*pow(x/Piv,alpha-belta*np.log(x/Piv))

def fun_Powerlaw(x,K,index,piv):
    return K*pow(x/piv,index)

def set_diffusebkg(ra1, dec1, lr=6, br=6, K = None, Kf = False, Kb=None, index =-2.733, indexf = False, file=None, piv=3, name=None, ifreturnratio=False, Kn=None, indexb=None, setdeltabypar=True, kbratio=1000, spec=None, alpha=None, alphaf=None, alphab=None, beta=None, betaf=None, betab=None):
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
    with uproot.open("../../data/gll_dust.root") as root_file:
        root_th2d = root_file["gll_region"]
        
        # 获取直方图信息
        X_nbins = root_th2d.member("fXaxis").member("fNbins")
        Y_nbins = root_th2d.member("fYaxis").member("fNbins")
        X_min = root_th2d.member("fXaxis").member("fXmin")
        X_max = root_th2d.member("fXaxis").member("fXmax")
        Y_min = root_th2d.member("fYaxis").member("fXmin")
        Y_max = root_th2d.member("fYaxis").member("fXmax")
        
        X_size = (X_max - X_min) / X_nbins
        Y_size = (Y_max - Y_min) / Y_nbins
        
        # 使用 uproot 提取直方图数据并转置
        data = root_th2d.values(flow=False).T  # flow=False 表示不包含溢出桶
        
        lranges = lr
        branges = br
        l, b = edm2gal(ra1, dec1)
        branges += abs(b)
        ll = np.arange(l - lranges, l + lranges, X_size)
        bb = np.arange(-branges, branges, Y_size)
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
        file = f'../../data/Diffusedata/{name}_dust_bkg_template.fits'
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
            Diffusespec.K.delta = deltatimek*kk * fluxUnit

        if Kb is not None:
            Diffusespec.K.bounds=np.array(Kb) * fluxUnit
        else:
            Diffusespec.K.bounds=np.array((kk/kbratio,kbratio*kk)) * fluxUnit
        Diffusespec.Kn = Kn
        Diffusespec.Kn.fix = True
    else:
        if spec is not None:
            Diffusespec = spec
        else:
            Diffusespec = Powerlaw()
        Diffuseshape = SpatialTemplate_2D(fits_file=file)
        Diffuse = ExtendedSource("Diffuse",spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
        Diffusespec.K = K * fluxUnit
        Diffusespec.K.fix=Kf
        if setdeltabypar:
            Diffusespec.K.delta = deltatimek*K * fluxUnit
        if Kb is not None:
            Diffusespec.K.bounds=np.array(Kb) * fluxUnit
        else:
            Diffusespec.K.bounds=np.array((K/kbratio,kbratio*K)) * fluxUnit



    Diffusespec.piv = piv * u.TeV
    Diffusespec.piv.fix=True

    if spec is None:
        Diffusespec.index = index
        Diffusespec.index.fix = indexf
        if indexb is not None:
            Diffusespec.index.bounds = indexb
        else:
            Diffusespec.index.bounds = (-4,-1)
    else:
        Diffusespec.alpha = alpha
        if alphaf is not None:
            Diffusespec.alpha.fix = alphaf
        if alphab is not None:
            Diffusespec.alpha.bounds = alphab
        Diffusespec.beta = beta
        if betaf is not None:
            Diffusespec.beta.fix = betaf
        if betab is not None:
            Diffusespec.beta.bounds = betab
    
    Diffuseshape.K = 1/u.deg**2
    if ifreturnratio:
        return Diffuse, [sa/0.00012671770357488944, ss, ss/2.745913003176557],
    else:
        return Diffuse

def set_diffusemodel(name, fits_file, K = 7.3776826e-13, Kf = False, Kb=None, index =-2.733, indexf = False, indexb = (-4,-1), piv=3, setdeltabypar=True, kbratio=1000, ratio=None, spec = Powerlaw(), be = 40, beb=(10,10000), index2=-3.3, index2b=(-4.5, -2.5), index2f=None):
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



    if  spec.name == "Cutoff_powerlaw" or spec.name == "Cutoff_powerlawM":
        Diffusespec.piv = piv * u.TeV
        Diffusespec.piv.fix=True
        Diffusespec.index = index
        Diffusespec.index.fix = indexf
        Diffusespec.index.bounds = indexb

    elif  spec.name == "SmoothlyBrokenPowerLaw" or spec.name == "SmoothlyBrokenPowerLawM":
        Diffusespec.pivot = piv * u.TeV
        Diffusespec.pivot.fix=True
        Diffusespec.alpha = index
        Diffusespec.alpha.fix = indexf
        Diffusespec.alpha.bounds = indexb
        Diffusespec.break_energy = be * u.TeV
        Diffusespec.break_energy.bounds = beb * u.TeV
        Diffusespec.beta = index2
        Diffusespec.beta.bounds = index2b
        Diffusespec.beta.fix = index2f
    else:
        Diffusespec.piv = piv * u.TeV
        Diffusespec.piv.fix=True
        Diffusespec.index = index
        Diffusespec.index.fix = indexf
        Diffusespec.index.bounds = indexb
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

def generate_random_coordinates(num_coords, galactic_latitude_limit=10):
    ra_list = []
    dec_list = []
    
    while len(ra_list) < num_coords:
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-20, 80)
        
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        galactic_latitude = coord.galactic.b.deg
        
        if abs(galactic_latitude) > galactic_latitude_limit:
            ra_list.append(ra)
            dec_list.append(dec)
    
    return ra_list, dec_list
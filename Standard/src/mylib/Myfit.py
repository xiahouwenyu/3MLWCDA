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
from Myspeedup import libdir

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

def set_all_parameters_free_or_fixed(model, free=True):
    """
    设置 astromodels 模型中所有参数为固定或自由状态
    
    参数:
    ----------
    model : astromodels.Model
        输入的 astromodels 模型
    free : bool, optional (默认=True)
        - 如果 True，放开所有参数（设置为自由状态）
        - 如果 False，固定所有参数（设置为固定状态）
    """
    # 遍历所有参数
    for param_name, param in model.parameters.items():
        param.free = free

from matplotlib import gridspec
def plot_all_model_maps(WCDA, lm, ra1, dec1, max_component_id=None, radius=10, reso=6):
    """
    绘制模型的所有组成部分（如点源、扩展源、diffuse等），并拼接为一张大图。

    Args:
        WCDA: 模型所属对象，需实现 _get_model_map 方法。
        lm: 模型管理器，需实现 get_number_of_point_sources 和 get_number_of_extended_sources 方法。
        ra1, dec1: 中心经度和纬度 (deg)
        max_component_id (int): 最大组件编号（如为None，则自动尝试遍历最多10个）
        xsize, ysize: 每幅图的像素大小
        reso: 每像素角分辨率（arcmin）

    Returns:
        fig: Matplotlib Figure 对象
    """
    if max_component_id is None:
        max_component_id = 10  # 默认尝试最多10个，直到失败为止

    xsize=radius*2*(60/reso)
    ysize=radius*2*(60/reso)

    maps = []
    valid_ids = []
    for i in range(max_component_id + 1):
        try:
            m = WCDA._get_model_map(str(i), lm.get_number_of_point_sources(), lm.get_number_of_extended_sources()).as_dense()
            maps.append(m)
            valid_ids.append(i)
        except Exception as e:
            print(f"Component {i} skipped: {e}")

    n_maps = len(maps)
    if n_maps == 0:
        raise ValueError("没有成功加载任何模型组件地图。")

    ncols = int(np.ceil(np.sqrt(n_maps)))
    nrows = int(np.ceil(n_maps / ncols))

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    spec = gridspec.GridSpec(nrows, ncols, figure=fig)

    for idx, (m, comp_id) in enumerate(zip(maps, valid_ids)):
        row = idx // ncols
        col = idx % ncols

        ax = fig.add_subplot(spec[row, col])
        hp.gnomview(
            m,
            rot=[ra1, dec1],
            xsize=xsize,
            ysize=ysize,
            reso=reso,
            norm='',
            title=f"Component {comp_id}",
            notext=True,
            cbar=False,
            hold=True,
            return_projected_map=False,
            sub=(nrows, ncols, idx + 1),
        )
        ax.set_title(f"Component {comp_id}", fontsize=10)

    plt.tight_layout()
    return fig

def find_source_id(lm, source_name: str) -> str:
    """
    使用 get_*_name 和 get_number_of_* 方法，判断一个源在模型中是第几个展源或点源。

    :param lm: 一个拥有 get_... 系列方法的分析模型对象。
    :param source_name: 要查询的源的名称。
    :return: 一个描述源身份的字符串。
    """
    
    # 首先，在点源中查找
    num_point_sources = lm.get_number_of_point_sources()
    for i in range(num_point_sources):
        current_name = lm.get_point_source_name(i)
        if current_name == source_name:
            # 索引 i 是从0开始的，所以序号是 i + 1
            return i, None

    # 如果不是点源，则在展源中查找
    num_extended_sources = lm.get_number_of_extended_sources()
    for i in range(num_extended_sources):
        current_name = lm.get_extended_source_name(i)
        if current_name == source_name:
            return None, i

    # 如果两个循环都结束了还没找到
    return None, None

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
        k={A0}, kb=({kbl}, {kbh}), alpha={alpha}, alphab=({indexel},{indexeh}), beta={beta}, betab=(0,3), betaf={indexf}, fitrange={rtp*pe}, kf={kf}, alphaf={indexf}, kn={Kscale}, setdeltabypar={setdeltabypar}, spec=Log_parabola())
lm.add_source({name})
                    """
                    exec(prompt)
                else:
                    log.info(f"Mor: fitrange={rtp*pe:.2f}")
                    prompt = f"""
{name} = setsorce("{name}", {ras}, {decs}, raf={pf}, decf={pf}, sf={sf}, piv={piv},
        k={A0}, kb=({kbl}, {kbh}), alpha={alpha}, alphab=({indexel},{indexeh}), beta={beta}, betab=(0,3), betaf={indexf}, fitrange={rtp*pe}, kf={kf}, alphaf={indexf}, kn={Kscale}, setdeltabypar={setdeltabypar}, spec=Log_parabola())
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
    """保存LHAASO模型为hsc格式，使用紧凑数组表示法"""
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

def convert_3ml_to_hsc(three_ml_model, region_name, Modelname, piv=50, save=True):
    """
    将3ML模型转换为与示例完全匹配的hsc格式
    
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
        save_lhaaso_model(lhaaso_model, f"{libdir}/../res/{region_name}/{Modelname}/Model_hsc.yaml")
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



def convert_xsq_to_3ml(input_file_path, output_file_path):
    """
    将 WCDA.yaml 文件转换为 Model_opt.yml 格式，确保标记点源和延展源，并保留 Diffuse 和 WCDA_bkg_renorm。
    
    Args:
        input_file_path (str): 输入 WCDA.yaml 文件路径
        output_file_path (str): 输出 Model_opt.yml 文件路径
    """
    import yaml
    # 加载 WCDA.yaml 文件
    with open(input_file_path, 'r') as file:
        wcda_data = yaml.safe_load(file)
    
    # 初始化输出字典
    converted_data = {}
    
    # 处理源信息
    source_dict = wcda_data.get('source_dict', {})
    for source_name, source_info in source_dict.items():
        spatial_model = source_info.get('spatial_model', {})
        spatial_type = spatial_model.get('spatial_type', '')
        sed_model = source_info.get('sed_model', {})
        
        # 通用光谱模型（Power Law 转换为 Log_parabola）
        spectrum = {
            'main': {
                'Log_parabola': {
                    'K': {
                        'value': sed_model['norm'][0] * float(sed_model['norm'][3]) * 1e-9,
                        'desc': 'Normalization',
                        'min_value': sed_model['norm'][0] * 0.1 * float(sed_model['norm'][3]) * 1e-9,
                        'max_value': sed_model['norm'][0] * 10 * float(sed_model['norm'][3]) * 1e-9,
                        'unit': 'keV-1 s-1 cm-2',
                        'is_normalization': True,
                        'delta': sed_model['norm'][1] * float(sed_model['norm'][3]) * 1e-9,
                        'free': True
                    },
                    'piv': {
                        'value': sed_model['E_0'] / 1e-9,
                        'desc': 'Pivot (keep this fixed)',
                        'unit': 'keV',
                        'is_normalization': False,
                        'delta': 0.1 / 1e-9,
                        'free': False
                    },
                    'alpha': {
                        'value': -sed_model['index'][0],  # Power Law 指数取负
                        'desc': 'index',
                        'min_value': -5.0,
                        'max_value': 0.0,
                        'unit': '',
                        'is_normalization': False,
                        'delta': 0.2,
                        'free': True
                    },
                    'beta': {
                        'value': 0.0,  # Power Law 的 beta 默认为 0
                        'desc': 'curvature',
                        'min_value': 0.0,
                        'max_value': 1.0,
                        'unit': '',
                        'is_normalization': False,
                        'delta': 0.1,
                        'free': False
                    }
                },
                'polarization': {}
            }
        }
        source_name = source_name.replace('+', 'P')  # 替换空格为下划线，确保兼容性
        source_name = source_name.replace('-', 'M')
        
        # 处理不同类型的源
        if source_name == 'iso_bg':
            # 等同于 WCDA_bkg_renorm 的背景源
            converted_data['WCDA_bkg_renorm (Parameter)'] = {
            "value": 1.0,
            "desc": "Renormalization for background map",
            "min_value": 0.5,
            "max_value": 1.5,
            "unit": '',
            "is_normalization": False,
            "delta": 0.01,
            "free": False
            }
        # elif source_name == 'gll_bg':
        #     # 等同于 Diffuse 的背景源
        #     converted_data['Diffuse (extended source)'] = {
        #         'spectrum': spectrum
        #     }
        elif spatial_type == 'ps':
            # 点源
            ra = spatial_model['ra'][0]
            dec = spatial_model['dec'][0]
            converted_data[source_name + " (point source)"] = {
                'position': {
                    'ra': {
                        'value': ra,
                        'desc': 'Right Ascension',
                        'min_value': ra - 0.5,
                        'max_value': ra + 0.5,
                        'unit': 'deg',
                        'is_normalization': False,
                        'delta': 0.1,
                        'free': True
                    },
                    'dec': {
                        'value': dec,
                        'desc': 'Declination',
                        'min_value': dec - 0.5,
                        'max_value': dec + 0.5,
                        'unit': 'deg',
                        'is_normalization': False,
                        'delta': 0.1,
                        'free': True
                    },
                    'equinox': 'J2000'
                },
                'spectrum': spectrum
            }
        elif spatial_type == 'gaussian':
            # 延展源
            lon0 = spatial_model['ra'][0]
            lat0 = spatial_model['dec'][0]
            sigma = spatial_model['ext'][0]
            converted_data[source_name + " (extended source)"] = {
                'Gaussian_on_sphere': {
                    'lon0': {
                        'value': lon0,
                        'desc': 'Longitude of center',
                        'min_value': lon0 - 0.5,
                        'max_value': lon0 + 0.5,
                        'unit': 'deg',
                        'is_normalization': False,
                        'delta': 0.1,
                        'free': True
                    },
                    'lat0': {
                        'value': lat0,
                        'desc': 'Latitude of center',
                        'min_value': lat0 - 0.5,
                        'max_value': lat0 + 0.5,
                        'unit': 'deg',
                        'is_normalization': False,
                        'delta': 0.1,
                        'free': True
                    },
                    'sigma': {
                        'value': sigma,
                        'desc': 'Standard deviation',
                        'min_value': max(sigma - 1, 0),
                        'max_value': sigma + 1,
                        'unit': 'deg',
                        'is_normalization': False,
                        'delta': 0.1,
                        'free': True
                    }
                },
                'spectrum': spectrum
            }
    
    # 将转换结果写入文件
    with open(output_file_path, 'w') as file:
        yaml.dump(converted_data, file, default_flow_style=False)

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
    

def fit(regionname, modelname, Detector,Model,s=None,e=None, mini = "minuit",verbose=False, savefit=True, ifgeterror=False, grids = None, donwtlimit=True, quiet=False, lmini = "minuit", ftol=1, max_function_calls = 500000, strategy=1, print_level=0):
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
    if not os.path.exists(f'{libdir}/../res/{regionname}/'):
        os.system(f'mkdir {libdir}/../res/{regionname}/')
    if not os.path.exists(f'{libdir}/../res/{regionname}/{modelname}/'):
        os.system(f'mkdir {libdir}/../res/{regionname}/{modelname}/')

    Model.save(f"{libdir}/../res/{regionname}/{modelname}/Model_init.yml", overwrite=True)
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
    elif isinstance(mini, str):
        # ROOTmini = threeML.minimizer.minimization.get_minimizer("ROOT")
        # print(ROOTmini.valid_setup_keys)
        if mini.lower() == "minuit":
            minuitmini = LocalMinimization(mini)
            minuitmini.setup(ftol=ftol, max_iter = max_function_calls, strategy=strategy, print_level=print_level) #, print_level=0
            jl.set_minimizer(minuitmini)
        elif mini.lower() == "root":
            rmini = LocalMinimization(mini)
            rmini.setup(ftol=ftol, max_function_calls = max_function_calls, strategy=strategy, verbosity=print_level) #, minimizer="Combined", verbosity=3, precision=1e-11
            jl.set_minimizer(rmini)
        else:
            jl.set_minimizer(mini)
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
        if not os.path.exists(f'{libdir}/../res/{regionname}/'):
            os.system(f'mkdir {libdir}/../res/{regionname}/')
        if not os.path.exists(f'{libdir}/../res/{regionname}/{modelname}/'):
            os.system(f'mkdir {libdir}/../res/{regionname}/{modelname}/')
        
        try:
            fig = Detector.display_fit(smoothing_kernel_sigma=0.25, display_colorbar=True)
            fig.savefig(f"{libdir}/../res/{regionname}/{modelname}/fit_result_{s}_{e}.pdf")
        except:
            pass
        Model.save(f"{libdir}/../res/{regionname}/{modelname}/Model.yml", overwrite=True)
        jl.results.write_to(f"{libdir}/../res/{regionname}/{modelname}/Results.fits", overwrite=True)
        jl.results.optimized_model.save(f"{libdir}/../res/{regionname}/{modelname}/Model_opt.yml", overwrite=True)
        with open(f"{libdir}/../res/{regionname}/{modelname}/Results.txt", "w") as f:
            f.write("\nFree parameters:\n")
            for l in freepars:
                f.write("%s\n" % l)
            f.write("\nFixed parameters:\n")
            for l in fixedpars:
                f.write("%s\n" % l)
            f.write("\nStatistical measures:\n")
            f.write(str(result[1].iloc[0])+"\n")
            f.write(str(jl.results.get_statistic_measure_frame().to_dict())+"\n")
            
        result[0].to_html(f"{libdir}/../res/{regionname}/{modelname}/Results_detail.html")
        result[0].to_csv(f"{libdir}/../res/{regionname}/{modelname}/Results_detail.csv")
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

def jointfit(regionname, modelname, Detector,Model,s=None,e=None,mini = "minuit",verbose=False, savefit=True, ifgeterror=False, grids=None, donwtlimit=True, quiet=False, lmini = "minuit", ftol=1, max_function_calls = 500000, strategy=1, print_level=0):
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
    if not os.path.exists(f'{libdir}/../res/{regionname}/'):
        os.system(f'mkdir {libdir}/../res/{regionname}/')
    if not os.path.exists(f'{libdir}/../res/{regionname}/{modelname}/'):
        os.system(f'mkdir {libdir}/../res/{regionname}/{modelname}/')

    Model.save(f"{libdir}/../res/{regionname}/{modelname}/Model_init.yml", overwrite=True)
    if s is not None and e is not None:
        for i in range(len(Detector)):
            Detector[i].set_active_measurements(s[i],e[i])
    datalist = DataList(*Detector)
    jl = JointLikelihood(Model, datalist, verbose=verbose)
    if mini == "grid":
        # Create an instance of the GRID minimizer
        grid_minimizer = GlobalMinimization("grid")

        # Create an instance of a local minimizer, which will be used by GRID
        local_minimizer = LocalMinimization(lmini)

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
        local_minimizer = LocalMinimization(lmini)

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
    elif isinstance(mini, str):
        # ROOTmini = threeML.minimizer.minimization.get_minimizer("ROOT")
        # print(ROOTmini.valid_setup_keys)
        if mini.lower() == "minuit":
            minuitmini = LocalMinimization(mini)
            minuitmini.setup(ftol=ftol, max_iter = max_function_calls, strategy=strategy, print_level=print_level) #, print_level=0
            jl.set_minimizer(minuitmini)
        elif mini.lower() == "root":
            rmini = LocalMinimization(mini)
            rmini.setup(ftol=ftol, max_function_calls = max_function_calls, strategy=strategy, verbosity=print_level) #, minimizer="Combined", verbosity=3, precision=1e-11
            jl.set_minimizer(rmini)
        else:
            jl.set_minimizer(mini)
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
        if not os.path.exists(f'{libdir}/../res/{regionname}/'):
            os.system(f'mkdir {libdir}/../res/{regionname}/')
        if not os.path.exists(f'{libdir}/../res/{regionname}/{modelname}/'):
            os.system(f'mkdir {libdir}/../res/{regionname}/{modelname}/')
        fig=[]
        for i in range(len(Detector)):
            try:
                fig.append(Detector[i].display_fit(smoothing_kernel_sigma=0.25, display_colorbar=True))
                fig[i].savefig(f"{libdir}/../res/{regionname}/{modelname}/fit_result_{s}_{e}.pdf")
            except:
                pass
        Model.save(f"{libdir}/../res/{regionname}/{modelname}/Model.yml", overwrite=True)
        jl.results.write_to(f"{libdir}/../res/{regionname}/{modelname}/Results.fits", overwrite=True)
        jl.results.optimized_model.save(f"{libdir}/../res/{regionname}/{modelname}/Model_opt.yml", overwrite=True)
        with open(f"{libdir}/../res/{regionname}/{modelname}/Results.txt", "w") as f:
            f.write("\nFree parameters:\n")
            for l in freepars:
                f.write("%s\n" % l)
            f.write("\nFixed parameters:\n")
            for l in fixedpars:
                f.write("%s\n" % l)
            f.write("\nStatistical measures:\n")
            f.write(str(result[1].iloc[0])+"\n")
            f.write(str(jl.results.get_statistic_measure_frame().to_dict())+"\n")
        result[0].to_html(f"{libdir}/../res/{regionname}/{modelname}/Results_detail.html")
        result[0].to_csv(f"{libdir}/../res/{regionname}/{modelname}/Results_detail.csv")
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

def process_yaml_file(file_path: str) -> None:
    """
    处理 YAML 文件，修改所有 fits_file.value 路径
    
    Args:
        file_path: YAML 文件路径
        libdir: 要添加的库目录路径
    """
    import yaml
    def process_value(path: str) -> str:
        """处理单个路径字符串"""
        # 处理 ../data/ 但缺少 Diffusedata 的情况
        if 'data/' in path and 'Diffusedata' not in path:
            parts = path.split('data/')
            path = f"{libdir}/../../data/Diffusedata/{parts[1]}"
        return path

    def traverse(node) -> None:
        """递归遍历 YAML 结构"""
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'fits_file' and isinstance(value, dict) and 'value' in value.keys():
                    value['value'] = process_value(value['value'])
                else:
                    traverse(value)
        elif isinstance(node, list):
            for item in node:
                traverse(item)

    # 读取 YAML 文件
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    # 处理所有嵌套的 fits_file
    traverse(data)

    # 保存修改后的 YAML 文件
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def find_and_update_fits_model_paths(
    input_fits_path: str, 
) -> None:
    """
    打开一个FITS文件，自动查找并根据指定逻辑修改HDU 1头部中的模型文件路径，
    并将结果保存到新的FITS文件中。

    它会查找所有以一个或多个 '../' 开头，后跟 'data/' 的路径。

    Args:
        input_fits_path (str): 输入的原始FITS文件路径。
        output_fits_path (str): 修改后要保存的新FITS文件路径。
        libdir (str): 用于构建新路径的库目录 (library directory)。
    """
    import re
    # 检查输入文件是否存在
    if not os.path.exists(input_fits_path):
        print(f"错误：输入文件不存在 -> '{input_fits_path}'")
        return
    output_fits_path = input_fits_path
    # 定义一个正则表达式来查找所有匹配的路径
    # 这个模式匹配: (任意数量的'../') + 'data/' + (文件名)
    # 文件名被假定为包含字母、数字、下划线、破折号和点
    path_pattern = re.compile(r"((?:\.\./)+data/[a-zA-Z0-9_./-]+\.fits)")

    try:
        # 使用 with 语句安全地打开 FITS 文件
        with fits.open(input_fits_path) as hdul:
            # 检查文件是否至少有两个HDU
            if len(hdul) < 2:
                print(f"错误：文件 '{input_fits_path}' 不包含至少两个HDU。")
                return

            # 获取第二个HDU的头部（索引为1）
            header = hdul[1].header

            # 检查 'MODEL' 关键字是否存在
            if 'MODEL' not in header:
                print(f"错误：在HDU 1中未找到 'MODEL' 关键字。")
                return
            
            # astropy会自动将所有 'CONTINUE' 卡片合并成一个完整的字符串
            model_string = header['MODEL']

            # 使用正则表达式查找所有匹配的路径
            # 使用 set 来存储找到的唯一路径，以防同一个路径在文件中出现多次
            found_paths = set(path_pattern.findall(model_string))

            if not found_paths:
                print("未找到符合'../../data/...'模式的路径。")
                # 如果没有找到匹配的路径，可以选择直接复制文件或不执行任何操作
                hdul.writeto(output_fits_path, overwrite=True)
                print(f"已将原始文件直接复制到 '{output_fits_path}'。")
                return

            print("在文件中找到以下需处理的路径:")
            
            # 对每个找到的唯一路径应用替换逻辑
            updated_model_string = model_string
            for original_path in found_paths:
                print(f"  - 正在处理: {original_path}")
                
                # 应用您的替换逻辑
                new_path = original_path # 默认为原始路径
                if 'data/' in original_path and 'Diffusedata' not in original_path:
                    parts = original_path.split('data/')
                    if len(parts) > 1:
                        # 使用 f-string 构建新路径
                        new_path = f"{libdir}/../../data/Diffusedata/{parts[1]}"
                        print(f"    -> 替换为: {new_path}")
                else:
                    print("    -> 路径不符合替换条件，保持不变。")

                # 在整个模型字符串中替换这个路径
                updated_model_string = updated_model_string.replace(original_path, new_path)
            
            # 更新头部中的 'MODEL' 关键字
            # astropy 会自动处理长字符串，并根据需要重新创建 'CONTINUE' 卡片
            header['MODEL'] = updated_model_string
            
            # 将修改后的HDUs保存到新文件
            hdul.writeto(output_fits_path, overwrite=True)
            
            print(f"\n成功更新所有找到的路径。")
            print(f"已将修改后的文件保存到: '{output_fits_path}'")

    except Exception as e:
        print(f"处理FITS文件时发生错误: {e}")

def load_modelpath(modelpath, changediffusedir=False):
    # silence_warnings()
    if modelpath[-1] == "/":
        modelpath = modelpath[:-1]
    if changediffusedir:
        find_and_update_fits_model_paths(modelpath+"/Results.fits")
        process_yaml_file(modelpath+"/Model_init.yml")
        process_yaml_file(modelpath+"/Model_opt.yml")
    try:
        results = load_analysis_results(modelpath+"/Results.fits")
    except Exception as e:
        print("No results found:", e)
        results = None

    try:
        lmini = load_model(modelpath+"/Model_init.yml")
    except Exception as e:
        print("No initial model found", e)
        lmini = None

    try:
        lmopt = load_model(modelpath+"/Model_opt.yml")
    except Exception as e:
        print("No optimized model found", e)
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
    TSresults.to_csv(f'{libdir}/../res/{region_name}/{Modelname}/Results.txt', sep='\t', mode='a', index=False)
    TSresults
    return TS, TSresults

def get_detector_params(detector):
    """根据探测器类型返回其参数配置"""
    if detector == "WCDA":
        kbs = (1e-15, 1e-11)
        indexbs = (-4, -1)
        kb = (1e-18, 1e-10)
        indexb = (-4.5, -0.5)
        piv = 3
        return kbs, indexbs, kb, indexb, piv
    elif detector == "KM2A":
        kbs = (1e-18, 1e-13)
        indexbs = (-5.5, -1.5)
        kb = (1e-18, 1e-13)
        indexb = (-5.5, -0.5)
        piv = 50
        return kbs, indexbs, kb, indexb, piv
    elif detector == "jf":
        kbs = (1e-15, 1e-11)
        indexbs = (-4, -1)
        kb = (1e-18, 1e-10)
        indexb = (-4.5, -0.5)
        piv = 20
        return kbs, indexbs, kb, indexb, piv
    else:
        raise ValueError("未知的探测器类型: %s" % detector)

def initialize_model(startfromfile, startfrommodel, fromcatalog, ra1, dec1, data_radius, model_radius, fixcatall, detector, rtsigma, rtflux, rtindex, rtp, ifext_mt_2):
    """根据输入参数初始化模型，并返回模型及源的计数"""
    lm = Model()
    if startfromfile:
        lm = load_model(startfromfile)
    elif startfrommodel:
        lm = startfrommodel
    elif fromcatalog:
        lm = getcatModel(ra1, dec1, data_radius, model_radius, fixall=fixcatall, detector=detector, rtsigma=rtsigma, rtflux=rtflux, rtindex=rtindex, rtp=rtp, ifext_mt_2=ifext_mt_2)
    
    # 计算初始的点源和展源数量
    npt = lm.get_number_of_point_sources()
    next = lm.get_number_of_extended_sources()
    if 'Diffuse' in lm.sources:
        next -= 1 # 弥散背景不计入展源计数

    return lm, npt, next

def add_diffuse(lm, ifDGE, freeDGE, indexb, kb, piv, ra1, dec1, model_radius, region_name, DGEk, DGEfile):
    """向模型中添加弥散背景"""
    if not ifDGE or 'Diffuse' in lm.sources:
        return lm, "", None
        
    if freeDGE:
        diffuse_source = set_diffusebkg(ra1, dec1, model_radius, model_radius, K=DGEk, Kf=False, indexf=False, indexb=indexb, name=region_name, Kb=kb, piv=piv)
        tag = "_DGE_free"
    else:
        diffuse_source = set_diffusebkg(ra1, dec1, model_radius, model_radius, file=DGEfile, piv=piv, name=region_name)
        tag = "_DGE_fix"
        
    lm.add_source(diffuse_source)
    return lm, tag, diffuse_source

def draw_model_map(region_name, model_name, sources, libdir, roi, ra1, dec1, data_radius, detector, cat, suffix=""):
    """绘制并保存当前模型的源分布图"""
    detectors = []
    if detector=="WCDA":
        detectors.append("WCDA")
        map_file = [f"{libdir}/../../data/sigmap_latest/fullsky_WCDA_20240731_0-6_2.6.fits.gz"]
    elif detector=="KM2A":
        detectors.append("KM2A")
        map_file = [f"{libdir}/../../data/sigmap_latest/fullsky_KM2A_20240731_4-13_3.6.fits.gz"]
    else:
        detectors.append("WCDA")
        detectors.append("KM2A")
        map_file = [f"{libdir}/../../data/sigmap_latest/fullsky_WCDA_20240731_0-6_2.6.fits.gz", f"{libdir}/../../data/sigmap_latest/fullsky_KM2A_20240731_4-13_3.6.fits.gz"]
    
    for map,det in zip(map_file, detectors):
        map2, _ = hp.read_map(map, h=True)
        map2 = maskroi(map2, roi)
        drawmap(region_name, model_name, sources, map2, ra1, dec1, rad=data_radius, contours=[10000], save=True, cat=cat, color="Fermi", savename=suffix+det)
    plt.show()

def get_maxres_lonlat(resu_map, nside=1024):
    """从残差图中找到最大值点的经纬度"""
    new_source_idx = np.where(resu_map == np.ma.max(resu_map))[0][0]
    lon, lat = hp.pix2ang(nside, new_source_idx, lonlat=True)
    return lon, lat

def plot_residual(resu_map, lon_array, lat_array, ra1, dec1, region_name, model_name, iter_num, libdir, radius=10, reso=6):
    """绘制并保存残差图"""

    
    plt.figure()
    hp.gnomview(resu_map, norm='', rot=[ra1, dec1], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso, title=model_name)
    plt.scatter(lon_array, lat_array, marker='x', color='red')
    
    res_dir = f'{libdir}/../res/{region_name}/'
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(f"{res_dir}/{model_name}_{iter_num}.png", dpi=300)
    plt.show()

def add_point_source(lm, name, lon, lat, indexb, kb, data_radius, piv, detector):
    """向模型中添加一个点源"""
    if detector == "jf":
        pt_source = setsorce(name, lon, lat, alphab=indexb, kb=kb, k=1e-15,fitrange=data_radius, piv=piv, spec=Log_parabola())
    else:
        pt_source = setsorce(name, lon, lat, indexb=indexb, kb=kb, fitrange=data_radius, piv=piv)
    lm.add_source(pt_source)
    return pt_source

def add_extended_source(lm, name, lon, lat, indexb, kb, data_radius, ifAsymm,  piv, detector):
    """向模型中添加一个展源（对称或非对称高斯）"""
    if detector == "jf":
        ext_source = setsorce(name, lon, lat, sigma=0.1, sb=(0,2), alphab=indexb,kb=kb, k=1e-15,fitrange=data_radius, piv=piv, spec=Log_parabola())
    else:
        if ifAsymm:
            ext_source = setsorce(name, lon, lat, a=0.1, ae=(0,2), e=0.1, eb=(0,1), theta=10, thetab=(-90,90), indexb=indexb, kb=kb, fitrange=data_radius, spat="Asymm",piv=piv)
        else:
            ext_source = setsorce(name, lon, lat, sigma=0.1, sb=(0,2), indexb=indexb, kb=kb, fitrange=data_radius,piv=piv)
    lm.add_source(ext_source)
    return ext_source

def log_TS(region_name, iter_num, ts_value, libdir):
    """记录每次迭代后的总TS值"""
    path = f'{libdir}/../res/{region_name}/{region_name}_TS.txt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 第一次迭代时用 "w" (写入), 之后用 "a" (追加)
    mode = "w" if iter_num == 1 else "a"
    with open(path, mode) as f:
        f.write("\nIter%d TS_total: %f\n" % (iter_num, ts_value))

# --- 重构后的主函数 ---

def Search(ra1, dec1, data_radius, model_radius, region_name, Mname, WCDA, roi, s, e,
           mini="ROOT", verbose=False, ifDGE=1, freeDGE=1, DGEk=None,
           DGEfile=f"{libdir}/../../data/G25_dust_bkg_template.fits", ifAsymm=False, ifnopt=True,
           startfromfile=None, startfrommodel=None, fromcatalog=False,
           cat={"TeVCat": [0, "s"], "PSR": [0, "*"], "SNR": [0, "o"], "3FHL": [0, "D"], "4FGL": [0, "d"]},
           detector="WCDA", fixcatall=False, extthereshold=9,
           rtsigma=8, rtflux=15, rtindex=2, rtp=8, ifext_mt_2=True):

    # 初始化
    pts, exts = [], []
    TS_all = []
    lon_array, lat_array = [] ,[]
    Modelname = f"{Mname}/Original"

    if not os.path.exists(f'{libdir}/../res/{region_name}/'):
        os.system(f'mkdir -p {libdir}/../res/{region_name}/')
    if not os.path.exists(f'{libdir}/../res/{region_name}/{Modelname}/'):
        os.system(f'mkdir -p {libdir}/../res/{region_name}/{Modelname}/')
    

    # 获取探测器参数
    kbs, indexbs, kb, indexb, piv = get_detector_params(detector)

    # 初始化模型
    lm, npt, next = initialize_model(startfromfile, startfrommodel, fromcatalog, ra1, dec1,
                                     data_radius, model_radius, fixcatall, detector,
                                     rtsigma, rtflux, rtindex, rtp, ifext_mt_2)

    if "Diffuse" not in [lm.get_extended_source_name(i) for i in range(lm.get_number_of_extended_sources())]:
        # 添加弥散背景
        lm, tDGE, diffuse_component = add_diffuse(lm, ifDGE, freeDGE, indexb, kb, piv, ra1, dec1, model_radius, region_name, DGEk, DGEfile)
        if diffuse_component:
            exts.append(diffuse_component)
    else:
        if freeDGE:
            tDGE = "_DGE_free"
        else:
            tDGE = "_DGE_fix"

    # 绘制初始模型图
    draw_model_map(region_name, Modelname, get_sources(lm), libdir, roi, ra1, dec1, data_radius * 2, detector, cat, "Oorg")

    lm.display(complete=True)
    # 首次拟合
    bestmodel = copy.deepcopy(lm)
    try:
        if detector == "jf":
            bestresult = jointfit(region_name, Modelname, WCDA, lm, s, e, mini=mini, verbose=verbose)
        else:
            bestresult = fit(region_name, Modelname, WCDA, lm, s, e, mini=mini, verbose=verbose)
    except Exception as e:
        log.error(f"拟合失败: {e}")
        return lm, None
    TS, _ = getTSall([], region_name, Modelname, bestresult, WCDA)
    TSorg = TS["TS_all"]
    TS_all.append(TSorg)

    # 绘制初始拟合后的模型图 (去除弥散背景)
    sources_no_diffuse = get_sources(lm, bestresult)
    sources_no_diffuse.pop("Diffuse", None)
    draw_model_map(region_name, Modelname, sources_no_diffuse, libdir, roi, ra1, dec1, data_radius * 2, detector, cat, Mname)

    bestmodelname = Modelname
    bestresultc = copy.deepcopy(bestresult)

    current_model_name = f"{Mname}/{npt}pt+{next}ext" + tDGE
    if not os.path.exists(f'{libdir}/../res/{region_name}/{current_model_name}/'):
        os.system(f'mkdir -p {libdir}/../res/{region_name}/{current_model_name}/')

    # 开始迭代搜索新源
    for N_src in range(100):
        # 计算残差图并找到最大值位置
        resu = getresaccuracy(WCDA, lm, plot=True, savepath=f'{libdir}/../res/{region_name}/',savename=f"{current_model_name}_residual.png", radius=data_radius)
        lon, lat = get_maxres_lonlat(resu)
        lon_array.append(lon)
        lat_array.append(lat)
        plot_residual(resu, lon_array, lat_array, ra1, dec1, region_name, Modelname, N_src, libdir)
        
        ptresult, TS_allpt = None, None
        
        # 步骤1: 尝试添加点源 (如果ifnopt为False)
        if not ifnopt:
            npt_temp = npt + 1
            pt_name = f"pt{npt_temp}"
            lm_cache_for_pt = copy.deepcopy(lm) # 备份当前模型
            pt = add_point_source(lm, pt_name, lon, lat, indexbs, kbs, data_radius, piv, detector)
            
            current_model_name = f"{Mname}/{npt_temp}pt+{next}ext" + tDGE
            if not os.path.exists(f'{libdir}/../res/{region_name}/{current_model_name}/'):
                os.system(f'mkdir -p {libdir}/../res/{region_name}/{current_model_name}/')
            try:
                if detector == "jf":
                    ptresult = jointfit(region_name, current_model_name, WCDA, lm, s, e, mini=mini, verbose=verbose)
                else:
                    ptresult = fit(region_name, current_model_name, WCDA, lm, s, e, mini=mini, verbose=verbose)
            except Exception as e:
                log.error(f"点源拟合失败: {e}")
                return lm, None
            TSpt, _ = getTSall([], region_name, current_model_name, ptresult, WCDA)
            TS_allpt = TSpt["TS_all"]
            lmpt = copy.deepcopy(lm) # 保存点源拟合后的模型

            # 绘制点源模型图
            sources_pt = get_sources(lm, ptresult)
            sources_pt.pop("Diffuse", None)
            draw_model_map(region_name, current_model_name, sources_pt, libdir, roi, ra1, dec1, data_radius*2, detector, cat)
            
            # 恢复模型以尝试展源
            lm = lm_cache_for_pt

        # 步骤2: 尝试添加展源
        next_temp = next + 1
        ext_name = f"ext{next_temp}"
        if ifnopt:
            current_model_name = f"{Mname}/{npt}pt+{next_temp}ext" + tDGE
        else: # 从点源尝试恢复了模型
            current_model_name = f"{Mname}/{npt}pt+{next_temp}ext" + tDGE
        if not os.path.exists(f'{libdir}/../res/{region_name}/{current_model_name}/'):
            os.system(f'mkdir -p {libdir}/../res/{region_name}/{current_model_name}/')
        
        ext = add_extended_source(lm, ext_name, lon, lat, indexbs, kbs, data_radius, ifAsymm, piv, detector)
        try:
            if detector == "jf":
                extresult = jointfit(region_name, current_model_name, WCDA, lm, s, e, mini=mini, verbose=verbose)
            else:
                extresult = fit(region_name, current_model_name, WCDA, lm, s, e, mini=mini, verbose=verbose)
        except Exception as e:
            log.error(f"展源拟合失败: {e}")
            return lm, None
        TSext, _ = getTSall([ext_name], region_name, current_model_name, extresult, WCDA)
        TS_allext = TSext["TS_all"]
        
        # 如果加入新源后TS提升小于25，则停止迭代
        current_max_ts = max(TS_allpt if TS_allpt is not None else -1, TS_allext)
        if current_max_ts - TS_all[0] < 25:
            log.info("新源贡献过小(TS<25)，停止迭代!")
            return bestmodel, bestresult

        # 绘制展源模型图
        sources_ext = get_sources(lm, extresult)
        sources_ext.pop("Diffuse", None)
        draw_model_map(region_name, current_model_name, sources_ext, libdir, roi, ra1, dec1, data_radius*2, detector, cat)

        # 步骤3: 比较点源和展源，确定本轮最佳模型
        if not ifnopt:
            # deltaTS 是展源模型和点源模型的TS差值
            deltaTS = TS_allext - TS_allpt
            if deltaTS >= extthereshold:
                log.info(f"展源更优!! deltaTS={deltaTS:.2f}")
                next += 1
                bestresultc = copy.deepcopy(extresult)
                bestmodelnamec = f"{npt}pt+{next}ext" + tDGE
                TS_all.append(TS_allext)
                exts.append(ext)
                # lm 已经是添加了展源的模型，无需改动
            else:
                log.info(f"点源更优!! deltaTS={deltaTS:.2f}")
                npt += 1
                # lm.remove_source(ext_name) # 移除刚添加的展源
                # pt = add_point_source(lm, f"pt{npt}", lon, lat, indexbs, kbs, data_radius, piv, detector) # 重新添加点源
                lm = lmpt
                WCDA.set_model(lm)
                bestresultc = copy.deepcopy(ptresult)
                bestmodelnamec = f"{npt}pt+{next}ext" + tDGE
                TS_all.append(TS_allpt)
                pts.append(pt)
        else: # 只考虑展源的情况
            next += 1
            bestresultc = copy.deepcopy(extresult)
            bestmodelnamec = f"{npt}pt+{next}ext" + tDGE
            TS_all.append(TS_allext)
            exts.append(ext)

        Modelname = bestmodelnamec # 更新当前最优的模型名
        log_TS(region_name, N_src + 1, TS_all[-1], libdir)
        
        # 步骤4: 判断本轮迭代是否显著提升，并决定是否继续
        if TS_all[N_src+1] - TS_all[N_src] > 25:
            log.info(f"模型 {bestmodelnamec} 更优!! deltaTS={TS_all[N_src+1] - TS_all[N_src]:.2f}")
            bestmodelname = bestmodelnamec
            bestresult = copy.deepcopy(bestresultc)
            bestmodel = copy.deepcopy(lm) # 保存当前模型状态
        else:
            log.info(f"模型 {bestmodelname} 已是最佳!! deltaTS={TS_all[N_src+1] - TS_all[N_src]:.2f}, 无需更多源!")
            bestmodel.display()
            final_sources = get_sources(bestmodel, bestresult)
            final_sources.pop("Diffuse", None)
            draw_model_map(region_name, "BestModel_"+bestmodelname, final_sources, libdir, roi, ra1, dec1, data_radius*2, detector, cat)
            log.info(f"最佳模型是 {bestmodelname}") 
            return bestmodel, bestresult
            
    return bestmodel, bestresult

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
    with uproot.open(f"{libdir}/../../data/gll_dust.root") as root_file:
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

        K = fun_Powerlaw(piv, K, index, 3)

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
        file = f'{libdir}/../../data/Diffusedata/{name}_dust_bkg_template.fits'
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
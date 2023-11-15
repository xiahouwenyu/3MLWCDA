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

from Mysigmap import *

#####   Model
def setsorce(name,ra,dec,raf=False,decf=False,rab=None,decb=None,
            sigma=None,sf=False,sb=None,radius=None,rf=False,rb=None,
            ################################ Spectrum
            k=1.3e-13,kf=False,kb=None,piv=3,pf=True,index=-2.6,indexf=False,indexb=None,alpha=-2.6,alphaf=False,alphab=None,beta=0,betaf=False,betab=None,

            fitrange=None,
            ################################ Continuous_injection_diffusion
            rdiff0=None, rdiff0f=False, rdiff0b=None, delta=None, deltaf=False, deltab=None,
            uratio=None, uratiof=False, uratiob=None,                          ##Continuous_injection_diffusion_legacy
            rinj=None, rinjf=True, rinjfb=None, b=None, bf = True, bb=None,                      ##Continuous_injection_diffusion
            incl=None, inclf=True, inclb=None, elongation=None, elongationf=True, elongationb=None,               ##Continuous_injection_diffusion_ellipse
            piv2=1, piv2f=True,

            ################################ Asymm Gaussian on sphere
            a=None, af=False, ab=None, e=None, ef=False, eb=None, theta=None, thetaf=False, thetab=None,
            
            spec=None,
            spat=None,
            *other,
            **kw):  # sourcery skip: extract-duplicate-method, low-code-quality
    """Create a Sources.

        Args:
            par: Parameters. if sigma is not None, is guassian,or rdiff0 is not None, is Continuous_injection_diffusion,such as this.
            parf: fix it?
            parb: boundary
        Returns:
            Source
    """
    
    fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)

    if spec is None:
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
    elif spat == "Asymm":
        spat=Asymm_Gaussian_on_sphere()
    elif spat == "Ellipse":
        spat=Ellipse_on_sphere()
    else:
        pass

    # print(spat.name)

    if sigma is None and rdiff0 is None and radius is None and a is None:
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
        source = ExtendedSource(name, spatial_shape=spat, spectral_shape=spec)


    def setspatParameter(parname,par,parf,parb,unit=""):
        nonlocal spat
        nonlocal spec
        prompt = f"""
if par != None:
    spat.{parname} = par {unit}
    spat.{parname}.fix = parf
if parb != None:
    spat.{parname}.bounds = parb {unit}
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
    spec.{parname}.bounds = parb {unit}
        """
        exec(prompt)

    #### set spectral
    spec.K = k * fluxUnit
    spec.K.fix = kf
    if kb != None:
        spec.K.bounds = kb * fluxUnit

    spec.piv = piv * u.TeV
    spec.piv.fix = pf

    if spec.name == "Log_parabola":
        setspecParameter("alpha",alpha,alphaf,alphab)
        setspecParameter("beta",beta,betaf,betab)
    elif spec.name == "Powerlaw" or spec.name == "PowerlawM":
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
        spat.sigma = sigma*u.degree
        spat.sigma.fix = sf
    if sb != None:
        spat.sigma.bounds = sb*u.degree

    
    setspatParameter("rdiff0",rdiff0,rdiff0f,rdiff0b,"* u.degree")
    setspatParameter("delta",delta,deltaf,deltab)
    setspatParameter("uratio",uratio,uratiof,uratiob)
    setspatParameter("b",b,bf,bb)
    setspatParameter("incl",incl,inclf,inclb)
    setspatParameter("elongation",elongation,elongationf,elongationb)
    setspatParameter("a",a,af,ab)
    setspatParameter("theta",theta,thetaf,thetab)
    setspatParameter("e",e,ef,eb)

    return source




def fit(regionname, modelname, WCDA,Model,s,e,mini = "minuit",verbose=False, savefit=True, ifgeterror=False):
    activate_progress_bars()
    WCDA.set_active_measurements(s,e)
    datalist = DataList(WCDA)
    jl = JointLikelihood(Model, datalist, verbose=verbose)
    if mini == "grid":
        # Create an instance of the GRID minimizer
        grid_minimizer = GlobalMinimization("grid")

        # Create an instance of a local minimizer, which will be used by GRID
        local_minimizer = LocalMinimization("minuit")

        # Define a grid for mu as 10 steps between 2 and 80
        my_grid = {Model.J0248.spatial_shape.lon0: np.linspace(Model.J0248.spatial_shape.lon0.value-2, Model.J0248.spatial_shape.lon0.value+2, 20), Model.J0248.spatial_shape.lat0: np.linspace(Model.J0248.spatial_shape.lat0.value-2, Model.J0248.spatial_shape.lat0.value+2, 10)}

        # Setup the global minimization
        # NOTE: the "callbacks" option is useless in a normal 3ML analysis, it is
        # here only to keep track of the evolution for the plot
        grid_minimizer.setup(
            second_minimization=local_minimizer, grid=my_grid #, callbacks=[get_callback(jl)]
        )

        # Set the minimizer for the JointLikelihood object
        jl.set_minimizer(grid_minimizer)
    elif mini == "PAGMO":
        _extracted_from_fit_30(jl)
    else:
        jl.set_minimizer(mini)

    result = jl.fit()

    freepars = []
    fixedpars = []
    for p in Model.parameters:
        par = Model.parameters[p]
        if par.free:
            freepars.append("%-45s %35.6g %s" % (p, par.value, par._unit))
        else:
            fixedpars.append("%-45s %35.6g %s" % (p, par.value, par._unit))

    if ifgeterror:
        result = list(result)
        result[0] = jl.get_errors()

    if savefit:
        time1 = strftime("%m-%d-%H", localtime())
        if not os.path.exists(f'../res/{regionname}/'):
            os.system(f'mkdir ../res/{regionname}/')
        if not os.path.exists(f'../res/{regionname}/{modelname}/'):
            os.system(f'mkdir ../res/{regionname}/{modelname}/')
        fig = WCDA.display_fit(smoothing_kernel_sigma=0.25, display_colorbar=True)
        fig.savefig(f"../res/{regionname}/{modelname}/fit_result_{s}_{e}.pdf")
        Model.save(f"../res/{regionname}/{modelname}/Model.yml", overwrite=True)
        jl.results.write_to(f"../res/{regionname}/{modelname}/Results.fits", overwrite=True)
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


# TODO Rename this here and in `fit`
def _extracted_from_fit_30(jl):
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

def getTSall(TSlist, region_name, Modelname, result, WCDA):
    TS = {}
    for sc in tqdm(TSlist):
        TS[sc]=result[0].compute_TS(sc,result[1][1]).values[0][2]
    llh = WCDA.get_log_like()
    TS_all = WCDA.cal_TS_all()
    TS["TS_all"] = TS_all
    TS["-log(likelihood)"] = -llh
    TSresults = pd.DataFrame([TS])
    TSresults.to_csv(f'../res/{region_name}/{Modelname}/Results.txt', sep='\t', mode='a', index=False)
    TSresults
    return TS, TSresults

def Search(ra1, dec1, data_radius, region_name, WCDA, s, e,  mini = "ROOT", ifDGE=1,fDGE=1,DGEk=1.8341549e-12,DGEfile="../../data/G25_dust_bkg_template.fits", ifAsymm=False, ifnopt=False, startfrom=None):
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

    if ifDGE:
        if fDGE:
            tDGE="_DGE_free"
            Diffuse = set_diffusebkg(
                            K = DGEk,
                            file=DGEfile,
                            Kf=False, indexf=False
                            )
        else:
            tDGE="_DGE_fix"
            Diffuse = set_diffusebkg(
                            K = DGEk,
                            file=DGEfile
                            )
        lm.add_source(Diffuse)
        exts.append(Diffuse)

    for N_src in range(100):
        data=np.zeros(1024*1024*12)
        bkg =np.zeros(1024*1024*12)
        model=np.zeros(1024*1024*12)
        for i, plane_id in enumerate(WCDA._active_planes):
            data_analysis_bin = WCDA._maptree[plane_id]
            if(N_src==0):
                this_model_map_hpx=np.zeros(1024*1024*12)
            else:
                this_model_map_hpx = WCDA._get_model_map(plane_id, npt, next+ifDGE).as_dense()
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
        resu=data-bkg-model
        resu=hp.sphtfunc.smoothing(resu,sigma=np.radians(smooth_sigma))

        new_source_idx = np.where(resu==np.ma.max(resu))[0][0]
        new_source_lon_lat=hp.pix2ang(1024,new_source_idx,lonlat=True)
        lon_array.append(new_source_lon_lat[0])
        lat_array.append(new_source_lon_lat[1])
        print(lon_array,lat_array)
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
                        indexb=(-4,-1),kb=(1e-16, 1e-10),
                        fitrange=data_radius)
            lm.add_source(pt)
            bestcache=copy.deepcopy(lm)
            Modelname=f"{npt}pt+{next}ext"+tDGE
            result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
            TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)
            TS_all.append(TS["TS_all"])
            pts.append(pt)

            sources = get_sources(lm,result)
            sources.pop("Diffuse")
            map2, skymapHeader = hp.read_map("../../data/signif_20210305_20230731_ihep_goodlist_nHit006_0.29.fits.gz.fits.gz",h=True)
            map2 = hp.ma(map2)
            fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat={ "TeVCat": [0, "s"],"PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"]}, color="Fermi")
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
                        indexb=(-4,-1),kb=(1e-16, 1e-10),
                        fitrange=data_radius, spat="Asymm")
        else:
            ext = setsorce(name,lon_array[N_src],lat_array[N_src], sigma=0.1, sb=(0,5),
                        indexb=(-4,-1),kb=(1e-16, 1e-10),
                        fitrange=data_radius)
        lm.add_source(ext)
        source.append(ext)
        result = fit(region_name+"_iter", Modelname, WCDA, lm, s, e,mini=mini)
        TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)

        sources = get_sources(lm,result)
        sources.pop("Diffuse")
        map2, skymapHeader = hp.read_map("../../data/signif_20210305_20230731_ihep_goodlist_nHit006_0.29.fits.gz.fits.gz",h=True)
        map2 = hp.ma(map2)
        fig = drawmap(region_name+"_iter", Modelname, sources, map2, ra1, dec1, rad=data_radius*2, contours=[10000],save=True, cat={ "TeVCat": [0, "s"],"PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"]}, color="Fermi")
        plt.show()

        if not ifnopt:
            if(TS["TS_all"]>TS_all[-1]):
                bestcache=copy.deepcopy(lm)
                bestmodelnamec=copy.copy(name)
                TS_all[-1]=TS["TS_all"]
                exts.append(ext)
                pts.pop()
            else:
                npt+=1
                next-=1
                Modelname=f"{npt}pt+{next}ext"+tDGE
                lm.remove_source(name)
                lm.add_source(pts[-1])
                name=f"pt{npt}"
                result = fit(region_name+"_iter", Modelname, WCDA, lm, 0, 5,mini="ROOT")
                TS, TSdatafram = getTSall([name], region_name+"_iter", Modelname, result, WCDA)
                TS_all[-1]=TS["TS_all"]
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
            bestmodelname=bestmodelnamec
            bestmodel=bestcache
        else:
            result = fit(region_name+"_iter", bestmodelname, WCDA, bestmodel, 0, 5,mini="ROOT")
            TS, TSdatafram = getTSall([], region_name+"_iter", bestmodelname, result, WCDA)
            return bestmodel,result



def fun_Logparabola(x,K,alpha,belta,Piv):
    return K*pow(x/Piv,alpha-belta*np.log(x/Piv))

def fun_Powerlaw(x,K,index,piv):
    return K*pow(x/piv,index)

def set_diffusebkg(K = 7.3776826e-13, Kf = True, Kb=None, index =-2.733, indexf = True, file="../../data/J0248_dust_bkg_template.fits", piv=3):
    fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    Diffuseshape = SpatialTemplate_2D(fits_file=file)
    Diffusespec = Powerlaw()
    Diffuse = ExtendedSource("Diffuse",spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
    Diffusespec.K = K * fluxUnit
    Diffusespec.K.fix=Kf
    if Kb:
        Diffusespec.K.bounds=Kb * fluxUnit
    else:
        Diffusespec.K.bounds=(0.2*K,5*K) * fluxUnit

    Diffusespec.piv = piv * u.TeV
    Diffusespec.piv.fix=True

    Diffusespec.index = index
    Diffusespec.index.fix = indexf
    Diffusespec.index.bounds = (-4,-1)
    Diffuseshape.K = 1/u.deg**2
    return Diffuse

def set_diffusemodel(name, K = 7.3776826e-13, Kf = False, Kb=None, index =-2.733, indexf = False, piv=3):
    fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    Diffuseshape = SpatialTemplate_2D(fits_file='../../data/j0248_diff_template.fits')
    Diffusespec = Powerlaw()
    Diffuse = ExtendedSource(name, spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
    Diffusespec.K = K * fluxUnit
    Diffusespec.K.fix=Kf

    if Kb:
        Diffusespec.K.bounds=Kb * fluxUnit
    else:
        Diffusespec.K.bounds=(0.2*K,5*K) * fluxUnit

    Diffusespec.piv = piv * u.TeV
    Diffusespec.piv.fix=True

    Diffusespec.index = index
    Diffusespec.index.fix = indexf
    Diffusespec.index.bounds = (-4,-1)
    Diffuseshape.K = 1/u.deg**2
    return Diffuse

def get_sources(lm,result):
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
                puv = result[1][0].loc[p,"positive_error"]
                plv = result[1][0].loc[p,"negative_error"]
                source[p.split('.')[-1]] = (1,par,par.value,puv,plv)
            else:
                source[p.split('.')[-1]] = (0,par,par.value,0,0)
            sources[name] = source
    return sources
from threeML import *
from WCDA_hal import HAL, HealpixConeROI, HealpixMapROI
from time import *
from Mymodels import *
import os
import numpy as np

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

    spec = Powerlaw() if spec is None else Log_parabola()
    
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
    else:
        pass

    if sigma is None and rdiff0 is None and radius is None:
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
    elif spec.name == "Powerlaw":
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

    return source




def fit(regionname, modelname, WCDA,Model,s,e,mini = "minuit",verbose=False,savefit=True):
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

    if savefit:
        time1 = strftime("%m-%d-%H", localtime())
        if not os.path.exists(f'../res/{regionname}/{modelname}/'):
            os.system(f'mkdir ../res/{regionname}/{modelname}/')
        # if not os.path.exists(f'../res/{regionname}/{modelname}/{time1}/'):
        #     os.system(f'mkdir ../res/{regionname}/{modelname}/{time1}/')
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

def fun_Logparabola(x,K,alpha,belta,Piv):
    return K*pow(x/Piv,alpha-belta*np.log(x/Piv))

def fun_Powerlaw(x,K,index,piv):
    return K*pow(x/piv,index)

def set_diffusebkg(K = 7.3776826e-13, Kf = True, index =-2.733, indexf = True):
    fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    Diffuseshape = SpatialTemplate_2D(fits_file='../../data/J0248_dust_bkg_template.fits')
    # Diffuseshape = SpatialTemplate_2D(fits_file='../../data/J0248_dust_bkg_template_icrs.fits')
    Diffusespec = Powerlaw()
    Diffuse = ExtendedSource("Diffuse",spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
    Diffusespec.K = K * fluxUnit
    Diffusespec.K.fix=Kf
    Diffusespec.K.bounds=(0.2*7.3776826e-13,5*7.3776826e-13) * fluxUnit

    Diffusespec.piv = 3 * u.TeV
    Diffusespec.piv.fix=True

    Diffusespec.index = index
    Diffusespec.index.fix = indexf
    Diffusespec.index.bounds = (-4,-1)
    Diffuseshape.K = 1/u.deg**2
    return Diffuse

def set_diffusemodel(name, K = 7.3776826e-13, Kf = False, index =-2.733, indexf = False):
    fluxUnit = 1. / (u.TeV * u.cm**2 * u.s)
    Diffuseshape = SpatialTemplate_2D(fits_file='../../data/j0248_diff_template.fits')
    Diffusespec = Powerlaw()
    Diffuse = ExtendedSource(name, spatial_shape=Diffuseshape,spectral_shape=Diffusespec)
    Diffusespec.K = K * fluxUnit
    Diffusespec.K.fix=Kf
    Diffusespec.K.bounds=(0.2*7.3776826e-14,5*7.3776826e-12) * fluxUnit

    Diffusespec.piv = 3 * u.TeV
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
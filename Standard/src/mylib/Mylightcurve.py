from threeML import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ROOT as rt


log = setup_logger(__name__)
log.propagate = False

#### . LC Model
def _SBPL(x,par):
    result = par[0] * (((x - par[5]) / par[1])**(-par[2] * par[3]) + ((x - par[5]) / par[1])**(-par[2] * par[4]))**(-1 / par[2])
    # result[x<par[5]]=0
    if x<0:
        result=0
    return result

def _afterglow(x, params):
    F0, tb, omega, alpha1, alpha2, t0=params
    result = F0 * pow(pow((x - t0) / tb, -omega * alpha1) + pow((x - t0) / tb, -omega * alpha2), -1 / omega)
    result[x<t0]=0
    return result

def FRP(x,par):
    result = []
    if type(x)==int: 
        if x>=par[9]:
            result.append(_SBPL(x,par))
        elif x>=par[5]:
            result.append(_SBPL(par[9],par)*pow(x/par[9],par[10]))
        else:
            result.append(0)
        return result[0]
    else:
        for xx in x:
            if xx>=par[9]:
                result.append(_SBPL(xx,par))
            elif xx<par[9] and xx>=par[5]:
                result.append(_SBPL(par[9],par)*pow(xx/par[9],par[10]))
            else:
                result.append(0)
        return np.array(result)

def SSQPL(x,par):
    x = np.array(x)
    part1 = pow(FRP(x, par), -par[6])
    part2 = pow(FRP(par[7], par) * pow(x / par[7],par[8]), -par[6])
    return pow(part1+part2, -(1/par[6]))

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 2 / stddev)**2)

def poly(x, a, b, c, d):
    return d*x**3 + c*x**2 + b*x + a

# Tools 
def raw2array(data):
    Tfit = []
    for i in range(len(data)):
        datacache = [data[i][j] for j in range(len(data[0]))]
        Tfit.append(datacache)
    return np.array(Tfit)

def poisson_fluctuation(x):
    """# 定义泊松涨落函数"""
    return np.random.poisson(x)

def p2sigma(p):
    import scipy.stats as stats
    """p value to sigma"""
    return -stats.norm.ppf(p/2)

def wavelet(time, counts, t1,t2, f0,fe, plot=True):
    """wavelet and plot"""
    import pywt
    dt = time[1]-time[0]
    rs = int((t1-time[0]+dt)/dt)
    re = int((t2-time[0]+dt)/dt)
    # 进行小波变换
    coeffs, freqs = pywt.cwt(counts, 1/np.concatenate((np.linspace(0.001, 1,1000),np.linspace(1, 10,1000))), 'morl',sampling_period=dt) #np.concatenate((cd,
    tj = ((freqs>f0) & (freqs<fe))
    print(coeffs[tj].shape, rs, re)
    max = np.max(abs(coeffs[tj][:, rs:re]))
    maxindex = np.unravel_index(np.argmax(abs(coeffs[tj][:, rs:re])),abs(coeffs[tj][:, rs:re]).shape)

    if plot:
        print("max c: ",max,maxindex,counts[rs:re][maxindex[1]], freqs[tj][maxindex[0]])
        # 绘制频率时间图
        fig, ax1 = plt.subplots(sharex = True)
        ct = ax1.contourf(time[rs:re], freqs[tj], abs(coeffs[tj][:, rs:re]), cmap='Blues') #, interpolation='nearest'   ,vmin=np.min(abs(coeffs[tj][:,(t0-bias)*bint:(te-bias)*bint])), vmax=np.max(abs(coeffs[tj][:,(t0-bias)*bint:(te-bias)*bint])),vmin=np.min(abs(coeffs[tj][:,(t0+bias)*bint:(te+bias)*bint])), vmax=np.max(abs(coeffs[tj][:,(t0+bias)*bint:(te+bias)*bint]))
        cax = fig.add_axes([1, 0.1, 0.02, 0.8])
        fig.colorbar(ct, cax=cax,label="Wavelet coefficients")

        ax1.scatter(counts[rs:re][maxindex[1]],freqs[tj][maxindex[0]],marker="x",c="r")
        ax1.set_title('Wavelet Transform')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale("log")
        ax1.set_ylim(f0,fe)
        ax1.set_xlim(t1,t2)

        ax2 = ax1.twinx()
        ax2.errorbar(time[rs:re], counts[rs:re], np.sqrt(counts[rs:re]), alpha=0.1,c="r")
        ax2.set_ylabel('Counts',color='tab:blue')
        plt.show()

        # plt.figure()
        # plt.imshow(abs(coeffs[tj][:,int((t0+bias)*bint):int((te+bias)*bint)]), aspect='auto', interpolation='nearest', origin="lower", norm=colors.LogNorm(vmin=np.min(abs(coeffs[tj][:,int((t0+bias)*bint):int((te+bias)*bint)])), vmax=np.max(abs(coeffs[tj][:,int((t0+bias)*bint):int((te+bias)*bint)]))), extent=[t0, te, freqs[tj][0], freqs[tj][-1]])
        # plt.colorbar()
        # plt.yscale("log")
    return time[rs:re], coeffs[tj][rs:re], freqs[tj]



# data binning
def nprebin(array,rebin):
    """# 对数组进行重新binning操作   add"""
    array = np.array(array)
    if len(array)%rebin:
        array = array[:-(len(array)%rebin)]
    return array.reshape((len(array)//rebin, rebin)).sum(axis=1)

def nprebinmean(array,rebin):
    """# 对数组进行重新binning操作   mean"""
    array = np.array(array)
    if len(array)%rebin:
        array = array[:-(len(array)%rebin)]
    return array.reshape((len(array)//rebin, rebin)).mean(axis=1)


def nprebine(array,rebin):
    """对数组进行重新binning操作    err"""
    if len(array)%rebin:
        array = np.array(array)
    return [np.sqrt((dd**2).sum()) for dd in array.reshape((len(array)//rebin, rebin))]





############################## .  小波谱
def is_power_of_2(num):
    """
    Returns whether num is a power of two or not

    :param num: an integer positive number
    :return: True if num is a power of 2, False otherwise
    """

    return num != 0 and ((num & (num - 1)) == 0)

def next_power_of_2(x):

    # NOTES for this black magic:
    # * .bit_length returns the number of bits necessary to represent self in binary
    # * x << y means 1 with the bits shifted to the left by y, which is the same as multiplying x by 2**y (but faster)

    return 1 << (x-1).bit_length()


#   waipy.cwt(data=data_norm, mother='DOG', dt=dt, param=2, s0=dt * 2, dj=0.25,
#                        J=7 / 0.25, pad=1, alpha=alpha, name='x')
def cwt(data, dt, mother, param, s0, dj, J=None):
    try:
        import pycwt
    except:
        activate_warnings()
        log.warning("No nodule named pycwt")
        silence_warnings()

    # Make sure we are dealing with a np.array
    data = np.array(data)

    # Check that data have a power of two size
    N = data.size
    assert is_power_of_2(N), "Sample size for CWT is %s, which is not a power of 2" % N

    # Maximum order
    if J is None:

        J = int(np.floor(np.log2(N * dt / s0) / dj))

    # Normalize and standardize data
    data = (data - data.mean()) / np.sqrt(np.var(data))

    # Compute variance of standardized data
    variance = np.var(data)

    # Autocorrelation (lag-1)
    alpha = np.corrcoef(data[0:-1], data[1:])[0, 1]

    # Setup mother wavelet according to user input

    if mother.upper()=='DOG':

        mother = pycwt.DOG(param)

    elif mother.upper()=='MORLET':

        mother = pycwt.Morlet(param)

    elif mother.upper()=='PAUL':

        mother = pycwt.Paul(param)

    elif mother.upper()=='MEXICAN HAT':

        mother = pycwt.DOG(2)

    else:

        raise ValueError("Wavelet %s is not known. Possible values are: DOG, MORLET, PAUL, MEXICAN HAT")

    # Perform Continuous Wavelet Transform
    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(data, dt, dj, s0, J, mother)

    # Compute power
    power = np.abs(wave)**2

    # Compute periods
    period = [e * mother.flambda() for e in scales]

    # Global normalized power spectrum
    global_ws = variance * (np.sum(power.conj().transpose(), axis=0) / N)

    results = {'autocorrelation': alpha, 'period': period, 'global_ws': global_ws,
               'scale': scales, 'wave': wave, 'freqs': freqs, 'coi': coi}

    return results

def wavelet_spectrum(time, counts, dt, t1, t2, plot=True, quiet=False, max_time_scale=None):
    """
    Compute and return the wavelet spectrum

    :param time: a list or a np.array instance containing the time corresponding to each bin in the light curve
    :param counts: a list or a np.array instance containing the counts in each bin
    :param dt: the size of the bin in the light curve
    :param t1: beginning of time interval to use for computation. Of course time.min() <= t1 < time.max()
    :param t2: end of time interval to use for computation. Of course time.min() < t2 <= time.max()
    :param plot: (True or False) whether to produce or not a plot of the spectrum. If False, None will be returned
    instead of the Figure instance
    :param quiet: if True, suppress output (default: False)
    :param max_time_scale: if provided, the spectrum will be computed up to this scale (default: None, i.e., use the
    maximum possible scale)
    :return: (results, fig): a tuple containing a dictionary with the results and the figure
    (a matplotlib.Figure instance)
    """

    counts_copy = np.copy(counts)

    idx = (time >= t1) & (time <= t2)
    counts_copy = counts_copy[idx]

    n_events = np.sum(counts_copy)

    if not quiet:

        print("Number of events: %i" % n_events)
        print("Rate: %s counts/s" % (n_events / (t2 - t1)))

    # Do the Continuous Wavelet transform

    # Default parameters
    s0 = 2 * dt # minimum scale
    dj = 0.125 * 2 # this controls the resolution (number of points)

    if max_time_scale is not None:

        # Compute the corresponding J
        J = int(np.floor(np.log2(max_time_scale / s0 * dj * 2) / dj))

    else:

        J = None

    result = cwt(data=counts_copy, mother='MEXICAN HAT', dt=dt, param=2, s0=s0, dj=dj, J=J)

    # import waipy
    # data_norm = waipy.normalize(counts_copy)
    # alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0, 1]
    # result = waipy.cwt(data=data_norm, mother='DOG', dt=dt, param=2, s0=dt * 2, dj=0.25,
    #                    j1=7 / 0.25, pad=1, lag1=alpha, name='x')
    #
    # result['autocorrelation'] = alpha

    if not quiet:

        print("Lag-1 autocorrelation = {:4.8f}".format(result['autocorrelation']))

    if plot:

        figure, sub = plt.subplots(1,1)

        _ = sub.plot(result['period'], (result['global_ws'] + result['autocorrelation']) / result['scale'], 'o')

        sub.set_xlabel(r"$\delta t$ (s)")
        sub.set_ylabel("Power")

        sub.set_xscale("log")
        sub.set_yscale("log")

        figure.tight_layout()

    else:

        figure = None

    return result, figure

def simulate_flat_poisson_background(input1, input2, dt, t0, t1, t2, type = "bkg"):
    rs = int((t1-t0)/dt)
    re = int((t2-t0)/dt)
    if type=="bkg":
        time = np.arange(t1, t2, dt)
        lc_t = np.array(input1[rs:re])
        lcf = np.random.poisson(lc_t)
    elif type=="tho":
        time = np.arange(t1, t2, dt)
        lc_t = np.array(input1[rs:re]+input2[rs:re])
        lcf = np.random.poisson(lc_t)
    return time, lcf

def worker(i, input1,input2, dt, t0, t1, t2, max_time_scale, results_to_save=('global_ws',), type = "bkg"):

    time, lc = simulate_flat_poisson_background(input1,input2, dt, t0, t1, t2, type = type)
    # print(i)

    try:

        result, _ = wavelet_spectrum(time, lc, dt, t1, t2, plot=False, quiet=True, max_time_scale=max_time_scale)

    except:

        raise

    # Delete everything except what needs to be saved
    keys_to_delete = filter(lambda x: (x not in results_to_save), result.keys())

    map(lambda x: result.pop(x), keys_to_delete)

    # Transform the results in float16 to save memory
    if 'global_ws' in results_to_save:

        result['global_ws'] = np.array(result['global_ws'], np.float16)

    pbar.update(1)
    return result

def background_spectrum(input1, input2, dt, t0, t1, t2, n_simulations=1000, plot=True, sig_level=68.0, max_time_scale=None, type = "bkg"):
    """
    Produce the wavelet spectrum for the background, i.e., a flat signal with Poisson noise with the provided rate.
    Using a simple Monte Carlo simulation, it also produces the confidence region at the requested confidence level.

    NOTE: if you request a very high confidence level, you need to ask for many simulations.

    :param rate: the rate of the background (in counts/s)
    :param dt: the binning of the light curve
    :param t1: start time of the light curve
    :param t2: stop time of the light curve
    :param n_simulations: number of simulations to run to produce the confidence region
    :param plot: whether or not to plot the results (default: True)
    :param sig_level: significance level (default: 68.0, corresponding to 68%)
    :return: (low_bound, median, hi_bound, figure)
    """
    global pbar
    pbar = tqdm(total=n_simulations)
    # worker_wrapper = functools.partial(worker,
    #                                    input=input, ebin=ebin, dt=dt, t1=t1, t2=t2,
    #                                    max_time_scale=max_time_scale,
    #                                    results_to_save=['global_ws'])

    # pool = multiprocess.Pool()

    all_results = []

    # Get one to get the periods (this is to spare memory)
    one_result = worker(0, input1, input2,  dt, t0, t1, t2, max_time_scale, results_to_save=['period', 'scale'], type = type)
    periods = np.array(one_result['period'])
    scales = np.array(one_result['scale'])

    try:

        # for i, res in enumerate(pool.imap_unordered(worker_wrapper, range(n_simulations), chunksize=100)):
        #for i, res in enumerate(itertools.imap(worker_wrapper, range(n_simulations))):
        for i in range(n_simulations):
            res = worker(i, input1, input2, dt=dt, t0=t0, t1=t1, t2=t2, max_time_scale=max_time_scale, results_to_save=['global_ws'], type = type)
            # sys.stderr.write("\r%i / %i" % (i+1, n_simulations))

            all_results.append(res)

    except:

        raise

    finally:
        print("Simulation finished!")
        # pool.close()

    low_bound = np.zeros_like(periods)
    median = np.zeros_like(periods)
    hi_bound = np.zeros_like(periods)

    delta = sig_level / 2.0

    for i, scale in enumerate(tqdm(scales,desc="scales")):

        # Get the value from all simulations at this scale
        values = np.array(list(map(lambda x:np.sqrt(x['global_ws'][i] / scale), all_results)))
        # print(values)
        p16, p50, p84 = np.percentile(values, [50.0 - delta, 50.0, 50.0 + delta])

        low_bound[i] = p16
        median[i] = p50
        hi_bound[i] = p84

    if plot:

        figure, sub = plt.subplots(1, 1)

        _ = sub.fill_between(periods, low_bound, hi_bound, alpha=0.5)
        _ = sub.plot(periods, median, lw=2, color='black')

        sub.set_xlabel(r"$\delta t$ (s)")
        sub.set_ylabel("Power")

        sub.set_xscale("log")
        sub.set_yscale("log")

        figure.tight_layout()

    else:

        figure = None

    return low_bound, median, hi_bound, figure

def plot_spectrum_with_background(spectrum_results, low_bound, median, hi_bound, **kwargs):

    figure, sub = plt.subplots(1, 1, **kwargs)

    _ = sub.plot(spectrum_results['period'],
                 (spectrum_results['global_ws'] + spectrum_results['autocorrelation']) / spectrum_results['scale'],
                 'o')

    _ = sub.fill_between(spectrum_results['period'], low_bound, hi_bound, alpha=0.5)
    _ = sub.plot(spectrum_results['period'], median, lw=2, color='black', linestyle='--')

    sub.set_xlabel(r"$\delta t$ (s)")
    sub.set_ylabel("Power")

    sub.set_xscale("log")
    sub.set_yscale("log")

    figure.tight_layout()

    return figure


########################### . LC class

class lc(object):
    """
        光变类

        Parameters:

        Returns:
    """ 
    # read
    filedata=""
    funcfile=""
    ffunc=None
    upfile=None
    bins=12
    ebin=0
    time=np.array([])
    counts=np.array([])
    counts_nt=np.array([])

    t0 = 0
    dt = 0.0001
    nt = 0
    te = 0

    linebkg = list(range(bins))
    func = list(range(bins))
    bkgp = []
    bkg=None
    tho=None
    method="from_root"

    #for GRB
    t90=None

    def __init__(self, file = None, tname = None, ebin=None, funcfile=None, bkgfile=None, bkgscale=None):
        """ 
            从root文件里读取TH1D作为光变
        """
        if file is not None:
            self.file=file
            if "root" in self.file:
                import uproot
                self.upfile = uproot.open(self.file)

                if tname != None:
                    self.time = self.upfile[tname].axis().centers()
                    self.counts = self.upfile[tname].values()
                    if bkgfile is not None:
                        bkgr = uproot.open(bkgfile)
                        self.bkg = bkgr[tname].values()
                        if bkgscale is not None:
                            self.bkg = self.bkg * bkgscale
                else:
                    self.time = self.upfile[f"mjd{ebin};1"].axis().centers()
                    self.counts = self.upfile[f"mjd{ebin};1"].values()

                if funcfile is not None:
                    self.funcfile = funcfile
                    if "root" in self.funcfile:
                        ffunc = rt.TFile(funcfile)
                        for i in range(12):
                            self.func[i] = ffunc.Get("F5%d"%i)
                            self.linebkg[i] = ffunc.Get("linebkg%d"%i)
                        self.bkg = [self.linebkg[ebin].Eval(tt)*self.dt for tt in self.time]
                        self.tho = [self.func[ebin].Eval(tt-226)*self.dt for tt in self.time]
                        # np.save("../../data/lc_data/GRB221009A/bkg.npy", self.bkg)
                        # np.save("../../data/lc_data/GRB221009A/tho.npy", self.tho)
                    elif "npy" in self.funcfile:
                        self.bkg = np.load(self.funcfile) #[-1]*self.dt
                        self.tho = np.load(self.funcfile.replace("bkg","tho")) #[-1]*self.dt
            elif "txt" in self.file:
                data = np.genfromtxt(file, delimiter=' ', names=True)
                Tdata = raw2array(data)
                self.time = Tdata[:,0]
                self.counts = Tdata[:,1]
            self.ebin = ebin
            self.dt=self.time[1]-self.time[0]
            self.t0=self.time[0]-self.dt
            self.nt=len(self.time)
            self.te=self.t0+self.nt*self.dt

    def __add__(self, others):
        newclass = lc()
        if len(self.time) > len(others.time):
            newclass.time = self.time
        else:
            newclass.time = others.time

        tolerance=0.00000005
        newclass.counts=self.counts
        if self.bkg is not None:
            newclass.bkg = self.bkg
        for i, tt in enumerate(tqdm(others.time)):
            indices = np.where(np.isclose(self.time, tt, rtol=tolerance, atol=tolerance))[0]
            newclass.counts[indices] = self.counts[indices]+others.counts[i]
            if others.bkg is not None:
                newclass.bkg[indices] = self.bkg[indices]+others.bkg[i]
        return newclass
    
    @property
    def bkgpar(self):
        if self.linebkg is not None:
            self.bkgp = self.linebkg[self.ebin].GetParameters()
        return self.bkgp
    def drawlc(self, t1 = 230, t2 = 334.8576, drawbkg = False):
        ax = plt.axes([0.1, 0.75, 0.65, 0.2])
        if self.bkg is not None and not drawbkg:
            self.counts_nt = self.counts-np.array(self.bkg)
        else:
            self.counts_nt = self.counts
        idx = (self.time >= t1) & (self.time <= t2)
        ax.plot(self.time[idx],self.counts_nt[idx],'k', linewidth=1.5)
        a=plt.show()

    def getdatafram(self, subbkg=True):
        if not subbkg:
            return pd.DataFrame(np.array([self.time, self.counts, np.sqrt(self.counts)]).T, columns=['x', 'y', 'yerr'])
        else:
            return pd.DataFrame(np.array([self.time, self.counts_nt, np.sqrt(self.counts)]).T, columns=['x', 'y', 'yerr'])

    def rebin(self, tobin):
        self.time, self.counts = nprebinmean(self.time,tobin), nprebin(self.counts,tobin)
        if self.bkg is not None:
            self.bkg = nprebin(self.bkg,tobin)
        if self.tho is not None:
            self.tho = nprebin(self.tho,tobin)
        self.dt=self.time[1]-self.time[0]
        self.t0=self.time[0]-self.dt
        self.nt=len(self.time)
        self.te=self.t0+self.nt*self.dt

    def fit(self, func, t1, t2):
        from scipy.optimize import curve_fit
        rs = int((t1-self.t0)/self.dt)
        re = int((t2-self.t0)/self.dt)
        params, covariance = curve_fit(func, self.time[rs:re], self.counts[rs:re], p0=[10000, 260, 4])
        plt.plot(self.time[rs:re], self.counts[rs:re])
        xx = np.arange(t1,t2,0.1)
        ll=""
        for i, par in enumerate(params):
            ll+=f"par1: {par:.2f} ± {np.sqrt(covariance[i][i]):.2f} . "
        plt.plot(xx, func(xx, *params), label=ll)
        plt.legend()

    def fitbkg(self, func, t1, t2, p0=None, asbkg=True, plot=False):
        from scipy.optimize import curve_fit
        rs = int((t1-self.t0)/self.dt)
        re = int((t2-self.t0)/self.dt)
        params, covariance = curve_fit(func, self.time[rs:re], self.bkg[rs:re], p0=p0)
        xx = np.arange(t1,t2,0.1)
        ll=""
        if plot:
            plt.plot(self.time[rs:re], self.bkg[rs:re])
            for i, par in enumerate(params):
                ll+=f"par{i}: {par:.2e} ± {np.sqrt(covariance[i][i]):.2e} . "
            plt.plot(xx, func(xx, *params), label=ll)
            plt.legend()
        if asbkg:
            self.bkg = [func(tt, *params) for tt in self.time]
        self.bkgp = params

    def Drawtimeband(self,t1, t2, tt1, tt2):
        fig = plt.figure()
        plt.plot(self.time,self.counts,label=f"WCDA  {tt1}->{tt2:.2f}")
        plt.fill_between(self.time,self.counts,alpha=0.3)
        plt.fill_between(self.time,self.counts,where=(self.time>=tt1) & (self.time<=tt2),alpha=0.2,color="r")
        plt.xlim(t1,t2)
        # if hebsf is not None:
        #     plt.plot(hebsf[:,1], hebsf[:,4]/max(hebsf[:,4])*max(data[:,4]),label=f"HEBS   {t0}->{thebs:.2f}")
        #     plt.fill_between(hebsf[:,1],hebsf[:,4]/max(hebsf[:,4])*max(data[:,4]),where=(hebsf[:,1]>=t0) & (hebsf[:,1]<=thebs),alpha=0.3,color="orange")

        if self.bkg is not None:
            plt.plot(self.time, self.bkg,c="black",label="Bkg")
        if self.tho is not None:
            plt.plot(self.time, self.tho+self.bkg,c="black",label="Model")
        plt.ylabel("Counts/s")
        plt.xlabel("T")
        plt.legend(loc="upper right")

    def simulate_flat_poisson_background(self, t1, t2, type = "bkg"):
        rs = int((t1-self.t0)/self.dt)
        re = int((t2-self.t0)/self.dt)
        if type=="bkg":
            lc_t = np.array(self.bkg[rs:re])
            lcf = np.random.poisson(lc_t)
        elif type=="tho":
            lc_t = np.array(self.bkg[rs:re]+self.tho[rs:re])
            lcf = np.random.poisson(lc_t)
        return self.time[rs:re], lcf

    def simratio(self, t1, t2):
        tt2, bbb = self.simulate_flat_poisson_background(t1, t2, type="bkg")
        ratio = []
        rs = int((t1-self.t0)/self.dt)
        re = int((t2-self.t0)/self.dt)
        binws = []
        for binw in tqdm(np.arange(-3,np.log10(15),0.01)):
            binw = 10**(binw)
            binws.append(binw)
            rebin=int(binw/self.dt)
            tt = nprebinmean(self.time[rs:re],rebin)
            dd = nprebin(self.counts[rs:re],rebin)
            bb = nprebin(bbb,rebin)
            var = dd.var()
            var2 = bb.var()
            rr = var/var2
            ratio.append(rr)
        return np.array(binws), np.array(ratio)

    def getMVT(self, binws, ratio, ifp=False):
        from scipy.optimize import curve_fit
        from sympy import symbols, diff, solve, sqrt, I, log

        if ifp:
            x, y, a, b, c, d, e = symbols('x y a b c d e')
            ae, be, ce, de, ee = symbols('ae be ce de ee')
            y=a+b*log(x,10)+c*log(x,10)**2+d*log(x,10)**3+e*log(x,10)**4
            yd = diff(y, x)
            xmin = solve(yd,x)
            exmin = [sqrt((diff(xmin,a)*ae)**2+(diff(xmin,b)*be)**2+(diff(xmin,c)*ce)**2+(diff(xmin,d)*de)**2+(diff(xmin,e)*ee)**2) for xmin in xmin]

        
        binws = np.array(binws)
        ratio = np.array(ratio)
        x = binws
        y = ratio/binws
        def log_parabola(x, a, b, c, d, e):
            return a + b * np.log10(x) + c * (np.log10(x))**2 + d * (np.log10(x))**3 + e * (np.log10(x))**4

        fitrange = (x>0.01) & (x<1.8)
        params, covariance = curve_fit(log_parabola, x[fitrange], y[fitrange])
        a, b, c, d, e = params
        ae, be, ce, de, ee = np.sqrt(np.diag(covariance))

        yvals=log_parabola(x[fitrange], a, b, c, d, e)
        # yvals=spline(x[fitrange])
        ii=0
        aa=True

        if ifp:
            while aa:
                MVT2 = xmin[ii].evalf(subs={"a":a, "b":b, "c":c, "d":d, "e":e})
                MVT2e = exmin[ii].evalf(subs={"a":a, "b":b, "c":c, "d":d, "e":e, "ae":ae, "be":be, "ce":ce, "de":de, "ee":ee})
                aa = (MVT2.has(I)) or (MVT2e.has(I))
                ii+=1
            print(MVT2, MVT2e)
        MVT = x[np.argmin(yvals)]
        plt.figure()
        plt.plot(x,y,'*',label='original values')
        plt.xscale("log")
        plt.yscale("log")
        if ifp:
            plt.plot(x[fitrange],yvals,'r',label=f'polyfit values ( MVT: {MVT2:.3f} ± {MVT2e:.3f})')
        else:
            plt.plot(x[fitrange],yvals,'r',label=f'polyfit values ( MVT: {MVT:.3f}')
        plt.xlabel("Bin width(s)")
        plt.ylabel("Ratio of variances / Bin width(s)")
        plt.legend()
        if ifp:
            return MVT, MVT2, MVT2e
        else:
            return MVT
    
    def wavelet(self, t1, t2, f1, f2):
        return wavelet(self.time, self.counts, t1, t2, f1, f2, plot=True)
    
    def getwavespec(self, t0, mi, label="WCDA", ifsimbkg=True, n_simulations=200, type = "tho", sig_level=99):
        t1 = t0+2**mi*self.dt
        results2, fig2 = wavelet_spectrum(self.time, self.counts, self.dt, t0 ,t1, plot=False)
        plt.scatter(results2['period'],np.sqrt((results2['global_ws'] + results2['autocorrelation']) / results2['scale']),c="black",s=10,marker="*",label=label)
        plt.xscale('log')
        plt.yscale('log')

        xx = np.hstack((np.arange(0,1,0.0001), np.arange(1,2000,0.1)))
        for j in range(20):
            bias=0.5**j
            bias2=2**(j+1)
            plt.plot(xx,bias*xx,"--",c="grey",alpha=0.5)
            plt.plot(xx,bias2*xx,"--",c="grey",alpha=0.5)
        plt.plot(xx,2*xx**(-1/2),"-.",c="black",alpha=0.5)

        plt.xlim(0.01,2000)
        plt.ylim(0.01,1000)
        plt.xlabel(r"$\Delta T (sec)$",size=15)
        plt.ylabel(r"$\sigma_{X,\Delta t}$",size=15)
        plt.grid()

        plt.xscale("log")
        plt.yscale("log")

        plt.legend(loc="upper right")

        if ifsimbkg:
            a,b = simulate_flat_poisson_background(self.bkg,self.tho, dt = self.dt, t0=self.t0, t1 = t0, t2 = t1, type="tho")
            print(len(a),len(b))
            low_bound, median, hi_bound, figure = background_spectrum(self.bkg, self.tho, dt = self.dt, t0=self.t0, t1 = t0, t2 = t1, n_simulations=n_simulations, plot=True, sig_level=sig_level, type = type)

            ax = figure.get_axes()[0]
            xx = np.hstack((np.arange(0,1,0.0001), np.arange(1,2000,0.1)))
            for j in range(20):
                bias=0.5**j
                bias2=2**(j+1)
                ax.plot(xx,bias*xx,"--",c="grey",alpha=0.5)
                ax.plot(xx,bias2*xx,"--",c="grey",alpha=0.5)
            ax.plot(xx,2*xx**(-1/2),"-.",c="black",alpha=0.5)

            ax.set_xlim(0.01,2000)
            ax.set_ylim(0.01,100)
            ax.scatter(results2['period'],np.sqrt((results2['global_ws'] + results2['autocorrelation']) / results2['scale']),s=40,marker="*",c=(np.sqrt((results2['global_ws'] + results2['autocorrelation']) / results2['scale'])>hi_bound),cmap="Spectral",label="WCDA")
            # ax.scatter(results1['period'],np.sqrt((results1['global_ws'] + results1['autocorrelation']) / results1['scale']),s=30,marker="^",c="black",label="HEBS")

            ax.set_xlabel(r"$\Delta T (sec)$",size=15)
            ax.set_ylabel(r"$\sigma_{X,\Delta t}$",size=15)
            ax.grid()

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(loc="upper right")
            # print(np.array(results2['period'])[np.sqrt((results2['global_ws'] + results2['autocorrelation']) / results2['scale'])>hi_bound])
from __future__ import division

import collections
import contextlib
import copy
from builtins import range, str
from typing import Union

import astromodels
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astromodels import Parameter
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft as convolve
from past.utils import old_div
from scipy.stats import poisson
from threeML.io.logging import setup_logger
from threeML.parallel import parallel_client
from threeML.plugin_prototype import PluginPrototype
from threeML.utils.statistics.gammaln import logfactorial
from tqdm.auto import tqdm

from hawc_hal.convolved_source import (ConvolvedExtendedSource2D,
                                       ConvolvedExtendedSource3D,
                                       ConvolvedPointSource,
                                       ConvolvedSourcesContainer)
from hawc_hal.healpix_handling import (FlatSkyToHealpixTransform,
                                       SparseHealpix, get_gnomonic_projection)
from hawc_hal.log_likelihood import log_likelihood
from hawc_hal.maptree import map_tree_factory
from hawc_hal.maptree.data_analysis_bin import DataAnalysisBin
from hawc_hal.maptree.map_tree import MapTree
from hawc_hal.psf_fast import PSFConvolutor
from hawc_hal.response import hawc_response_factory
from hawc_hal.util import ra_to_longitude

import os

log = setup_logger(__name__)
log.propagate = False

import concurrent.futures

class HAL(PluginPrototype):
    """
    The HAWC Accelerated Likelihood plugin for 3ML.
    :param name: name for the plugin
    :param maptree: Map Tree (either ROOT or hdf5 format)
    :param response: Response of HAWC (either ROOT or hd5 format)
    :param roi: a ROI instance describing the Region Of Interest
    :param flat_sky_pixels_size: size of the pixel for the flat sky projection (Hammer Aitoff)
    :param n_workers: number of workers to use for the parallelization of reading
    :param set_transits: specifies the number of transits to use for the given maptree.
    ROOT files, default=1
    """

    def __init__(
        self,
        name,
        maptree,
        response_file,
        roi,
        flat_sky_pixels_size: float = 0.17,
        n_workers: int = 1,
        set_transits=None,
    ):
        # Store ROI
        self._roi = roi
        self._n_workers = n_workers

        # optionally specify n_transits
        if set_transits is not None:
            log.info(f"Setting transits to {set_transits}")
            n_transits = set_transits

        else:
            n_transits = None
            log.info("Using transits contained in maptree")

        # Set up the flat-sky projection
        self.flat_sky_pixels_size = flat_sky_pixels_size
        self._flat_sky_projection = self._roi.get_flat_sky_projection(
            self.flat_sky_pixels_size
        )

        # Read map tree (data)
        self._maptree = map_tree_factory(
            maptree, roi=self._roi, n_transits=n_transits, n_workers=self._n_workers
        )

        # Read detector response_file
        self._response = hawc_response_factory(
            response_file_name=response_file, n_workers=self._n_workers
        )

        # All energy/nHit bins are loaded in memory
        self._all_planes = list(self._maptree.analysis_bins_labels)

        self._planes = len(self._all_planes)

        # self.deg2_to_rad2 = 0.00030461741978670857

        # def get_responses(energy_bin_id, dec_bin1, dec_bin2):
        #     this_response_bin1 = dec_bin1[energy_bin_id]
        #     this_response_bin2 = dec_bin2[energy_bin_id]

        #     c1, c2 = this_response_bin1.declination_center, this_response_bin2.declination_center

        #     idx = (self._flat_sky_projection.decs >= c1) & (self._flat_sky_projection.decs < c2) & self._active_flat_sky_mask

        #     ss1 = this_response_bin1.sim_signal_events_per_bin*self._flat_sky_projection.project_plane_pixel_area * self.deg2_to_rad2 / this_response_bin1.sim_differential_photon_fluxes
        #     ss2 = this_response_bin2.sim_signal_events_per_bin*self._flat_sky_projection.project_plane_pixel_area * self.deg2_to_rad2 / this_response_bin1.sim_differential_photon_fluxes

        #     w1 = (self._flat_sky_projection.decs[idx] - c2) / (c1 - c2)
        #     w2 = (self._flat_sky_projection.decs[idx] - c1) / (c2 - c1)

        #     return idx, np.ascontiguousarray(w1), np.ascontiguousarray(w2), np.ascontiguousarray(ss1), np.ascontiguousarray(ss2)
        
        # self._resultss = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for energy_bin_id in range(self._response.n_energy_planes):
        #         results = list(executor.map(lambda bins: get_responses(str(energy_bin_id), *bins), zip(self._dec_bins_to_consider[:-1], self._dec_bins_to_consider[1:])))
        #         self._resultss.append(results)

        # Use a renormalization of the background as nuisance parameter
        # NOTE: it is fixed to 1.0 unless the user explicitly sets it free (experimental)
        self._nuisance_parameters = collections.OrderedDict()
        # self._nuisance_parameters['%s_bkg_renorm' % name] = Parameter('%s_bkg_renorm' % name, 1.0,
        self._nuisance_parameters[f"{name}_bkg_renorm"] = Parameter(
            f"{name}_bkg_renorm",
            1.0,
            min_value=0.5,
            max_value=1.5,
            delta=0.01,
            desc="Renormalization for background map",
            free=False,
            is_normalization=False,
        )

        # Instance parent class

        # super(HAL, self).__init__(name, self._nuisance_parameters)
        # python3 new way of doing things
        super().__init__(name, self._nuisance_parameters)

        self._likelihood_model = None

        # These lists will contain the maps for the point sources
        self._convolved_point_sources = ConvolvedSourcesContainer()
        # and this one for extended sources
        self._convolved_ext_sources = ConvolvedSourcesContainer()

        # The active planes list always contains the list of *indexes* of the active planes
        self._active_planes = None

        # Set up the transformations from the flat-sky projection to Healpix, as well as the list of active pixels
        # (one for each energy/nHit bin). We make a separate transformation because different energy bins might have
        # different nsides
        self._active_pixels = collections.OrderedDict()
        self._flat_sky_to_healpix_transform = collections.OrderedDict()

        for bin_id in self._maptree:
            this_maptree = self._maptree[bin_id]
            this_nside = this_maptree.nside
            this_active_pixels = roi.active_pixels(this_nside)

            this_flat_sky_to_hpx_transform = FlatSkyToHealpixTransform(
                self._flat_sky_projection.wcs,
                "icrs",
                this_nside,
                this_active_pixels,
                (
                    self._flat_sky_projection.npix_width,
                    self._flat_sky_projection.npix_height,
                ),
                order="bilinear",
            )

            self._active_pixels[bin_id] = this_active_pixels
            self._flat_sky_to_healpix_transform[bin_id] = this_flat_sky_to_hpx_transform

        # This will contain a list of PSF convolutors for extended sources, if there is any in the model

        self._psf_convolutors = None

        # Pre-compute the log-factorial factor in the likelihood, so we do not keep to computing it over and over
        # again.
        self._log_factorials = collections.OrderedDict()

        # We also apply a bias so that the numerical value of the log-likelihood stays small. This helps when
        # fitting with algorithms like MINUIT because the convergence criterium involves the difference between
        # two likelihood values, which would be affected by numerical precision errors if the two values are
        # too large
        self._saturated_model_like_per_maptree = collections.OrderedDict()

        # The actual computation is in a method so we can recall it on clone (see the get_simulated_dataset method)
        self._compute_likelihood_biases()

        # This will save a clone of self for simulations
        self._clone = None

        # Integration method for the PSF (see psf_integration_method)
        self._psf_integration_method = "exact"

    @property
    def psf_integration_method(self):
        """
        Get or set the method for the integration of the PSF.

        * "exact" is more accurate but slow, if the position is free to vary it adds a lot of time to the fit. This is
        the default, to be used when the position of point sources are fixed. The computation in that case happens only
        once so the impact on the run time is negligible.
        * "fast" is less accurate (up to an error of few percent in flux) but a lot faster. This should be used when
        the position of the point source is free, because in that case the integration of the PSF happens every time
        the position changes, so several times during the fit.

        If you have a fit with a free position, use "fast". When the position is found, you can fix it, switch to
        "exact" and redo the fit to obtain the most accurate measurement of the flux. For normal sources the difference
        will be small, but for very bright sources it might be up to a few percent (most of the time < 1%). If you are
        interested in the localization contour there is no need to rerun with "exact".

        :param mode: either "exact" or "fast"
        :return: None
        """

        return self._psf_integration_method

    @psf_integration_method.setter
    def psf_integration_method(self, mode):
        assert mode.lower() in [
            "exact",
            "exactparallel",
            "fast",
            "ffast",
            "adaptive"
        ], "PSF integration method must be either 'exact' or 'fast'"

        self._psf_integration_method = mode.lower()

    def _setup_psf_convolutors(self):
        central_response_bins = self._response.get_response_dec_bin(
            self._roi.ra_dec_center[1]
        )

        self._psf_convolutors = collections.OrderedDict()
        for bin_id in central_response_bins:
            # Only set up PSF convolutors for active bins.
            if bin_id in self._active_planes:
                self._psf_convolutors[bin_id] = PSFConvolutor(
                    central_response_bins[bin_id].psf, self._flat_sky_projection
                )

    def _compute_likelihood_biases(self):
        for bin_label in self._maptree:
            data_analysis_bin = self._maptree[bin_label]

            this_log_factorial = np.sum(
                logfactorial(data_analysis_bin.observation_map.as_partial().astype(int))
            )
            self._log_factorials[bin_label] = this_log_factorial

            # As bias we use the likelihood value for the saturated model
            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()

            sat_model = np.clip(obs - bkg, 1e-50, None).astype(np.float64)

            self._saturated_model_like_per_maptree[bin_label] = (
                log_likelihood(obs, bkg, sat_model) - this_log_factorial
            )

    def get_saturated_model_likelihood(self):
        """
        Returns the likelihood for the saturated model (i.e. a model exactly equal to observation - background).

        :return:
        """
        return sum(self._saturated_model_like_per_maptree.values())

    def set_active_measurements(self, bin_id_min=None, bin_id_max=None, bin_list=None):
        """
        Set the active analysis bins to use during the analysis. It can be used in two ways:

        - Specifying a range: if the response and the maptree allows it, you can specify a minimum id and a maximum id
        number. This only works if the analysis bins are numerical, like in the normal fHit analysis. For example:

            > set_active_measurement(bin_id_min=1, bin_id_max=9)

        - Specifying a list of bins as strings. This is more powerful, as allows to select any bins, even
        non-contiguous bins. For example:

            > set_active_measurement(bin_list=[list])

        :param bin_id_min: minimum bin (only works for fHit analysis. For the others, use bin_list)
        :param bin_id_max: maximum bin (only works for fHit analysis. For the others, use bin_list)
        :param bin_list: a list of analysis bins to use
        :return: None
        """

        # Check for legal input
        if bin_id_min is not None:
            assert (
                bin_id_max is not None
            ), "If you provide a minimum bin, you also need to provide a maximum bin."

            # Make sure they are integers
            bin_id_min = int(bin_id_min)
            bin_id_max = int(bin_id_max)

            self._active_planes = []
            for this_bin in range(bin_id_min, bin_id_max + 1):
                this_bin = str(this_bin)
                if this_bin not in self._all_planes:
                    raise ValueError(f"Bin {this_bin} is not contained in this maptree.")

                self._active_planes.append(this_bin)

        else:
            assert (
                bin_id_max is None
            ), "If you provie a maximum bin, you also need to provide a minimum bin."

            assert bin_list is not None

            self._active_planes = []

            for this_bin in bin_list:
                # if not this_bin in self._all_planes:
                if this_bin not in self._all_planes:
                    raise ValueError(f"Bin {this_bin} is not contained in this maptree.")

                self._active_planes.append(this_bin)

        if self._likelihood_model:
            self.set_model(self._likelihood_model)

    def display(self, verbose=False):
        """
        Prints summary of the current object content.
        """

        log.info("Region of Interest: ")
        log.info("-------------------")
        self._roi.display()

        log.info("")
        log.info("Flat sky projection: ")
        log.info("--------------------")

        log.info(
            f"Width x height {self._flat_sky_projection.npix_width} x {self._flat_sky_projection.npix_height} px"
        )
        # log.info("Width x height: %s x %s px" % (self._flat_sky_projection.npix_width,
        #                                      self._flat_sky_projection.npix_height))
        log.info(f"Pixel sizes: {self._flat_sky_projection.pixel_size} deg")
        # log.info("Pixel sizes: %s deg" % self._flat_sky_projection.pixel_size)

        log.info("")
        log.info("Response: ")
        log.info("---------")

        self._response.display(verbose)

        log.info("")
        log.info("Map Tree: ")
        log.info("----------")

        self._maptree.display()

        log.info("")
        # log.info("Active energy/nHit planes ({}):".format(len(self._active_planes)))
        log.info(f"Active energy/nHit planes ({len(self._active_planes)}):")
        log.info("-------------------------------")
        log.info(self._active_planes)

    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        self._likelihood_model = likelihood_model_instance

        # Reset
        self._convolved_point_sources.reset()
        self._convolved_ext_sources.reset()

        # For each point source in the model, build the convolution class

        for source in list(self._likelihood_model.point_sources.values()):
            this_convolved_point_source = ConvolvedPointSource(
                source, self._response, self._flat_sky_projection
            )

            self._convolved_point_sources.append(this_convolved_point_source)

        # Samewise for extended sources
        ext_sources = list(self._likelihood_model.extended_sources.values())

        # NOTE: ext_sources evaluate to False if empty
        if ext_sources:
            # We will need to convolve

            self._setup_psf_convolutors()

            for source in ext_sources:
                if source.spatial_shape.n_dim == 2:
                    this_convolved_ext_source = ConvolvedExtendedSource2D(
                        source, self._response, self._flat_sky_projection
                    )

                else:
                    this_convolved_ext_source = ConvolvedExtendedSource3D(
                        source, self._response, self._flat_sky_projection
                    )

                self._convolved_ext_sources.append(this_convolved_ext_source)

    def get_excess_background(self, ra: float, dec: float, radius: float):
        """Calculates excess (data-bkg), background, and model counts at
        different radial distances from origin of radial profile.


        Parameters
        ----------
        ra : float
            RA of origin of radial profile
        dec : float
           Dec of origin of radial profile
        radius : float
           distance from origin of radial profile

        Returns
        -------
           returns a tuple of numpy arrays with info of areas (steradian) and
           signal excess, background, and model in units of counts to be used
           in the get_radial_profile method.
        """

        radius_radians = np.deg2rad(radius)

        total_counts = np.zeros(len(self._active_planes), dtype=float)
        background = np.zeros_like(total_counts)
        observation = np.zeros_like(total_counts)
        model = np.zeros_like(total_counts)
        signal = np.zeros_like(total_counts)
        area = np.zeros_like(total_counts)

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        longitude = ra_to_longitude(ra)
        latitude = dec
        center = hp.ang2vec(longitude, latitude, lonlat=True)

        for i, energy_id in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[energy_id]
            this_nside = data_analysis_bin.observation_map.nside

            radial_bin_pixels = hp.query_disc(
                this_nside, center, radius_radians, inclusive=False
            )

            # calculate the areas per bin by the product
            # of pixel area by the number of pixels at each radial bin
            area[i] = hp.nside2pixarea(this_nside) * radial_bin_pixels.shape[0]

            # NOTE: select active pixels according to each radial bin
            bin_active_pixel_indexes = np.intersect1d(
                self._active_pixels[energy_id], radial_bin_pixels, return_indices=True
            )[1]

            # obtain the excess, background, and expected excess at
            # each radial bin
            data = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()
            mdl = self._get_model_map(
                energy_id, n_point_sources, n_ext_sources
            ).as_partial()

            # select counts only from the pixels within specifid distance from
            # origin of radial profile
            bin_data = np.array([data[i] for i in bin_active_pixel_indexes])
            bin_bkg = np.array([bkg[i] for i in bin_active_pixel_indexes])
            bin_model = np.array([mdl[i] for i in bin_active_pixel_indexes])

            this_data_tot = np.sum(bin_data)
            this_bkg_tot = np.sum(bin_bkg)
            this_model_tot = np.sum(bin_model)

            background[i] = this_bkg_tot
            observation[i] = this_data_tot
            model[i] = this_model_tot
            signal[i] = this_data_tot - this_bkg_tot

        return area, signal, background, model

    def get_radial_profile(
        self,
        ra: float,
        dec: float,
        active_planes: list = None,
        max_radius: float = 3.0,
        n_radial_bins: int = 30,
        model_to_subtract: astromodels.Model = None,
        subtract_model_from_model: bool = False,
    ):
        """Calculates radial profiles for a source in units of excess counts
           per steradian

        Args:
            ra (float): RA of origin of radial profile
            dec (float): Declincation of origin of radial profile
            active_planes (np.ndarray, optional): List of active planes over
            which to average. Defaults to None.
            max_radius (float, optional): Radius up to which evaluate the
            radial profile. Defaults to 3.0.
            n_radial_bins (int, optional): Number of radial bins to use for
            the profile. Defaults to 30.
            model_to_subtract (astromodels.model, optional): Another model to
            subtract from the data excess. Defaults to None.
            subtract_model_from_model (bool, optional): If True, and
            model_to_subtract is not None,
            subtract model from model too. Defaults to False.

        Returns:
            tuple(np.ndarray): returns list of radial distances, excess expected
            counts, excess counts, counts uncertainty, and list of sorted active_planes
        """
        # default is to use all active bins
        if active_planes is None:
            active_planes = self._active_planes

        # Make sure we use bins with data
        good_planes = [plane_id in active_planes for plane_id in self._active_planes]
        plane_ids = set(active_planes) & set(self._active_planes)

        offset = 0.5
        delta_r = 1.0 * max_radius / n_radial_bins
        radii = np.array([delta_r * (r + offset) for r in range(n_radial_bins)])

        # Get area of all pixels in a given circle
        # The area of each ring is then given by the difference between two
        # subsequent circe areas.
        area = np.array(
            [self.get_excess_background(ra, dec, r + offset * delta_r)[0] for r in radii]
        )

        temp = area[1:] - area[:-1]
        area[1:] = temp

        # signals
        signal = np.array(
            [self.get_excess_background(ra, dec, r + offset * delta_r)[1] for r in radii]
        )

        temp = signal[1:] - signal[:-1]
        signal[1:] = temp

        # backgrounds
        bkg = np.array(
            [self.get_excess_background(ra, dec, r + offset * delta_r)[2] for r in radii]
        )

        temp = bkg[1:] - bkg[:-1]
        bkg[1:] = temp

        counts = signal + bkg

        # model
        # convert 'top hat' excess into 'ring' excesses.
        model = np.array(
            [self.get_excess_background(ra, dec, r + offset * delta_r)[3] for r in radii]
        )

        temp = model[1:] - model[:-1]
        model[1:] = temp

        if model_to_subtract is not None:
            this_model = copy.deepcopy(self._likelihood_model)
            self.set_model(model_to_subtract)

            model_subtract = np.array(
                [
                    self.get_excess_background(ra, dec, r + offset * delta_r)[3]
                    for r in radii
                ]
            )

            temp = model_subtract[1:] - model_subtract[:-1]
            model_subtract[1:] = temp

            signal -= model_subtract

            if subtract_model_from_model:
                model -= model_subtract

            self.set_model(this_model)

        # NOTE: weights are calculated as expected number of gamma-rays/number
        # of background counts.here, use max_radius to evaluate the number of
        # gamma-rays/bkg counts. The weights do not depend on the radius,
        # but fill a matrix anyway so there's no confusion when multiplying
        # them to the data later. Weight is normalized (sum of weights over
        # the bins = 1).

        np.array(self.get_excess_background(ra, dec, max_radius)[1])[good_planes]

        total_bkg = np.array(self.get_excess_background(ra, dec, max_radius)[2])[
            good_planes
        ]

        total_model = np.array(self.get_excess_background(ra, dec, max_radius)[3])[
            good_planes
        ]

        w = np.divide(total_model, total_bkg)
        weight = np.array([w / np.sum(w) for _ in radii])

        # restrict profiles to the user-specified analysis bins
        area = area[:, good_planes]
        signal = signal[:, good_planes]
        model = model[:, good_planes]
        counts = counts[:, good_planes]
        bkg = bkg[:, good_planes]

        # average over the analysis bins
        excess_data = np.average(signal / area, weights=weight, axis=1)
        excess_error = np.sqrt(np.sum(counts * weight * weight / (area * area), axis=1))
        excess_model = np.average(model / area, weights=weight, axis=1)

        return radii, excess_model, excess_data, excess_error, sorted(plane_ids)

    def plot_radial_profile(
        self,
        ra: float,
        dec: float,
        active_planes: list = None,
        max_radius: float = 3.0,
        n_radial_bins: int = 30,
        model_to_subtract: astromodels.Model = None,
        subtract_model_from_model: bool = False,
    ):
        """Plots radial profiles of data-background & model

        Args:
            ra (float): RA of origin of radial profile
            dec (float): Declination of origin of radial profile.
            active_planes (np.ndarray, optional): List of analysis bins over
            which to average.
            Defaults to None.
            max_radius (float, optional): Radius up to which the radial profile
            is evaluate; also used as the radius for the disk to calculate the
            gamma/hadron weights. Defaults to 3.0.
            n_radial_bins (int, optional): number of radial bins used for ring
            calculation. Defaults to 30.
            model_to_subtract (astromodels.model, optional): Another model that
            is to be subtracted from the data excess. Defaults to None.
            subtract_model_from_model (bool, optional): If True and
            model_to_subtract is not None, subtract from model too.
            Defaults to False.

        Returns:
            tuple(matplotlib.pyplot.Figure, pd.DataFrame): plot of data-background
            & model radial profile for source and a dataframe with all
            values for easy retrieval
        """

        (
            radii,
            excess_model,
            excess_data,
            excess_error,
            plane_ids,
        ) = self.get_radial_profile(
            ra,
            dec,
            active_planes,
            max_radius,
            n_radial_bins,
            model_to_subtract,
            subtract_model_from_model,
        )

        # add a dataframe for easy retrieval for calculations of surface
        # brighntess, if necessary.
        df = pd.DataFrame(columns=["Excess", "Bkg", "Model"], index=radii)
        df.index.name = "Radii"
        df["Excess"] = excess_data
        df["Bkg"] = excess_error
        df["Model"] = excess_model

        fig, ax = plt.subplots(figsize=(10, 8))

        plt.errorbar(
            radii,
            excess_data,
            yerr=excess_error,
            capsize=0,
            color="black",
            label="Excess (data-bkg)",
            fmt=".",
        )

        plt.plot(radii, excess_model, color="red", label="Model")

        plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", numpoints=1, fontsize=16)
        plt.axhline(0, color="deepskyblue", linestyle="--")

        x_limits = [0, max_radius]
        plt.xlim(x_limits)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.ylabel(r"Apparent Radial Excess [sr$^{-1}$]", fontsize=18)
        plt.xlabel(
            f"Distance from source at ({ra:0.2f} $^{{\circ}}$, {dec:0.2f} $^{{\circ}}$)",
            fontsize=18,
        )

        if len(plane_ids) == 1:
            title = f"Radial Profile, bin {plane_ids[0]}"

        else:
            title = "Radial Profile"
            # tmptitle = f"Radial Profile, bins \n{plane_ids}"
            # width = 80
            # title = "\n".join(
            # tmptitle[i : i + width] for i in range(0, len(tmptitle), width)
            # )
            # title = tmptitle

        plt.title(title)

        ax.grid(True)

        with contextlib.suppress(Exception):
            plt.tight_layout()
        # try:
        #
        # plt.tight_layout()
        #
        # except Exception:
        #
        # pass

        return fig, df

    def display_spectrum(self):
        """
        Make a plot of the current spectrum and its residuals (integrated over space)

        :return: a matplotlib.Figure
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        total_counts = np.zeros(len(self._active_planes), dtype=float)
        total_model = np.zeros_like(total_counts)
        model_only = np.zeros_like(total_counts)
        net_counts = np.zeros_like(total_counts)
        yerr_low = np.zeros_like(total_counts)
        yerr_high = np.zeros_like(total_counts)

        for i, energy_id in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[energy_id]

            this_model_map_hpx = self._get_expectation(
                data_analysis_bin, energy_id, n_point_sources, n_ext_sources
            )

            this_model_tot = np.sum(this_model_map_hpx)
            this_data_tot = np.sum(data_analysis_bin.observation_map.as_partial())
            this_bkg_tot = np.sum(data_analysis_bin.background_map.as_partial())

            total_counts[i] = this_data_tot
            net_counts[i] = this_data_tot - this_bkg_tot
            model_only[i] = this_model_tot

            this_wh_model = this_model_tot + this_bkg_tot
            total_model[i] = this_wh_model

            if this_data_tot >= 50.0:
                # Gaussian limit
                # Under the null hypothesis the data are distributed as a Gaussian with mu = model
                # and sigma = sqrt(model)
                # NOTE: since we neglect the background uncertainty, the background is part of the
                # model
                yerr_low[i] = np.sqrt(this_data_tot)
                yerr_high[i] = np.sqrt(this_data_tot)

            else:
                # Low-counts
                # Under the null hypothesis the data are distributed as a Poisson distribution with
                # mean = model, plot the 68% confidence interval (quantile=[0.16,1-0.16]).
                # NOTE: since we neglect the background uncertainty, the background is part of the
                # model
                quantile = 0.16
                mean = this_wh_model
                y_low = poisson.isf(1 - quantile, mu=mean)
                y_high = poisson.isf(quantile, mu=mean)
                yerr_low[i] = mean - y_low
                yerr_high[i] = y_high - mean

        residuals = old_div((total_counts - total_model), np.sqrt(total_model))
        residuals_err = [
            old_div(yerr_high, np.sqrt(total_model)),
            old_div(yerr_low, np.sqrt(total_model)),
        ]

        yerr = [yerr_high, yerr_low]

        return self._plot_spectrum(net_counts, yerr, model_only, residuals, residuals_err)

    def _plot_spectrum(self, net_counts, yerr, model_only, residuals, residuals_err):
        fig, subs = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [2, 1], "hspace": 0}, figsize=(14, 8)
        )
        planes = np.array(self._active_planes)
        subs[0].errorbar(
            planes,
            net_counts,
            yerr=yerr,
            capsize=0,
            color="black",
            label="Net counts",
            fmt=".",
        )

        subs[0].plot(planes, model_only, label="Convolved model")

        subs[0].legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", numpoints=1)

        # Residuals
        subs[1].axhline(0, linestyle="--")

        subs[1].errorbar(planes, residuals, yerr=residuals_err, capsize=0, fmt=".")

        y_limits = [min(net_counts[net_counts > 0]) / 2.0, max(net_counts) * 2.0]

        subs[0].set_yscale("log", nonpositive="clip")
        subs[0].set_ylabel("Counts per bin")
        subs[0].set_xticks([])

        subs[1].set_xlabel("Analysis bin")
        subs[1].set_ylabel(r"$\frac{{cts - mod - bkg}}{\sqrt{mod + bkg}}$")
        subs[1].set_xticks(planes)
        subs[1].set_xticklabels(self._active_planes, rotation=30)

        subs[0].set_ylim(y_limits)

        return fig

    @property
    def number_of_workers(self):
        """
        Get or set the number of workers to use for the parallelization of the computation of the likelihood.
        """
        return self._n_workers

    def get_log_like(self, individual_bins=False, return_null=False):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """
        # import time
        # start_time = time.time()
        
        # NOTE: multi-processing definition done a as global variable
        # done this way to avoid pickling issues
        # global process_bin

        if return_null is True:
            n_point_sources = 0
            n_ext_sources = 0
        else:
            n_point_sources = self._likelihood_model.get_number_of_point_sources()
            n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

            # Make sure that no source has been added since we filled the cache
            assert (
                n_point_sources == self._convolved_point_sources.n_sources_in_cache
                and n_ext_sources == self._convolved_ext_sources.n_sources_in_cache
            ), "The number of sources has changed. Please re-assign the model to the plugin."

        # This will hold the total log-likelihood
        total_log_like = 0
        log_like_per_bin = {}
        
        # Time statistics
        # time_stats = {
        #     'get_expectation': 0,
        #     'log_likelihood': 0,
        #     'total': 0
        # }

        for bin_id in self._active_planes:
            data_analysis_bin = self._maptree[bin_id]

            # Time get_expectation
            # t0 = time.time()
            this_model_map_hpx = self._get_expectation(
                data_analysis_bin, bin_id, n_point_sources, n_ext_sources
            )
            # time_stats['get_expectation'] += time.time() - t0

            # Now compare with observation
            bkg_renorm = list(self._nuisance_parameters.values())[0].value

            obs: np.ndarray = data_analysis_bin.observation_map.as_partial()
            bkg: np.ndarray = data_analysis_bin.background_map.as_partial() * bkg_renorm

            # Time log_likelihood calculation
            # t0 = time.time()
            this_pseudo_log_like = log_likelihood(obs, bkg, this_model_map_hpx)
            # time_stats['log_likelihood'] += time.time() - t0

            total_log_like += (
                this_pseudo_log_like
                - self._log_factorials[bin_id]
                - self._saturated_model_like_per_maptree[bin_id]
            )

            if individual_bins is True:
                log_like_per_bin[bin_id] = (
                    this_pseudo_log_like
                    - self._log_factorials[bin_id]
                    - self._saturated_model_like_per_maptree[bin_id]
                )

        # Calculate total time
        # time_stats['total'] = time.time() - start_time
        
        # Print time statistics (optional)
        # print("\nTime Statistics:")
        # print(f"Total time: {time_stats['total']:.4f} seconds")
        # print(f"  _get_expectation: {time_stats['get_expectation']:.4f} seconds ({time_stats['get_expectation']/time_stats['total']*100:.1f}%)")
        # print(f"  log_likelihood: {time_stats['log_likelihood']:.4f} seconds ({time_stats['log_likelihood']/time_stats['total']*100:.1f}%)")
        # print(f"  Other operations: {time_stats['total'] - time_stats['get_expectation'] - time_stats['log_likelihood']:.4f} seconds")

        if individual_bins is True:
            for k in log_like_per_bin:
                log_like_per_bin[k] /= total_log_like
            return total_log_like, log_like_per_bin
        # else:
        return total_log_like

    def write(self, response_file_name, map_tree_file_name):
        """
        Write this dataset to disk in HDF format.

        :param response_file_name: filename for the response
        :param map_tree_file_name: filename for the map tree
        :return: None
        """

        self._maptree.write(map_tree_file_name)
        self._response.write(response_file_name)

    def get_simulated_dataset(self, name):
        """
        Return a simulation of this dataset using the current model with current parameters.

        :param name: new name for the new plugin instance
        :return: a HAL instance
        """

        # First get expectation under the current model and store them, if we didn't do it yet

        if self._clone is None:
            n_point_sources = self._likelihood_model.get_number_of_point_sources()
            n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

            expectations = collections.OrderedDict()

            for bin_id in self._maptree:
                data_analysis_bin = self._maptree[bin_id]
                if bin_id not in self._active_planes:
                    expectations[bin_id] = None

                else:
                    expectations[bin_id] = (
                        self._get_expectation(
                            data_analysis_bin, bin_id, n_point_sources, n_ext_sources
                        )
                        + data_analysis_bin.background_map.as_partial()
                    )

            if parallel_client.is_parallel_computation_active():
                # Do not clone, as the parallel environment already makes clones

                clone = self

            else:
                clone = copy.deepcopy(self)

            self._clone = (clone, expectations)

        # Substitute the observation and background for each data analysis bin
        for bin_id in self._clone[0]._maptree:
            data_analysis_bin = self._clone[0]._maptree[bin_id]

            if bin_id not in self._active_planes:
                continue

            else:
                # Active plane. Generate new data
                expectation = self._clone[1][bin_id]
                new_data = np.random.poisson(
                    expectation, size=(1, expectation.shape[0])
                ).flatten()

                # Substitute data
                data_analysis_bin.observation_map.set_new_values(new_data)

        # Now change name and return
        self._clone[0]._name = name
        # Adjust the name of the nuisance parameter
        old_name = list(self._clone[0]._nuisance_parameters.keys())[0]
        new_name = old_name.replace(self.name, name)
        self._clone[0]._nuisance_parameters[new_name] = self._clone[
            0
        ]._nuisance_parameters.pop(old_name)

        # Recompute biases
        self._clone[0]._compute_likelihood_biases()

        return self._clone[0]

    def _get_expectation(self, data_analysis_bin, energy_bin_id, n_point_sources, n_ext_sources):
        # import time
        # 初始化时间统计字典
        # time_stats = {
        #     'point_sources': 0,
        #     'ext_sources_compute': 0,
        #     'ext_sources_reshape': 0,
        #     'psf_convolve': 0,
        #     'coordinate_transform': 0,
        #     'total': 0
        # }
        # total_start = time.perf_counter()

        # Compute the expectation from the model
        this_model_map = None

        # 统计点源处理时间
        # point_start = time.perf_counter()
        for pts_id in range(n_point_sources):
            this_conv_pnt_src: ConvolvedPointSource = self._convolved_point_sources[pts_id]

            expectation_per_transit = this_conv_pnt_src.get_source_map(
                energy_bin_id,
                tag=None,
                psf_integration_method=self._psf_integration_method,
            )

            expectation_from_this_source = (
                expectation_per_transit * data_analysis_bin.n_transits
            )

            if this_model_map is None:
                # First addition
                this_model_map = expectation_from_this_source
            else:
                this_model_map += expectation_from_this_source
        # time_stats['point_sources'] = time.perf_counter() - point_start

        # Now process extended sources
        if n_ext_sources > 0:
            # 统计延展源计算时间
            # ext_compute_start = time.perf_counter()
            this_ext_model_map = np.zeros(self._flat_sky_projection.ras.shape[0])

            def compute_expectation(ext_id):
                this_conv_src: Union[
                    ConvolvedExtendedSource2D, ConvolvedExtendedSource3D
                ] = self._convolved_ext_sources[ext_id]
                return this_conv_src.get_source_map(energy_bin_id)
            
            max_workers = os.getenv('MAX_WORKERS_PER_ENGINE', None)
            if max_workers is not None:
                max_workers = int(max_workers)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(compute_expectation, range(n_ext_sources)))
            
            for expectation_per_transit in results:
                if this_ext_model_map is None:
                    # First addition
                    this_ext_model_map[expectation_per_transit[0]] = expectation_per_transit[1]
                else:
                    this_ext_model_map[expectation_per_transit[0]] += expectation_per_transit[1]
            # time_stats['ext_sources_compute'] = time.perf_counter() - ext_compute_start

            # 统计延展源reshape时间
            # ext_reshape_start = time.perf_counter()
            this_ext_model_map = this_ext_model_map.reshape((
                self._flat_sky_projection.npix_height,
                self._flat_sky_projection.npix_width
            )).T
            # time_stats['ext_sources_reshape'] = time.perf_counter() - ext_reshape_start

            # 统计PSF卷积时间
            # psf_start = time.perf_counter()
            if this_model_map is None:
                # Only extended sources
                this_model_map = (
                    self._psf_convolutors[energy_bin_id].extended_source_image(
                        this_ext_model_map
                    )
                    * data_analysis_bin.n_transits
                )
            else:
                this_model_map += (
                    self._psf_convolutors[energy_bin_id].extended_source_image(
                        this_ext_model_map
                    )
                    * data_analysis_bin.n_transits
                )
            # time_stats['psf_convolve'] = time.perf_counter() - psf_start

        # 统计坐标转换时间
        # transform_start = time.perf_counter()
        # Now transform from the flat sky projection to HEALPiX
        if this_model_map is not None:
            # First divide for the pixel area because we need to interpolate brightness
            this_model_map = (
                this_model_map / self._flat_sky_projection.project_plane_pixel_area
            )

            this_model_map_hpx = self._flat_sky_to_healpix_transform[energy_bin_id](
                this_model_map, fill_value=0.0
            )

            # Now multiply by the pixel area of the new map to go back to flux
            this_model_map_hpx *= hp.nside2pixarea(data_analysis_bin.nside, degrees=True)
        else:
            # No sources
            this_model_map_hpx = 0.0
        # time_stats['coordinate_transform'] = time.perf_counter() - transform_start

        # 计算总时间
        # time_stats['total'] = time.perf_counter() - total_start

        # 打印时间统计信息
        # print("\n_get_expectation 时间统计:")
        # print(f"总耗时: {time_stats['total']:.4f}秒")
        # print(f"  点源处理: {time_stats['point_sources']:.4f}秒 ({time_stats['point_sources']/time_stats['total']*100:.1f}%)")
        # if n_ext_sources > 0:
        #     print(f"  延展源计算: {time_stats['ext_sources_compute']:.4f}秒 ({time_stats['ext_sources_compute']/time_stats['total']*100:.1f}%)")
        #     print(f"  延展源reshape: {time_stats['ext_sources_reshape']:.4f}秒 ({time_stats['ext_sources_reshape']/time_stats['total']*100:.1f}%)")
        #     print(f"  PSF卷积: {time_stats['psf_convolve']:.4f}秒 ({time_stats['psf_convolve']/time_stats['total']*100:.1f}%)")
        # print(f"  坐标转换: {time_stats['coordinate_transform']:.4f}秒 ({time_stats['coordinate_transform']/time_stats['total']*100:.1f}%)")

        return this_model_map_hpx

    @staticmethod
    def _represent_healpix_map(
        fig, hpx_map, longitude, latitude, xsize, resolution, smoothing_kernel_sigma
    ):
        proj = get_gnomonic_projection(
            fig, hpx_map, rot=(longitude, latitude, 0.0), xsize=xsize, reso=resolution
        )

        if smoothing_kernel_sigma is not None:
            # Get the sigma in pixels
            sigma = old_div(smoothing_kernel_sigma * 60, resolution)

            proj = convolve(
                list(proj),
                Gaussian2DKernel(sigma),
                nan_treatment="fill",
                preserve_nan=True,
            )

        return proj

    def display_fit(self, smoothing_kernel_sigma=0.1, display_colorbar=False):
        """
        Make a figure containing 4 maps for each active analysis bins with respectively model, data,
        background and residuals. The model, data and residual maps are smoothed, the background
        map is not.

        :param smoothing_kernel_sigma: sigma for the Gaussian smoothing kernel, for all but
        background maps
        :param display_colorbar: whether or not to display the colorbar in the residuals
        :return: a matplotlib.Figure
        """

        n_point_sources = self._likelihood_model.get_number_of_point_sources()
        n_ext_sources = self._likelihood_model.get_number_of_extended_sources()

        # This is the resolution (i.e., the size of one pixel) of the image
        resolution = 3.0  # arcmin

        # The image is going to cover the diameter plus 20% padding
        xsize = self._get_optimal_xsize(resolution)

        n_active_planes = len(self._active_planes)
        n_columns = 4

        fig, subs = plt.subplots(
            n_active_planes,
            n_columns,
            figsize=(2.7 * n_columns, n_active_planes * 2),
            squeeze=False,
        )

        prog_bar = tqdm(total=len(self._active_planes), desc="Smoothing planes")

        images = ["None"] * n_columns

        for i, plane_id in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[plane_id]

            # Get the center of the projection for this plane
            this_ra, this_dec = self._roi.ra_dec_center

            # Make a full healpix map for a second
            whole_map = self._get_model_map(
                plane_id, n_point_sources, n_ext_sources
            ).as_dense()

            # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
            longitude = ra_to_longitude(this_ra)

            # Declination is already between -90 and 90
            latitude = this_dec

            # Background and excess maps
            bkg_subtracted, _, background_map = self._get_excess(
                data_analysis_bin, all_maps=True
            )

            # Make all the projections: model, excess, background, residuals
            proj_model = self._represent_healpix_map(
                fig,
                whole_map,
                longitude,
                latitude,
                xsize,
                resolution,
                smoothing_kernel_sigma,
            )
            # Here we removed the background otherwise nothing is visible
            # Get background (which is in a way "part of the model" since the uncertainties are neglected)
            proj_data = self._represent_healpix_map(
                fig,
                bkg_subtracted,
                longitude,
                latitude,
                xsize,
                resolution,
                smoothing_kernel_sigma,
            )
            # No smoothing for this one (because a goal is to check it is smooth).
            proj_bkg = self._represent_healpix_map(
                fig, background_map, longitude, latitude, xsize, resolution, None
            )
            proj_residuals = proj_data - proj_model

            # Common color scale range for model and excess maps
            vmin = min(np.nanmin(proj_model), np.nanmin(proj_data))
            vmax = max(np.nanmax(proj_model), np.nanmax(proj_data))

            # Plot model
            images[0] = subs[i][0].imshow(
                proj_model, origin="lower", vmin=vmin, vmax=vmax
            )
            subs[i][0].set_title("model, bin {}".format(data_analysis_bin.name))

            # Plot data map
            images[1] = subs[i][1].imshow(proj_data, origin="lower", vmin=vmin, vmax=vmax)
            subs[i][1].set_title("excess, bin {}".format(data_analysis_bin.name))

            # Plot background map.
            images[2] = subs[i][2].imshow(proj_bkg, origin="lower")
            subs[i][2].set_title("background, bin {}".format(data_analysis_bin.name))

            # Now residuals
            images[3] = subs[i][3].imshow(proj_residuals, origin="lower")
            subs[i][3].set_title("residuals, bin {}".format(data_analysis_bin.name))

            # Remove numbers from axis
            for j in range(n_columns):
                subs[i][j].axis("off")

            if display_colorbar:
                for j, image in enumerate(images):
                    plt.colorbar(image, ax=subs[i][j])

            prog_bar.update(1)

        # fig.set_tight_layout(True)
        fig.set_layout_engine("tight")

        return fig

    def _get_optimal_xsize(self, resolution):
        return 2.2 * self._roi.data_radius.to("deg").value / (resolution / 60.0)

    def display_stacked_image(self, smoothing_kernel_sigma=0.5):
        """
        Display a map with all active analysis bins stacked together.

        :param smoothing_kernel_sigma: sigma for the Gaussian smoothing kernel to apply
        :return: a matplotlib.Figure instance
        """

        # This is the resolution (i.e., the size of one pixel) of the image in arcmin
        resolution = 3.0

        # The image is going to cover the diameter plus 20% padding
        xsize = self._get_optimal_xsize(resolution)

        active_planes_bins = [self._maptree[x] for x in self._active_planes]

        # Get the center of the projection for this plane
        this_ra, this_dec = self._roi.ra_dec_center

        # Healpix uses longitude between -180 and 180, while R.A. is between 0 and 360. We need to fix that:
        longitude = ra_to_longitude(this_ra)

        # Declination is already between -90 and 90
        latitude = this_dec

        total = None

        for i, data_analysis_bin in enumerate(active_planes_bins):
            # Plot data
            background_map = data_analysis_bin.background_map.as_dense()
            this_data = data_analysis_bin.observation_map.as_dense() - background_map
            idx = np.isnan(this_data)
            # this_data[idx] = hp.UNSEEN

            if i == 0:
                total = this_data

            else:
                # Sum only when there is no UNSEEN, so that the UNSEEN pixels will stay UNSEEN
                total[~idx] += this_data[~idx]

        delta_coord = (self._roi.data_radius.to("deg").value * 2.0) / 15.0

        fig, sub = plt.subplots(1, 1)

        proj = self._represent_healpix_map(
            fig, total, longitude, latitude, xsize, resolution, smoothing_kernel_sigma
        )

        cax = sub.imshow(proj, origin="lower")
        fig.colorbar(cax)
        sub.axis("off")

        hp.graticule(delta_coord, delta_coord)

        return fig

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()

    def get_number_of_data_points(self):
        """
        Return the number of active bins across all active analysis bins

        :return: number of active bins
        """

        n_points = 0

        for bin_id in self._maptree:
            n_points += self._maptree[bin_id].observation_map.as_partial().shape[0]

        return n_points

    def _get_model_map(self, plane_id, n_pt_src, n_ext_src):
        """
        This function returns a model map for a particular bin
        """

        if plane_id not in self._active_planes:
            raise ValueError(f"{plane_id} not a plane in the current model")

        model_map = SparseHealpix(
            self._get_expectation(self._maptree[plane_id], plane_id, n_pt_src, n_ext_src),
            self._active_pixels[plane_id],
            self._maptree[plane_id].observation_map.nside,
        )

        return model_map

    def _get_excess(self, data_analysis_bin, all_maps=True):
        """
        This function returns the excess counts for a particular bin
        if all_maps=True, also returns the data and background maps
        """
        data_map = data_analysis_bin.observation_map.as_dense()
        bkg_map = data_analysis_bin.background_map.as_dense()
        excess = data_map - bkg_map

        if all_maps:
            return excess, data_map, bkg_map
        return excess

    def _write_a_map(self, file_name, which, fluctuate=False, return_map=False):
        """
        This writes either a model map or a residual map, depending on which one is preferred
        """
        which = which.lower()
        assert which in ["model", "residual"]

        n_pt = self._likelihood_model.get_number_of_point_sources()
        n_ext = self._likelihood_model.get_number_of_extended_sources()

        map_analysis_bins = collections.OrderedDict()

        if fluctuate:
            poisson_set = self.get_simulated_dataset("model map")

        for plane_id in self._active_planes:
            data_analysis_bin = self._maptree[plane_id]

            bkg = data_analysis_bin.background_map
            obs = data_analysis_bin.observation_map

            if fluctuate:
                model_excess = (
                    poisson_set._maptree[plane_id].observation_map
                    - poisson_set._maptree[plane_id].background_map
                )
            else:
                model_excess = self._get_model_map(plane_id, n_pt, n_ext)

            if which == "residual":
                bkg += model_excess

            if which == "model":
                obs = model_excess + bkg

            this_bin = DataAnalysisBin(
                plane_id,
                observation_hpx_map=obs,
                background_hpx_map=bkg,
                active_pixels_ids=self._active_pixels[plane_id],
                n_transits=data_analysis_bin.n_transits,
                scheme="RING",
            )

            map_analysis_bins[plane_id] = this_bin

        # save the file
        new_map_tree = MapTree(map_analysis_bins, self._roi)
        new_map_tree.write(file_name)

        if return_map:
            return new_map_tree

    def write_model_map(self, file_name, poisson_fluctuate=False, test_return_map=False):
        """
        This function writes the model map to a file.
        The interface is based off of HAWCLike for consistency
        """
        if test_return_map:
            log.warning("test_return_map=True should only be used for testing purposes!")
        return self._write_a_map(file_name, "model", poisson_fluctuate, test_return_map)

    def write_residual_map(self, file_name, test_return_map=False):
        """
        This function writes the residual map to a file.
        The interface is based off of HAWCLike for consistency
        """
        if test_return_map:
            log.warning("test_return_map=True should only be used for testing purposes!")
        return self._write_a_map(file_name, "residual", False, test_return_map)

    def cal_TS_all(self):
        '''
        This function calculate the TS of all analysis bins. 

        :return: TS_all
        '''
        TS_all=0
        for i, this_bin in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[this_bin]

            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()

            zero_model=np.zeros(len(obs))
            this_model_map_hpx = self._get_expectation(data_analysis_bin, this_bin, self._likelihood_model.get_number_of_point_sources(), self._likelihood_model.get_number_of_extended_sources())

            TS_all += -2*(log_likelihood(obs,bkg,zero_model)-log_likelihood(obs,bkg,this_model_map_hpx))

        return TS_all
    
    def get_bkg_llh(self):
        '''
        This function calculate the bkg log likelihood of all analysis bins. 

        :return: bkg_llh
        '''
        bkg_llh=0
        source_llh=0
        
        for i, this_bin in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[this_bin]

            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()
            zero_model=np.zeros(len(obs))
            # this_model_map_hpx = self._get_expectation(data_analysis_bin, this_bin, self._likelihood_model.get_number_of_point_sources(), self._likelihood_model.get_number_of_extended_sources())
            bkg_llh += log_likelihood(obs,bkg,zero_model) - self._log_factorials[this_bin] - self._saturated_model_like_per_maptree[this_bin]
            # source_llh += log_likelihood(obs,bkg,this_model_map_hpx) 
        return bkg_llh

    def cal_TS_array(self,n_pts,n_exts):
        '''
        This function calculate the TS of each analysis bins. 

        :return: TS_array
        '''
        TS_array=[]
        for i, this_bin in enumerate(self._active_planes):
            data_analysis_bin = self._maptree[this_bin]

            obs = data_analysis_bin.observation_map.as_partial()
            bkg = data_analysis_bin.background_map.as_partial()

            zero_model=np.zeros(len(obs))
            this_model_map_hpx = self._get_expectation(data_analysis_bin,this_bin, n_pts,n_exts)

            TS_array.append(-2*(log_likelihood(obs,bkg,zero_model)-log_likelihood(obs,bkg,this_model_map_hpx)))
        #print(TS_all)

        return TS_array

    def write_each_model_map(self,filename):
        '''
        This function wtites each model map to files
        '''
        prog_bar = tqdm(total = len(self._active_planes), desc="Write maps of analysis bins")

        n_pts = self._likelihood_model.get_number_of_point_sources()
        n_ext = self._likelihood_model.get_number_of_extended_sources()
        for bin_id, this_bin in enumerate(self._active_planes):
            pixid = self._active_pixels[this_bin]
            data_analysis_bin=self._maptree._analysis_bins[this_bin]
            obs_raw=data_analysis_bin.observation_map.as_partial()
            bkg_raw=data_analysis_bin.background_map.as_partial()
            
            for pts_id in range(n_pts):
                this_conv_src = self._convolved_point_sources[pts_id]
                expectation_per_transit = this_conv_src.get_source_map(this_bin,tag=None,psf_integration_method=self._psf_integration_method)
                expectation_from_this_source = expectation_per_transit * data_analysis_bin.n_transits
                this_model_map = expectation_from_this_source
                this_model_map = old_div(this_model_map, self._flat_sky_projection.project_plane_pixel_area)
                this_model_map_hpx = self._flat_sky_to_healpix_transform[this_bin](this_model_map, fill_value=0.0)
                this_model_map_hpx *= hp.nside2pixarea(data_analysis_bin.nside, degrees=True)
                _ON=np.zeros(hp.nside2npix(data_analysis_bin.nside))
                _BK=np.zeros(hp.nside2npix(data_analysis_bin.nside))
                _Model=np.zeros(hp.nside2npix(data_analysis_bin.nside))
                for index,_pix_id in enumerate(pixid):
                    _ON[_pix_id] = obs_raw[index]
                    _BK[_pix_id] = bkg_raw[index]
                    _Model[_pix_id] = this_model_map_hpx[index]
                hp.write_map("%s_pts%d_bin%s.fits.gz"%(filename,pts_id,this_bin),[_ON,_BK,_Model],overwrite=True)



            if(n_ext>0):
                for ext_id in range(n_ext):
                    this_conv_src = self._convolved_ext_sources[ext_id]
                    expectation_per_transit = this_conv_src.get_source_map(this_bin)
                    this_ext_model_map = expectation_per_transit
                    this_model_map = (self._psf_convolutors[this_bin].extended_source_image(this_ext_model_map) * data_analysis_bin.n_transits)
                    this_model_map = old_div(this_model_map, self._flat_sky_projection.project_plane_pixel_area)
                    this_model_map_hpx = self._flat_sky_to_healpix_transform[this_bin](this_model_map, fill_value=0.0)
                    this_model_map_hpx *= hp.nside2pixarea(data_analysis_bin.nside, degrees=True)
                    _ON=np.zeros(hp.nside2npix(data_analysis_bin.nside))
                    _BK=np.zeros(hp.nside2npix(data_analysis_bin.nside))
                    _Model=np.zeros(hp.nside2npix(data_analysis_bin.nside))
                    for index,_pix_id in enumerate(pixid):
                        _ON[_pix_id] = obs_raw[index]
                        _BK[_pix_id] = bkg_raw[index]
                        _Model[_pix_id] = this_model_map_hpx[index]
                    hp.write_map("%s_ext%d_bin%s.fits.gz"%(filename,ext_id,this_bin),[_ON,_BK,_Model],overwrite=True)

            prog_bar.update(1)

    #def (self):
    #    '''
    #    return:
    #    '''

    def define_Nexts(self,num=20):
        '''
        define N extend sources for user
        return: sources list
        '''
        source=[]
        for i in range(num):
            #spectrum = Powerlaw()
            spectrum = Log_parabola()
            shape = Gaussian_on_sphere()
            source1 = ExtendedSource("s%d"%(i),spatial_shape=shape,spectral_shape=spectrum)
            source.append(source1)
        return source

    def define_Nexts_PL(self,num=10):
        '''
        define N extend sources for user
        return: sources list
        '''
        source=[]
        for i in range(num):
            spectrum = Powerlaw()
            shape = Gaussian_on_sphere()
    
            source1 = ExtendedSource("s%d"%(i),spatial_shape=shape,spectral_shape=spectrum)
            source.append(source1)
        return source
        


    def define_Npts(self, ra_ , dec_,num=10):
        '''
        define N point sources for user
        return: sources list
        '''
        source=[]
        for i in range(num):
            spectrum = Powerlaw()
            source1 = PointSource("s%d"%(i),ra=ra_, dec=dec_,spectral_shape=spectrum)
            source.append(source1)
        return source
    
    def calcu_flux_of_every_bins(self,source_array,instrument):
        '''Only fit the spectrum.K for plotting  points on the spectra'''
        '''prarm1: source array [src1,src2,...] '''
        '''prarm2: instrument[WCDA or KM2A]'''
        '''return: spectrum.K'''        

        lm_=Model()
        instrument_copy= copy.copy(instrument)
        source_copy_array=[]
         
        for source in source_array:
            source_copy = copy.copy(source)
            if(str(source_copy.free_parameters.keys()).find("Cutoff_powerlaw")>0):
                source_copy.spectrum.main.Cutoff_powerlaw.index.fix=True
                source_copy.spectrum.main.Cutoff_powerlaw.xc.fix=True
            elif(str(source_copy.free_parameters.keys()).find("Powerlaw")>0):
                source_copy.spectrum.main.Powerlaw.index.fix=True
            elif(str(source_copy.free_parameters.keys()).find("Log_parabola")>0):
                source_copy.spectrum.main.Log_parabola.alpha.fix=True
                source_copy.spectrum.main.Log_parabola.beta.fix=True
            if(str(source_copy.free_parameters.keys()).find("position")>0):
                source_copy.position.ra.fix=True
                source_copy.position.dec.fix=True
            if(str(source_copy.free_parameters.keys()).find("Disk_on_sphere")>0):
                source_copy.Disk_on_sphere.lon0.fix=True
                source_copy.Disk_on_sphere.lat0.fix=True
                source_copy.Disk_on_sphere.radius.fix=True
            if(str(source_copy.free_parameters.keys()).find("Gaussian_on_sphere")>0):
                source_copy.Gaussian_on_sphere.lon0.fix=True
                source_copy.Gaussian_on_sphere.lat0.fix=True
                source_copy.Gaussian_on_sphere.sigma.fix=True

        #source_copy = copy.deepcopy(source)
            lm_.add_source(source_copy)
        #lm_.display(complete=True )
            source_copy_array.append(source_copy)

        source_flux_array = [[0] * len(instrument._active_planes) for _ in range(len(source_array))]
        for i_,inhit in enumerate(instrument._active_planes):
            instrument_copy.set_active_measurements(inhit,inhit)
            datalist_ = DataList(instrument_copy)
            jl_ = JointLikelihood(lm_, datalist_, verbose=False)
            jl_.set_minimizer("minuit")
            param_df1, like_df1 = jl_.fit()

            b=instrument._response.get_response_dec_bin(instrument._roi.ra_dec_center[1])[inhit]
            flux_yy=np.zeros(len(b.sim_signal_events_per_bin), dtype=float)
            iii=np.linspace(-3,4,len(b.sim_signal_events_per_bin))
            th1=ROOT.TH1D("","",len(b.sim_signal_events_per_bin)\
                  ,np.log10(b.sim_energy_bin_low[0]),np.log10(b.sim_energy_bin_hi[len(b.sim_signal_events_per_bin)-1]))

            for isrc,src in enumerate(source_copy_array):
                for j in range(len(b.sim_signal_events_per_bin)):

                #print(np.sqrt(b.sim_energy_bin_hi[j]*b.sim_energy_bin_low[j]))
                    if(str(src.free_parameters.keys()).find("Cutoff")>0):
                        _flux =  b.sim_signal_events_per_bin[j] \
                            * src.spectrum.main.Cutoff_powerlaw(1e9*np.sqrt(b.sim_energy_bin_hi[j]*b.sim_energy_bin_low[j]))/b.sim_differential_photon_fluxes[j]
                    elif(str(src.free_parameters.keys()).find("Pow")>0):
                        _flux =  b.sim_signal_events_per_bin[j] \
                            * src.spectrum.main.Powerlaw(1e9*np.sqrt(b.sim_energy_bin_hi[j]*b.sim_energy_bin_low[j]))/b.sim_differential_photon_fluxes[j]
                    elif(str(src.free_parameters.keys()).find("Log_")>0):
                        _flux =  b.sim_signal_events_per_bin[j] \
                            * src.spectrum.main.Log_parabola(1e9*np.sqrt(b.sim_energy_bin_hi[j]*b.sim_energy_bin_low[j]))/b.sim_differential_photon_fluxes[j]
                    th1.SetBinContent(j+1,1e9*_flux)
                    flux_yy[j]=_flux

                x_ = ctypes.c_double(1.0)
                quanti_ = ctypes.c_double(0.5)
            #fig=plt.figure(figsize=(10,8))
            #plt.scatter(iii,flux_yy)
                th1.GetQuantiles(1,x_,quanti_)
                #print("median= %f , %f TeV"%(np.double(x_),pow(10.,np.double(x_))))
                if(str(src.free_parameters.keys()).find("Cutoff_powerlaw")>0):
                    source_flux_array[isrc][i_]=[(pow(10.,np.double(x_))),\
                                             src.spectrum.main.Cutoff_powerlaw(1e9*pow(10.,np.double(x_))),\
                                             src.spectrum.main.Cutoff_powerlaw(1e9*pow(10.,np.double(x_)))*param_df1.values[isrc][3]/param_df1.values[isrc][0]]
                elif(str(src.free_parameters.keys()).find("Pow")>0):
                    source_flux_array[isrc][i_]=[(pow(10.,np.double(x_))),\
                                             src.spectrum.main.Powerlaw(1e9*pow(10.,np.double(x_))),\
                                             src.spectrum.main.Powerlaw(1e9*pow(10.,np.double(x_)))*param_df1.values[isrc][3]/param_df1.values[isrc][0]]
                elif(str(src.free_parameters.keys()).find("Log_")>0):
                    source_flux_array[isrc][i_]=[pow(10.,np.double(x_)),\
                                             src.spectrum.main.Log_parabola(1e9*pow(10.,np.double(x_))),\
                                             src.spectrum.main.Log_parabola(1e9*pow(10.,np.double(x_)))*param_df1.values[isrc][3]/param_df1.values[isrc][0]]

        for source in source_array:
            source_copy = copy.copy(source)
            if(str(source_copy.parameters.keys()).find("Cutoff_powerlaw")>0):
                source_copy.spectrum.main.Cutoff_powerlaw.index.fix=False
                source_copy.spectrum.main.Cutoff_powerlaw.xc.fix=False

            elif(str(source_copy.parameters.keys()).find("Powerlaw")>0):
                source_copy.spectrum.main.Powerlaw.index.fix=False
            elif(str(source_copy.parameters.keys()).find("Log_parabola")>0):
                source_copy.spectrum.main.Log_parabola.alpha.fix=False
                source_copy.spectrum.main.Log_parabola.beta.fix=False
            if(str(source_copy.parameters.keys()).find("position")>0):
                source_copy.position.ra.free=True
                source_copy.position.dec.free=True
            if(str(source_copy.parameters.keys()).find("Disk_on_sphere")>0):
                source_copy.Disk_on_sphere.lon0.free=True
                source_copy.Disk_on_sphere.lat0.free=True
                source_copy.Disk_on_sphere.radius.free=True
            if(str(source_copy.parameters.keys()).find("Gaussian_on_sphere")>0):
                source_copy.Gaussian_on_sphere.lon0.free=True
                source_copy.Gaussian_on_sphere.lat0.free=True
                source_copy.Gaussian_on_sphere.sigma.free=True

        return source_flux_array



    #def write_expectation(self, data_analysis_bin, energy_bin_id, n_point_sources, n_ext_sources):
    def write_Model_map(self,filename ,binid_start,binid_stop,n_point_sources,n_ext_sources):

        # write the expectation from the model
        pixid=self._roi.active_pixels(1024)
        NUM_of_sources=n_point_sources+n_ext_sources
        model_    = [0] * NUM_of_sources
        #model_hpx = [0] * NUM_of_sources
        map_hpx   = [np.zeros(1024*1024*12)] * NUM_of_sources
        for bid in range(binid_start,binid_stop+1):
            energy_bin_id=str(bid)
            data_analysis_bin=self._maptree._analysis_bins[energy_bin_id]

            if(n_point_sources > 0):
                for pts_id in range(n_point_sources):

                    this_conv_src = self._convolved_point_sources[pts_id]

                    expectation_per_transit = this_conv_src.get_source_map(energy_bin_id,
                                                                           tag=None,
                                                                           psf_integration_method=self._psf_integration_method)

                    expectation_from_this_source = expectation_per_transit * data_analysis_bin.n_transits

                    model_[pts_id] +=expectation_from_this_source

            # Now process extended sources
            if(n_ext_sources > 0):
                for ext_id in range(n_ext_sources):
                    this_conv_src = self._convolved_ext_sources[ext_id]
                    expectation_per_transit = this_conv_src.get_source_map(energy_bin_id)
                    # Now convolve with the PSF
                    this_model_map = (self._psf_convolutors[energy_bin_id].extended_source_image(expectation_per_transit) *
                                        data_analysis_bin.n_transits)
                    
                    model_[n_point_sources+ext_id] += this_model_map

            # Now transform from the flat sky projection to HEALPiX
            # First divide for the pixel area because we need to interpolate brightness
        for srcs in range(NUM_of_sources):
            this_model_map = old_div(model_[srcs], self._flat_sky_projection.project_plane_pixel_area)

            this_model_map_hpx = self._flat_sky_to_healpix_transform[energy_bin_id](this_model_map, fill_value=0.0)

            #model_hpx[srcs] = this_model_map_hpx * hp.nside2pixarea(data_analysis_bin.nside, degrees=True)
            map_hpx[srcs][pixid] = this_model_map_hpx * hp.nside2pixarea(data_analysis_bin.nside, degrees=True)

        hp.write_map("%s_bin_%d_to_%d.fits.gz"%(filename,binid_start,binid_stop),map_hpx,overwrite=True)
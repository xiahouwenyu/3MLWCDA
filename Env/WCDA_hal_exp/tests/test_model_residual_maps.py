from __future__ import division
from past.utils import old_div
import pytest
from os.path import dirname
from WCDA_hal import HealpixConeROI, HAL
from WCDA_hal.maptree import map_tree_factory
from threeML import Model, JointLikelihood, DataList
import astropy.units as u
from astromodels import PointSource, ExtendedSource, Powerlaw, Gaussian_on_sphere

from conftest import check_map_trees

try:
    import ROOT
except:
    has_root = False
else:
    has_root = True

skip_if_ROOT_is_not_available = pytest.mark.skipif(
    not has_root, reason="No ROOT available"
)

@skip_if_ROOT_is_not_available
def test_model_residual_maps(geminga_maptree, geminga_response, geminga_roi):

    #data_radius = 5.0
    #model_radius = 7.0
    output = dirname(geminga_maptree)

    ra_src, dec_src = 101.7, 16.0
    maptree, response, roi  = geminga_maptree, geminga_response, geminga_roi

    WCDA = HAL("WCDA", maptree, response, roi)

    # Use from bin 1 to bin 9
    WCDA.set_active_measurements(1, 9)

    # Display information about the data loaded and the ROI
    WCDA.display()

    '''
    Define model: Two sources, 1 point, 1 extended

    Same declination, but offset in RA

    Different spectral index, but both power laws

    '''
    pt_shift=3.0
    ext_shift = 2.0

    # First source
    spectrum1 = Powerlaw()
    source1 = PointSource("point", ra=ra_src + pt_shift, dec=dec_src, spectral_shape=spectrum1)

    spectrum1.K = 1e-12 / (u.TeV * u.cm ** 2 * u.s)
    spectrum1.piv = 1 * u.TeV
    spectrum1.index = -2.3

    spectrum1.piv.fix = True
    spectrum1.K.fix = True
    spectrum1.index.fix = True

    # Second source
    shape = Gaussian_on_sphere(lon0=ra_src - ext_shift, lat0=dec_src, sigma=0.3)
    spectrum2 = Powerlaw()
    source2 = ExtendedSource("extended", spatial_shape=shape, spectral_shape=spectrum2)

    spectrum2.K = 1e-12 / (u.TeV * u.cm ** 2 * u.s)
    spectrum2.piv = 1 * u.TeV
    spectrum2.index = -2.0  

    shape.lon0.fix=True
    shape.lat0.fix=True
    shape.sigma.fix=True

    spectrum2.piv.fix = True
    spectrum2.K.fix = True
    spectrum2.index.fix = True

    # Define model with both sources
    model = Model(source1, source2)

    # Define the data we are using
    data = DataList(WCDA)

    # Define the JointLikelihood object (glue the data to the model)
    jl = JointLikelihood(model, data, verbose=False)

    # This has the effect of loading the model cache 
    fig = WCDA.display_spectrum()

    # the test file names
    model_file_name = "{0}/test_model.hdf5".format(output)
    residual_file_name = "{0}/test_residual.hdf5".format(output)

    # Write the map trees for testing
    model_map_tree = WCDA.write_model_map(model_file_name, poisson_fluctuate=True, test_return_map=True)
    residual_map_tree = WCDA.write_residual_map(residual_file_name, test_return_map=True)

    # Read the maps back in
    WCDA_model = map_tree_factory(model_file_name,roi)
    WCDA_residual = map_tree_factory(residual_file_name,roi)


    check_map_trees(WCDA_model, model_map_tree)
    check_map_trees(WCDA_residual, residual_map_tree)

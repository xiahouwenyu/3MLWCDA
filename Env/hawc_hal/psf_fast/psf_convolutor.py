# ==============================================================================
# 顶部导入和配置 (保持不变)
# ==============================================================================
# ... (所有 imports 和 PYFFTW_AVAILABLE 的设置) ...
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
import time

from numpy.fft import rfftn, irfftn
try:
    import pyfftw
    from pyfftw.interfaces import numpy_fft as pyfftw_fft
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(300)
    PYFFTW_AVAILABLE = True
    print("PyFFTW 已找到并启用缓存。")
except ImportError:
    PYFFTW_AVAILABLE = False
    print("警告：pyfftw 未安装。")
# PYFFTW_AVAILABLE = False  # 强制使用 NumPy 的 FFT 实现
# ==============================================================================
from scipy.fftpack import helper
from .psf_interpolator import PSFInterpolator
from .psf_wrapper import PSFWrapper


# ==============================================================================
# 辅助函数 _centered (保持不变)
# ==============================================================================
def _centered(arr, newsize):
    # ...
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


# ==============================================================================
# 最终的 PSFConvolutor 类
# ==============================================================================
class PSFConvolutor(object):
    
    def __init__(self, psf_wrapper, flat_sky_proj):
        # --- 初始化代码基本保持原样 ---
        self._psf = psf_wrapper
        self._flat_sky_proj = flat_sky_proj
        interpolator = PSFInterpolator(psf_wrapper, flat_sky_proj)
        psf_stamp = interpolator.point_source_image(flat_sky_proj.ra_center, flat_sky_proj.dec_center)
        kernel_radius_px = int(np.ceil(old_div(self._psf.kernel_radius, flat_sky_proj.pixel_size)))
        pixels_to_keep = kernel_radius_px * 2
        if not (pixels_to_keep <= psf_stamp.shape[0] and pixels_to_keep <= psf_stamp.shape[1]):
            print("The kernel is too large...")
        xoff = (psf_stamp.shape[0] - pixels_to_keep) // 2
        yoff = (psf_stamp.shape[1] - pixels_to_keep) // 2
        self._kernel = psf_stamp[yoff:-yoff, xoff:-xoff]
        self._kernel = old_div(self._kernel, self._kernel.sum())

        # --- 为两种方法准备通用的参数 ---
        self._expected_shape = (flat_sky_proj.npix_height, flat_sky_proj.npix_width)
        s1 = np.array(self._expected_shape)
        s2 = np.array(self._kernel.shape)
        shape = s1 + s2 - 1
        self._fshape = [helper.next_fast_len(int(d)) for d in shape]
        
        # 【关键】保持原始的 _fslice 定义，因为它对复现至关重要
        self._fslice = tuple([slice(0, int(sz)) for sz in shape])
        
        # --- 为 PyFFTW 预计算 PSF FFT ---
        # 确保使用双精度
        kernel_float64 = self._kernel.astype(np.float64)
        if PYFFTW_AVAILABLE:
            self._psf_fft_pyfftw = pyfftw_fft.rfftn(kernel_float64, self._fshape)

    @property
    def kernel(self):
        return self._kernel

    def extended_source_image(self, ideal_image):
        """
        最终的统一入口函数。
        它只使用一个经过验证的、高性能且稳健的方法。
        """
        # 我们不再进行对比，直接使用最终的、最好的实现
        
        if PYFFTW_AVAILABLE:
            # --- 使用 PyFFTW (首选) ---
            
            # 1. 前向 FFT (使用双精度)
            fft_image = pyfftw_fft.rfftn(ideal_image.astype(np.float64), self._fshape)
            
            # 2. 逆向 FFT
            ret = pyfftw_fft.irfftn(fft_image * self._psf_fft_pyfftw, self._fshape)

            # 3. 【关键】精确复刻原始的两步裁剪逻辑
            ret_intermediate = ret[self._fslice].copy()
            conv = _centered(ret_intermediate, self._expected_shape)

            # 4. 【关键】施加物理约束以保证数值稳定性
            np.maximum(conv, 0, out=conv)
            
            return conv

        else:
            # --- 回退到原始 NumPy 实现 (如果 pyfftw 不可用) ---
            kernel_float64 = self._kernel.astype(np.float64)
            psf_fft_numpy = rfftn(kernel_float64, self._fshape)
            
            ret = irfftn(rfftn(ideal_image, self._fshape) * psf_fft_numpy, self._fshape)[self._fslice].copy()
            conv = _centered(ret, self._expected_shape)
            
            # 也对 NumPy 版本施加约束，确保行为一致
            np.maximum(conv, 0, out=conv)
            
            return conv
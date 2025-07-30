from tqdm import tqdm
import healpy as hp
import numpy as np

import matplotlib, sys
# sys.path.append(__file__[:-12])
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

import Mymap as mt

import ROOT

import copy

# import root_numpy as rn
import uproot

import matplotlib.colors as mcolors

from scipy.optimize import curve_fit

from Mycoord import *

from Myspeedup import libdir, runllhskymap, runllhskymap_mp

import MapPalette

from Myfit import find_source_id

from threeML import setup_logger

import os

log = setup_logger(__name__)
log.propagate = False

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
    on[np.isinf(on)] = hp.UNSEEN
    off[np.isinf(off)] = hp.UNSEEN
    # on = hp.ma(on)
    # off = hp.ma(off)

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
    elif signif == 0:
        sig = (on - bk) / np.sqrt(bk)
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
def getresaccuracy(
    WCDA,
    lm=None,
    signif=17,
    smooth_sigma=0.3,  #0.3 WCDA_r32 KM2A_r32
    alpha=3.24e-5,  # <--- 修正1: 默认值设为None，以触发动态计算
    combine='weighted',
    active_sources=None, #["S147", "PSR"]
    plot=False,
    savepath=None,
    savename=None,
    radius=10,
    reso=6
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
    if not isinstance(WCDA, list):
        WCDA = [WCDA]

    on_all = []
    bk_all = []
    nmbk_all = []
    for WCDA in WCDA:
        if lm:
            WCDA.set_model(lm)
        npt = int(lm.get_number_of_point_sources()) if lm else int(0)
        next = int(lm.get_number_of_extended_sources()) if lm else int(0)
        if npt == 0:
            pta = []
        else:
            pta = np.zeros(int(npt))
        if next == 0:
            exta = []
        else:
            exta = np.zeros(int(next))

        if active_sources:
            for source in active_sources:
                ids = find_source_id(lm, source)
                if ids[0] is not None:
                    pta[ids[0]] = 1
                if ids[1] is not None:
                    exta[ids[1]] = 1

        for i, bin_id in enumerate(WCDA._active_planes):
            dmap, bkmap = WCDA._get_excess(WCDA._maptree[bin_id], all_maps=True)[1:]

            model_map_val = 0.0
            keepsource = []
            keepsourcename = "all"
            if lm:
                keepsourcename = "None"
                Residual = "Residual"
                # 模型计算部分保持不变
                if active_sources:
                    model_map = WCDA._get_model_map(bin_id,  0, 0).as_dense()
                    for idx, keep in enumerate(pta):
                        if not keep:
                            model_map += WCDA._get_model_map(bin_id, idx+1, 0).as_dense()
                            if idx != 0:
                                model_map -= WCDA._get_model_map(bin_id, idx, 0).as_dense()
                    for idx, keep in enumerate(exta):
                        if not keep:
                            model_map += WCDA._get_model_map(bin_id, 0, idx+1).as_dense()
                            if idx != 0:
                                model_map -= WCDA._get_model_map(bin_id, 0, idx).as_dense()
                    model_map_val = model_map
                else:
                    model_map_val = WCDA._get_model_map(bin_id, npt, next).as_dense()
            else:
                Residual = ""
            
            on = dmap
            bk = bkmap + model_map_val
            on[np.isnan(on)]=hp.UNSEEN
            bk[np.isnan(bk)]=hp.UNSEEN
            bkmap[np.isnan(bkmap)]=hp.UNSEEN
            on=hp.ma(on)
            bk=hp.ma(bk)
            bkmap = hp.ma(bkmap)
            on_all.append(on)
            bk_all.append(bk)
            nmbk_all.append(bkmap)

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
        # --- 修正: 增强权重计算的稳定性 ---
        epsilon_weight = 1e-20 # 用于防止除以零的小数
        weights = []
        for on, bk in zip(on_all, nmbk_all):
            sum_bk = np.ma.sum(bk)
            if sum_bk > 0:
                # 只有在背景计数有效时才计算权重
                # weight = (np.ma.sum(on) - sum_bk) / (sum_bk + epsilon_weight)
                # 更稳健的权重计算，考虑信噪比
                signal = np.ma.sum(on) -  np.ma.sum(bk)
                noise = np.sqrt(np.ma.sum(on) + alpha * np.ma.sum(bk)) #alpha**2 * 
                weight = signal / (noise + epsilon_weight) if noise > 0 else 0

                # term1_ratio = (1. + alpha) / alpha * np.ma.sum(on) / (np.ma.sum(on) + sum_bk / alpha + epsilon_weight)
                # term2_ratio = (1. + alpha) * sum_bk / alpha / (sum_bk + sum_bk / alpha + epsilon_weight)
                
                # log_term1 = np.ma.sum(on) * np.log(np.maximum(term1_ratio, epsilon_weight))
                # log_term2 = sum_bk / alpha * np.log(np.maximum(term2_ratio, epsilon_weight))
                
                # sig = np.sqrt(2 * np.maximum(log_term1 + log_term2, 0)) if np.ma.sum(on) > sum_bk else 0
                # weight = sig

                weights.append(weight)
            else:
                # print(sum_bk)
                # 如果背景为0或无效，则该能段权重为0
                weights.append(0.0)
        
        weights = np.array(weights)

        # 清理可能出现的inf和nan值，并将负权重置为0（通常我们只关心信号为正的能段）
        weights[~np.isfinite(weights)] = 0.0
        weights[weights < 0] = 0.0

        sum_of_weights = np.sum(weights)
        if sum_of_weights > 0:
            weights = weights / sum_of_weights # 归一化
        else:
            # 如果所有权重都为0，发出警告，后续计算将无意义
            log.warning("All bin weights are zero in 'weighted' mode. The resulting map will be empty.")
            # 权重数组保持为全零
        
        print("Calculated robust weights:", weights)
        # --- 权重计算修正结束 ---

        if isinstance(smooth_sigma, np.ndarray):
            # 确保分母不为零
            if np.sum(weights) > 0:
                smooth_sigma = np.sqrt(np.sum(weights**2 * smooth_sigma**2) / np.sum(weights)**2)
            else:
                # 如果没有有效的权重，就取平均值作为备用
                smooth_sigma = np.sqrt(np.mean(smooth_sigma**2))
        print("Weighted smooth_sigma:", smooth_sigma)

        on_all = np.ma.array(on_all)
        bk_all = np.ma.array(bk_all)

        net_signal_stack = on_all - bk_all
        sum_weighted_net_signal = np.ma.sum(weights[:, np.newaxis] * net_signal_stack, axis=0)

        variance_term_stack = on_all + alpha * bk_all
        sum_weighted_variance = np.ma.sum((weights**2)[:, np.newaxis] * variance_term_stack, axis=0)

        epsilon = 1e-30
        # off_w = sum_weighted_variance / (2 * alpha)
        off_w = (sum_weighted_variance - sum_weighted_net_signal) / ( 1 + alpha + epsilon )
        # denominator = alpha**2 + alpha
        # off_w = (sum_weighted_variance - sum_weighted_net_signal) / (denominator + epsilon)
        # on_w = sum_weighted_net_signal + alpha * off_w
        on_w = off_w + sum_weighted_net_signal

        # 物理上，计数值不能为负。这部分是正确的。
        on_w = np.ma.maximum(on_w, 0)
        off_w = np.ma.maximum(off_w, 0)
        bk_w =  off_w #alpha *

        # 2. 将这个正确的主掩码应用到我们新计算的 on_w 和 bk_w 上。
        on_w.mask = on_all[0].mask
        bk_w.mask = on_all[0].mask
        # --- >>> 修正代码结束 <<< ---

        # --- 修正: 不要将有效的0值设置为UNSEEN ---
        # 设置掩码，而不是填充值。hp.ma会自动处理
        on_w[on_all[0].mask] = hp.UNSEEN
        bk_w[on_all[0].mask] = hp.UNSEEN
        on_w.set_fill_value(hp.UNSEEN)
        bk_w.set_fill_value(hp.UNSEEN)
        # on_w[on_w==0] = hp.UNSEEN  <-- 移除
        # bk_w[bk_w==0] = hp.UNSEEN  <-- 移除
        on_w = hp.ma(on_w)
        bk_w = hp.ma(bk_w)
        # --- 修正结束 ---

        # # === 新增：可视化权重分布 ===
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 4))
        # plt.plot(weights, 'o-', label='Bin Weights')
        # plt.xlabel('Energy Bin Index')
        # plt.ylabel('Weight')
        # plt.title('Weight Distribution Across Bins')
        # plt.savefig('weight_distribution.png')
        # plt.close()

        # # === 平滑前的ON/OFF图 ===
        # hp.gnomview(on_w, norm='', rot=[279.5, -7], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso,
        #             title='ON Map Before Smoothing')
        # # hp.gnomview(on_w, title='ON Map Before Smoothing')
        # plt.savefig('on_before_smooth.png')
        # plt.close()
        
        # hp.gnomview(bk_w, norm='', rot=[279.5, -7], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso,
        #             title='OFF Map Before Smoothing')
        # # hp.mollview(bk_w, title='OFF Map Before Smoothing')
        # plt.savefig('off_before_smooth.png')
        # plt.close()

        # === 关键步骤：执行平滑 ===
        ON_smooth, BK_smooth = _smooth_and_scale_maps(on_w, bk_w, smooth_sigma)
        
        # # === 平滑后的ON/OFF图 ===
        # hp.gnomview(ON_smooth, norm='', rot=[279.5, -7], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso,
        #             title='ON Map After Smoothing')
        # # hp.mollview(ON_smooth, title='ON Map After Smoothing')
        # plt.savefig('on_after_smooth.png')
        # plt.close()
        
        # hp.gnomview(BK_smooth, norm='', rot=[279.5, -7], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso,
        #             title='OFF Map After Smoothing')
        # # hp.mollview(BK_smooth, title='OFF Map After Smoothing')
        # plt.savefig('off_after_smooth.png')
        # plt.close()

        # ON_smooth, BK_smooth = _smooth_and_scale_maps(on_w, bk_w, smooth_sigma)
        resu = compute_significance_mapfast(ON_smooth, BK_smooth, alpha=alpha, signif=signif) #signif=signif,signif=0
        resu_all = resu
    elif combine == 'none':
        ii=0
        for on, bk in zip(on_all, bk_all):
            sigma = smooth_sigma[ii] if isinstance(smooth_sigma, np.ndarray) else smooth_sigma
            ON_smooth, BK_smooth = _smooth_and_scale_maps(on, bk, sigma)
            sig = compute_significance_mapfast(ON_smooth, BK_smooth, signif=signif, alpha=final_alpha)
            resu_all.append(sig)
            ii+=1

    if savename is None:
        keepsourcename = active_sources[0]+"+".join(active_sources[1:]) if active_sources is not None else keepsourcename
        savename = f"sigmapfast_{combine}_{signif}_{keepsourcename}"
    # (文件保存和绘图逻辑保持不变)
    # --- 文件保存和绘图逻辑保持不变 ---
    if savepath:
        if savepath[-1] != '/':
            savepath += '/'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        # healpy.write_map可以接受list of maps
        hp.write_map(savepath+f"{savename}.fits", resu_all, overwrite=True)

    if plot:
        resu_to_plot = resu_all
        if isinstance(resu_all, list):
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
                # hp.mollview(m, title=f"bin {i}", sub=(nrows, ncols, i + 1))
                new_source_idx = np.where(m == np.ma.max(m))[0][0]
                new_source_lon_lat = hp.pix2ang(1024, new_source_idx, lonlat=True)
                # 强制隐藏所有坐标轴元素
                ax = plt.gca()
                ax.set_xticks([])  # 隐藏x轴刻度
                ax.set_yticks([])  # 隐藏y轴刻度
                ax.set_xlabel('')  # 隐藏x轴标签
                ax.set_ylabel('')  # 隐藏y轴标签
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_axis_off() 
                hp.gnomview(m, rot=[new_source_lon_lat[0], new_source_lon_lat[1]], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso, title=f"bin {i}", return_projected_map=True, sub=(nrows, ncols, i + 1), cbar=True, reuse_axes=False) # 您可以控制是否显示 colorbar , reuse_axes=True
                # ax.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
                hp.graticule(dpar=1, dmer=1, verbose=False, color='gray', alpha=0.5)
                
                if new_source_lon_lat and len(new_source_lon_lat) == 2:
                    hp.projplot(new_source_lon_lat[0], new_source_lon_lat[1], 'rx', lonlat=True, markersize=10)
                else:
                    log.warning("new_source_lon_lat is not properly defined or has invalid values.")
                # plt.legend()
            plt.tight_layout()
            plt.axis('off') 
            plt.show()
            if savepath and savename:
                plt.savefig(f"{savepath}{savename}.png",dpi=300)
        else:
            # 适用于 combine='sum' 或 'weighted'
            resu = resu_all
            new_source_idx = np.where(resu == np.ma.max(resu))[0][0]
            new_source_lon_lat = hp.pix2ang(1024, new_source_idx, lonlat=True)
            resu.set_fill_value(hp.UNSEEN)
            resu = hp.ma(resu)
            plt.figure(figsize=(8, 8))
            hp.gnomview(resu, norm='', rot=[new_source_lon_lat[0], new_source_lon_lat[1]], xsize=radius*2*(60/reso), ysize=radius*2*(60/reso), reso=reso,
                        title=(f"{Residual} Significance Map {savename.replace('sigmapfast_', '').replace('_', ' ')}"))
            hp.graticule(dpar=1, dmer=1, verbose=False)
            if new_source_lon_lat and len(new_source_lon_lat) == 2:
                hp.projplot(new_source_lon_lat[0], new_source_lon_lat[1], 'rx', lonlat=True, markersize=10)
                # plt.scatter(x, y, marker='x', color='red', s=100, label=f'Hottest Spot {new_source_lon_lat}')
            else:
                log.warning("new_source_lon_lat is not properly defined or has invalid values.")
            # plt.legend()
            # 标记最亮点的代码可以加在这里
            # hp.projtext(new_source_lon_lat[0], new_source_lon_lat[1], 'X', lonlat=True, color='red')
            print(f"Hottest spot at (lo1n, lat): {new_source_lon_lat}")
            plt.show()
            if savepath and savename:
                plt.savefig(f"{savepath}{savename}.png",dpi=300)

    return resu_all

def getmap(WCDA, roi, name="J0248", signif=17, smoothsigma = [0.42, 0.32, 0.25, 0.22, 0.18, 0.15], 
           save = False, 
           binc="all",
           stack=[],
           modelindex=None,
           pta=[], exta=[],
           smooth=False,
            stack_sigma=None
           ):  # sourcery skip: default-mutable-arg, low-code-quality# sourcery skip: default-mutable-arg
    """Get counts map.

        Args:
            pta=[1,0,1], exta=[0,0]: if you have 3 pt sources and 2 ext sources, and you only want to keep 1st and 3st sources,you do like this.
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    
    #Initialize
    amap = []
    nside=2**10
    npix=hp.nside2npix(nside)
    pixarea = 4 * np.pi/npix
    pixIdx = np.arange(npix)
    pixid=roi.active_pixels(1024)

    signal=np.full(npix, hp.UNSEEN, dtype=np.float64)
    background=np.full(npix, hp.UNSEEN, dtype=np.float64)
    modelmap=np.full(npix, hp.UNSEEN, dtype=np.float64)
    modelbkg=np.full(npix, hp.UNSEEN, dtype=np.float64)
    alpha=np.full(npix, hp.UNSEEN, dtype=np.float64)

    new_lats = hp.pix2ang(nside, pixIdx)[0]
    new_lons = hp.pix2ang(nside, pixIdx)[1]
    mask = ((-new_lats + np.pi/2 < -20./180*np.pi) | (-new_lats + np.pi/2 > 80./180*np.pi))

    if binc=="all":
        binc=WCDA._active_planes

    for bin in binc:
        smooth_sigma=smoothsigma[int(bin)]
        active_bin = WCDA._maptree._analysis_bins[bin]
        if modelindex:
            model=WCDA._get_expectation(active_bin,bin,modelindex[0],modelindex[1])
        else:
            model=WCDA._get_expectation(active_bin,bin,0,0)
            for i,pt in enumerate(pta):
                if not pt:
                    model += WCDA._get_expectation(active_bin,bin,i+1,0)
                    if i != 0:
                        model -= WCDA._get_expectation(active_bin,bin,i,0)
            
            for i,ext in enumerate(exta):
                if not ext:
                    model += WCDA._get_expectation(active_bin,bin,0,i+1)
                    if i != 0:
                        model -= WCDA._get_expectation(active_bin,bin,0,i)

        obs_raw=active_bin.observation_map.as_partial()
        bkg_raw=active_bin.background_map.as_partial()
        res_raw=obs_raw-model
        for i,pix in enumerate(tqdm(pixid)):
            signal[pix]=obs_raw[i]
            background[pix]=bkg_raw[i]
            modelmap[pix]=model[i]
            modelbkg[pix]=model[i]+bkg_raw[i]
            theta, phi = hp.pix2ang(nside, pix)
            theta = np.pi/2 - theta
            alpha[pix]=2*smooth_sigma*1.51/60./np.sin(theta) #

        log.info("Mask all")
        signal=hp.ma(signal)
        background=hp.ma(background)
        modelmap=hp.ma(modelmap)
        modelbkg=hp.ma(modelbkg)
        alpha=hp.ma(alpha)

        if smooth:
            log.info("Smooth Sig")
            signal_smoothed=hp.sphtfunc.smoothing(signal,sigma=np.radians(smooth_sigma))
            signal_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(signal,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

            log.info("Smooth bkg")
            background_smoothed=hp.sphtfunc.smoothing(background,sigma=np.radians(smooth_sigma))
            background_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(background,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

            log.info("Smooth Modelbkg")
            modelbkg_smoothed=hp.sphtfunc.smoothing(modelbkg,sigma=np.radians(smooth_sigma))
            modelbkg_smoothed2=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(modelbkg,sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea
        else:
            signal_smoothed=np.array([])
            signal_smoothed2=np.array([])
            background_smoothed=np.array([])
            background_smoothed2=np.array([])
            modelbkg_smoothed=np.array([])
            modelbkg_smoothed2=np.array([])

        if save:
            log.info("Save!")
            hp.mollview(signal_smoothed,title="Mollview image RING",norm='hist',unit='Excess')
            hp.graticule()
            plt.savefig(f"{libdir}/../res/%s_excess_nHit0%s_%.2f.pdf"%(name, bin, smooth_sigma))
            hp.write_map(f"{libdir}/../data/%s_nHit0%s_%.2f.fits.gz"%(name, bin, smooth_sigma),[signal, background, signal_smoothed, background_smoothed, signal_smoothed2, background_smoothed2, modelmap, alpha],overwrite=True)

        amap.append([signal, background, modelbkg, 
                     signal_smoothed, background_smoothed, modelbkg_smoothed,
                     signal_smoothed2, background_smoothed2, modelbkg_smoothed2,
                     modelmap, alpha])
    if stack != []:
        summap = copy.deepcopy(amap)
        for i, bin in enumerate(binc):
        # for i, weight in enumerate(stack):
            weight=stack[int(bin)]
            for j in range(10):
                summap[i][j] *= weight
                if j in [6, 7, 8]:
                    summap[i][j] *= weight
        outmap = [np.ma.sum([bin[i] for bin in summap],axis=0) for i in tqdm(range(11))]
        # outmap[-1] = np.ma.sqrt(np.ma.sum([bin[-1]**2*stack[i] for i,bin in enumerate(amap) if i<6],axis=0))
        if stack_sigma:
            smooth_sigma=stack_sigma
        else:
            log.info("Set stack_sigma automatelly!!!")
            stack_sigma=smoothsigma[len(WCDA._maptree._analysis_bins)] #int(list(WCDA._maptree._analysis_bins.keys())[-1])+1
        for i,pix in enumerate(tqdm(pixid)):
            theta, phi = hp.pix2ang(nside, pix)
            theta = np.pi/2 - theta
            alpha[pix]=2*stack_sigma*1.51/60./np.sin(theta)
        alpha=hp.ma(alpha)
        outmap[-1]=alpha

        for mapp in outmap:
            mapp.fill_value=hp.UNSEEN
        amap.append(outmap)
    return amap

def stack_map(map, stack=None):
    """stack map together.

        Args:
            stack: weight, usually signal to noise ratio.
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    if stack is None:
        return map
    summap = copy.deepcopy(map)
    for i, weight in enumerate(stack):
        for j in range(11):
            summap[i][j] *= weight
            if j in [6, 7, 8]:
                summap[i][j] *= weight
    outmap = [np.ma.sum([bin[i] for bin in summap],axis=0) for i in tqdm(range(10))]
    outmap[-1] = np.ma.sqrt(np.ma.sum([bin[-1]**2*stack[i] for i,bin in enumerate(map) if i<6],axis=0))
    for mapp in outmap:
        mapp.fill_value=hp.UNSEEN
    return outmap

def smoothmap(mapall, smooth_sigma = 0.2896):
    """Get smooth map.

        Args:
            smooth_sigma:
        Returns:
            ----------
            >>> [[signal, background, modelbkg, \\
            signal_smoothed, background_smoothed, modelbkg_smoothed, \\
            signal_smoothed2, background_smoothed2, modelbkg_smoothed2, \\
            modelmap, alpha]....] \\
    """
    nside=2**10
    npix=hp.nside2npix(nside)
    pixarea = 4 * np.pi/npix
    # for i in tqdm([0,1,2]):
    #     mapall[i] = hp.ma(mapall[i])

    log.info("Smooth Sig")
    mapall[3]=hp.sphtfunc.smoothing(mapall[0],sigma=np.radians(smooth_sigma))
    mapall[6]=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(mapall[0],sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    log.info("Smooth bkg")
    mapall[4]=hp.sphtfunc.smoothing(mapall[1],sigma=np.radians(smooth_sigma))
    mapall[7]=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(mapall[1],sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    log.info("Smooth Modelbkg")
    mapall[5]=hp.sphtfunc.smoothing(mapall[2],sigma=np.radians(smooth_sigma))
    mapall[8]=1./(4.*np.pi*np.radians(smooth_sigma)*np.radians(smooth_sigma))*(hp.sphtfunc.smoothing(mapall[2],sigma=np.radians(smooth_sigma/np.sqrt(2))))*pixarea

    log.info("Mask all")
    for i in tqdm(range(3,8)):
        mapall[i] = hp.ma(mapall[i])

    return mapall

import math
def Draw_ellipse(e_x, e_y, a, e, e_angle, color, linestyle, alpha=0.5, coord="C", ax=None, label=None, lw=None):
    """
        画椭圆

        Parameters:

        Returns:
            >>> None
    """ 
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    b=a*np.sqrt(1-e**2)
    for angles in angles_circle:
        or_x = a * np.cos(angles)
        or_y = b * np.sin(angles)
        length_or = np.sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + e_angle/180*np.pi
        new_x = e_x + length_or * np.cos(new_theta) #
        new_y = e_y + length_or * np.sin(new_theta)
        dnew_x = new_x-e_x
        new_x = e_x+dnew_x/np.cos(np.radians(new_y))
        x.append(new_x)
        y.append(new_y)
    if coord=="G":
        x,y = edm2gal(x,y)
    if ax is None:
        ax=plt.gca()
    ax.plot(x,y, color=color, linestyle=linestyle,alpha=alpha, label=label, linewidth=lw)


def high_pass_filter(image, cutoff_freq):
    """
        图像高通滤波

        Parameters:

        Returns:
            >>> None
    """ 
    # 进行二维傅里叶变换
    f_transform = np.fft.fft2(image)
    
    # 将零频率分量移到中心
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 获取图像大小
    rows, cols = image.shape
    
    # 创建一个高通滤波器
    high_pass_filter = np.ones((rows, cols))
    center_row, center_col = rows // 2, cols // 2
    high_pass_filter[center_row - cutoff_freq:center_row + cutoff_freq, 
                     center_col - cutoff_freq:center_col + cutoff_freq] = 0
    
    # 进行傅里叶逆变换
    filtered_transform_shifted = f_transform_shifted * high_pass_filter
    filtered_transform = np.fft.ifftshift(filtered_transform_shifted)
    filtered_image = np.abs(np.fft.ifft2(filtered_transform))
    
    return filtered_image


def drawfits(fits_file_path = '/data/home/cwy/Science/3MLWCDA/Standard/res/S147/S147_mosaic.fits', fig=None, vmin=None, vmax=None, drawalpha=False, iffilter=False, cmap=plt.cm.Greens, cutl=0.2, cutu=1, filter=1, alpha=1):
    """
        画fits文件

        Parameters:
            fig: 画在什么上?
            drawalpha: 是否画一定透明度的?
            iffilter: 是否对一定阈值外的部分进行完全透明?
            cutl: 阈值下限 0-1
            cutu: 阈值上限 0-1
            alpha: 透明度

        Returns:
            >>> fig, wcs, data
    """ 
    from astropy.io import fits
    from astropy.wcs import WCS
# fits_file_path = '/data/home/cwy/Science/3MLWCDA/Standard/res/S147/S147_mosaic.fits'; vmin=-15; vmax=30
    # 打开 FITS 文件
    hdul = fits.open(fits_file_path)
    log.info(str(hdul.info()))

    # 获取数据和坐标信息
    data = hdul[0].data
    wcs = WCS(hdul[0].header)

    shape = wcs.array_shape
    a = wcs.pixel_to_world(0, 0)
    b = wcs.pixel_to_world(shape[1]-1, shape[0]-1)
    log.info(f"{str(wcs)} \n {shape} \n {a} \n {b}")

    # 关闭 FITS 文件
    hdul.close()

    # 检测坐标系类型
    if "RA" in wcs.wcs.ctype[0] and "DEC" in wcs.wcs.ctype[1]:
        # 如果包含 "RA" 和 "DEC"，则是赤道坐标
        xlabel = 'RA (J2000)'
        ylabel = 'Dec (J2000)'
    elif "GLON" in wcs.wcs.ctype[0] and "GLAT" in wcs.wcs.ctype[1]:
        # 如果包含 "GLON" 和 "GLAT"，则是银道坐标
        xlabel = 'Galactic Longitude'
        ylabel = 'Galactic Latitude'
    else:
        # 如果无法判断，默认使用 "X-axis" 和 "Y-axis"
        xlabel = 'X-axis'
        ylabel = 'Y-axis'

    data[np.isnan(data)]=0
    # 绘制图像

    if (not vmin) or (not vmax):
        vmin = data.min()
        vmax = data.max()
    if fig:
        ax = fig.gca()
        if not drawalpha:
            from matplotlib.colors import Normalize

            alphas = Normalize(vmin, vmax, clip=True)(data)
            alphas = np.clip(alphas, cutl, cutu)
            alphas[alphas<=cutl]=0
            if iffilter:
                alphas = high_pass_filter(alphas, filter)
                alphas = Normalize(vmin, vmax, clip=True)(alphas)
                alphas = np.clip(alphas, cutl, cutu)
            alphas[alphas<=cutl]=0

            colors = Normalize(vmin, vmax)(data)
            colors = cmap(colors)

            colors[..., -1] = alphas

            im = ax.imshow(colors, origin='lower', extent=[a.ra.value, b.ra.value, a.dec.value, b.dec.value], vmin=0, vmax=(cutl+cutu)/2, interpolation='bicubic', alpha=alpha)
            return fig
        else:
            im = ax.imshow(data, cmap=cmap, origin='lower', extent=[a.ra.value, b.ra.value, a.dec.value, b.dec.value], vmin=vmin, vmax=vmax, interpolation='bicubic', alpha=alpha)
            return fig
    else:
        fig, ax = plt.subplots(subplot_kw={'projection': wcs})
        im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='bicubic')
        # 添加坐标轴标签
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # 添加坐标网格
        ax.coords.grid()

        # 显示坐标轴的度标尺
        ax.coords[0].set_format_unit('degree')
        ax.coords[1].set_format_unit('degree')

        # 添加色标
        cbar = plt.colorbar(im, ax=ax, label='Intensity')
        plt.show()
        return fig, wcs, data

def printfits(file_path):
    """
    打开一个FITS文件并输出其中所有的信息
    
    参数:
        file_path (str): FITS文件的路径
        
    返回:
        None (直接打印输出文件内容)
    """

    try:
        # 打开FITS文件
        with fits.open(file_path) as hdul:
            print(f"成功打开FITS文件: {file_path}")
            print(f"文件包含 {len(hdul)} 个HDU (Header Data Unit)")
            print("="*80)
            
            # 遍历所有HDU
            for i, hdu in enumerate(hdul):
                print(f"\nHDU {i}:")
                print(f"类型: {hdu.__class__.__name__}")
                
                # 打印头部信息
                print("\n头部信息:")
                print(repr(hdu.header))
                
                # 打印数据信息
                print("\n数据信息:")
                if hdu.data is not None:
                    if isinstance(hdu.data, np.ndarray):
                        print(f"数据类型: {hdu.data.dtype}")
                        print(f"数据形状: {hdu.data.shape}")
                        
                        # 对于小数组，打印全部内容；对于大数组，只打印摘要
                        if hdu.data.size < 100:
                            print("数据内容:")
                            print(hdu.data)
                        else:
                            print("数据摘要:")
                            print(f"最小值: {np.nanmin(hdu.data)}")
                            print(f"最大值: {np.nanmax(hdu.data)}")
                            print(f"平均值: {np.nanmean(hdu.data)}")
                            print(f"标准差: {np.nanstd(hdu.data)}")
                    else:
                        print(hdu.data)
                else:
                    print("此HDU不包含数据")
                
                print("="*80)
                
    except Exception as e:
        print(f"处理文件时出错: {file_path}")
        print(f"错误信息: {str(e)}")
    
def heal2fits(map, name, ra_min = 82, ra_max = 88, xsize=0.1, dec_min=26, dec_max=30, ysize=0.1, nside=1024, ifplot=False, ifnorm=False, check=False, alpha=1, projection="CAR", coord="C", saveCAR=0, flip=0, ifcounts=False):
    """
        将healpix天图转fits天图

        Parameters:
            name: 保存fits文件路径
        Returns:
            >>> None
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    # 将RA和DEC范围转换为SkyCoord对象
    ra=np.arange(ra_min, ra_max, xsize)+xsize/2; dec=np.arange(dec_min, dec_max, ysize)+ysize/2
    log.info(f"{len(ra)} {len(dec)}")
    X,Y = np.meshgrid(ra, dec)
    coords = SkyCoord(ra=X, dec=Y, unit="deg", frame="icrs")
    # 使用SkyCoord对象获取对应的HEALPix像素索引
    npix=hp.nside2npix(nside)
    pixarea = 4*np.pi/npix
    pix_indices = hp.ang2pix(nside, coords.ra.degree, coords.dec.degree, lonlat=True)
    map[map==hp.UNSEEN]=0
    map[map<=-1000000]=0

    # 创建一个新的FITS文件，其中包含指定方形区域的数据
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = int(len(ra))
    header["NAXIS2"] = int(len(dec))
    if coord=="C":
        header["CTYPE1"] = f"RA---CAR"
        header["CTYPE2"] = f"DEC--CAR"
    else: 
        header["CTYPE1"] = f"GLON-CAR"
        header["CTYPE2"] = f"GLAT-CAR"
    # header["CRVAL1"] = ra.mean()
    # header["CRVAL2"] = dec.mean()
    # header["CRPIX1"] = header["NAXIS1"] / 2
    # header["CRPIX2"] = header["NAXIS2"] / 2
    header["CRVAL1"] = ra.mean()
    header["CRVAL2"] = 0
    header["CRPIX1"] = header["NAXIS1"] / 2 #(180-ra.mean())/xsize + 
    header["CRPIX2"] = (0-dec.mean())/ysize + header["NAXIS2"] / 2 # - xsize/2
    header["CD1_1"] = xsize #*np.cos(np.radians(header["CRVAL2"]))
    header["CD2_2"] = ysize
    if flip:
        header['FLIP_LR'] = True

    wcs = WCS(header)
    log.info(f"{str(wcs)}")

    # 创建一个空的二维数组，用于存储提取的数据
    extracted_data = np.zeros((header["NAXIS2"], header["NAXIS1"]))
    
    # 将HEALPix数据的指定区域复制到新数组中
    extracted_data = map[pix_indices]
    extracted_data[np.isnan(extracted_data)]=0

    if ifcounts:
        area = np.radians(xsize)*np.radians(ysize)*np.ones((len(dec), len(ra)))*np.cos(np.radians(Y))
        extracted_data = extracted_data*area/pixarea

    if ifplot:
        plt.imshow(extracted_data.data, extent=[ra_min, ra_max, dec_min, dec_max], origin="lower", aspect='auto', vmin=np.min(extracted_data.data), vmax=np.max(extracted_data.data))
        plt.gca().invert_xaxis()
        plt.colorbar()

    if ifnorm:
        extracted_data = extracted_data-extracted_data.min()
        extracted_data = extracted_data**alpha
        area = np.radians(xsize)*np.radians(ysize)*np.ones((len(dec), len(ra)))*np.cos(np.radians(Y))
        integral = extracted_data*area #/pixarea*area
        extracted_data = extracted_data/integral.sum()

    if check:
        plt.figure()
        plt.imshow(area, extent=[ra_min, ra_max, dec_min, dec_max], origin="lower")
        plt.gca().invert_xaxis()

    if saveCAR:
        # 将提取的数据保存到FITS文件
        fits.writeto(name+"_CAR.fits", np.array(extracted_data), header, overwrite=True)

    if projection == "CAR":
        fits.writeto(name, np.array(extracted_data), header, overwrite=True)

    if projection is not "CAR":
        wcs2 = WCS(naxis=2)
        log.info(f"Trans CAR to {projection}: \n {str(wcs)}")
        if coord=="C":
            wcs2.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']
        else:
            wcs2.wcs.ctype = [f'GLON-{projection}', f'GLAT-{projection}']
        wcs2.wcs.crval = [ra.mean(), 0]
        wcs2.wcs.crpix = [header["NAXIS1"] / 2, (0-dec.mean())/xsize + header["NAXIS2"] / 2]
        if flip:
            # extracted_data = np.fliplr(extracted_data)
            wcs2.wcs.cdelt = np.array([-xsize, ysize])
        else:
            wcs2.wcs.cdelt = np.array([xsize, ysize])
        wcs2.wcs.cunit = ['deg', 'deg']
        

        from reproject import reproject_interp
        hdu = fits.PrimaryHDU(extracted_data, header=header)
        array, footprint = reproject_interp(hdu, wcs2, shape_out=[len(dec), len(ra)], parallel=10)
        fits.writeto(name, np.array(array), wcs2.to_header(), overwrite=True)

def drawmap(region_name, Modelname, sources, map, ra1, dec1, rad=6, contours=[3, 5], save=False, savename=None, zmin=None, zmax=None, cat={ "LHAASO": [0, "P"],"TeVCat": [0, "s"], "PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"], "4FGL": [0, "d"], "YMC": [0, "^"], "GYMC":[0, "v"], "WR":[0, "X"], "size": 20, "markercolor": "grey", "labelcolor": "black", "angle": 60, "catext": 1}, color="Fermi", colorlabel="", legend=True, Drawdiff=False, ifdrawfits=False, fitsfile=None, vmin=None, vmax=None, drawalpha=False, iffilter=False, cmap=plt.cm.Greens, cutl=0.2, cutu=1, filter=1, alphaf=1,     
    colors=None, grid=False, dpi=300, drawLHAASO=False, detector = "WCDA"
        ):  # sourcery skip: extract-duplicate-method
    """Draw a healpix map with fitting results.

        Args:
            sources: use function get_sources() to get the fitting results.
            cat: catalog to draw. such as {"TeVCat":[1,"s"],"PSR":[0,"*"],"SNR":[1,"o"]}, first in [1,"s"] is about if add a label?
                "o" is the marker you choose.
                The catalog you can choose is:
                     TeVCat/3FHL/4FGL/PSR/SNR/AGN/QSO/Simbad
        Returns:
            ----------
            >>> fig
    """
    import os
    if not os.path.exists(f'{libdir}/../res/{region_name}/'):
        os.system(f'mkdir {libdir}/../res/{region_name}/')
    if not os.path.exists(f'{libdir}/../res/{region_name}/{Modelname}/'):
        os.system(f'mkdir {libdir}/../res/{region_name}/{Modelname}/')
    from matplotlib.patches import Ellipse
    fig = mt.hpDraw(region_name, Modelname, map,ra1,dec1,
            radx=rad/np.cos(dec1/180*np.pi),rady=rad, zmin=zmin, zmax=zmax,
            colorlabel=colorlabel, contours=contours, save=False, cat=cat, color=color, Drawdiff=Drawdiff, grid=grid, dpi=dpi
            )
    ax = plt.gca()
    if colors is None:
        import MapPalette
        colors = MapPalette.colorall 
    # colors=list(mcolors.TABLEAU_COLORS.keys()) #CSS4_COLORS
    # colors=['tab:red',
    #         'tab:blue',
    #         'tab:green',
    #         'tab:purple',
    #         'tab:orange',
    #         'tab:brown',
    #         'tab:pink',
    #         'tab:olive',
    #         'tab:cyan',
    #         'tab:gray']

    if drawLHAASO:
        from Myfit import getcatModel, get_sources
        lm = getcatModel(ra1, dec1, rad/2, rad/2, ifext_mt_2=True, detector=detector)
        sources = get_sources(lm)

    i=0
    ifasymm=False
    for sc in sources.keys():
        source = sources[sc]
        for par in source.keys():
            if par in ['lon0', 'ra']:
                x = source[par][2]
                xeu = source[par][3]
                xel = source[par][4]     
            elif par in ["lat0","dec"]:
                y = source[par][2]
                yeu = source[par][3]
                yel = source[par][4]
            elif par in ["sigma","rdiff0","radius", "a"]:
                sigma = 2*source[par][2]
                sigmau = 2*source[par][3]
                sigmal = 2*source[par][4]
            elif par in ["e", "elongation"]:
                ifasymm=True
                e = source[par][2]
                eu = source[par][3]
                el = source[par][4]
            elif par in ["theta", "incl"]:
                ifasymm=True
                theta = source[par][2]
                thetau = source[par][3]
                thetal = source[par][4]


        if sources[sc]['type'] == 'extended source' and not ifasymm:
            plt.errorbar(x, y, yerr=(np.abs([yel]), np.abs([yeu])), xerr=(np.abs([xel]), np.abs([xeu])), fmt='o',markersize=2,capsize=1,elinewidth=1,color=colors[i], label=sc)
            error_ellipse = Ellipse((x, y), width=sigma/np.cos(np.radians(y)), height=sigma, edgecolor=colors[i], fill=False,linestyle="-")
            ax.add_artist(error_ellipse)
            error_ellipse = Ellipse((x, y), width=(sigma+sigmau)/np.cos(np.radians(y)), height=sigma+sigmau, edgecolor=colors[i], fill=False,linestyle="--", alpha=0.5)
            ax.add_artist(error_ellipse)
            error_ellipse = Ellipse((x, y), width=(sigma-abs(+sigmal))/np.cos(np.radians(y)), height=sigma-abs(sigmal), edgecolor=colors[i], fill=False,linestyle="--", alpha=0.5)
            ax.add_artist(error_ellipse)
        elif ifasymm:
            plt.errorbar(x, y, yerr=(np.abs([yel]), np.abs([yeu])), xerr=(np.abs([xel]), np.abs([xeu])), fmt='o',markersize=2,capsize=1,elinewidth=1,color=colors[i], label=sc)
            # log.info(x,y,sigma,e,theta)
            Draw_ellipse(x,y,sigma,e,theta,colors[i],"-")
            ifasymm = False
        else:
            plt.errorbar(x, y, yerr=(np.abs([yel]), np.abs([yeu])), xerr=(np.abs([xel]), np.abs([xeu])), fmt='o',markersize=2,capsize=1,elinewidth=1,color=colors[i],label=sc)
        i+=1
        # if i==1:
        #     i+=1

    if ifdrawfits:
        if fitsfile:
            drawfits(fits_file_path=fitsfile, fig=fig, vmin=vmin, vmax=vmax, drawalpha=drawalpha, iffilter=iffilter, cmap=cmap, cutl=cutl, cutu=cutu, filter=filter, alpha=alphaf)
        else:
            drawfits(fig=fig, vmin=vmin, vmax=vmax, drawalpha=drawalpha, iffilter=iffilter, cmap=cmap, cutl=cutl, cutu=cutu, filter=filter, alpha=alphaf)

    if legend:
        plt.legend()
    if save or savename:
        if savename==None:
            plt.savefig(f"{libdir}/../res/{region_name}/{Modelname}/???_sig_llh_model.png",dpi=dpi)
            plt.savefig(f"{libdir}/../res/{region_name}/{Modelname}/???_sig_llh_model.pdf")
        else:
            plt.savefig(f"{libdir}/../res/{region_name}/{Modelname}/{savename}.png",dpi=dpi)
            plt.savefig(f"{libdir}/../res/{region_name}/{Modelname}/{savename}.pdf")

    return fig

def gaussian(x,a,mu,sigma):
    return a*np.exp(-((x-mu)/sigma)**2/2)

def getsig1D(S, region_name=None, Modelname=None, name=None, showexp=True, logy=True, ylimsclae=2, xlimscale=10, bins=None):
    """
        从healpix显著性天图S画一维显著性分布并保存

        Parameters:

        Returns:
            >>> None
    """ 
    S=S[S!=0]
    if bins is None:
        bins = 4*int(max(S.compressed())+5)
        if bins<100:
            bins=100
    bin_y,bin_x,patches=plt.hist(S.compressed(),bins=bins)
    plt.close()
    bin_x=np.array(bin_x)
    bin_y=np.array(bin_y)
    
    fit_range = np.logical_and(bin_x>np.max([-5, min(S.compressed())]), bin_x<np.min([5,max(S.compressed())]))
    wdt=(bin_x[1]-bin_x[0])/2.
    try:
        popt, pcov = curve_fit(
            gaussian,
            bin_x[fit_range] + wdt,
            bin_y[fit_range[:-1]],
            bounds=([100, -1, 0], [50000000, 2, 10]),
        )
    except (ValueError, IndexError) as e:
        print(f"Error in curve fitting: {e}. Using first 100 bins for fitting.")
        popt, pcov = curve_fit(
            gaussian,
            bin_x[:100] + wdt,
            bin_y[:100],
            bounds=([100, -2, 0], [50000000, 2, 10]),
        )
    #popt,pcov = curve_fit(gaussian,bin_x[fit_range[0:-1]]+(bin_x[1]-bin_x[0])/2.,bin_y[fit_range[0:-1]],bounds=([100,-2,0],[50000000,2,10]))
    log.info("************************")
    log.info(popt)
    log.info("************************")
    log.info("max Significance= %.1f"%(max(S.compressed())))

    plt.figure()
    #plt.plot([0.,0.],[1,1e6],'k--',linewidth=0.5)
    if showexp:
        plt.plot(
            (bin_x[:bins] + bin_x[1:bins+1]) / 2,
            gaussian((bin_x[:bins] + bin_x[1:bins+1]) / 2, popt[0], 0, 1),
            '--',
            label='expectation',
        )
    plt.plot((bin_x[:bins] + bin_x[1:bins+1]) / 2, bin_y, label="data")
    plt.plot(
        (bin_x[:bins] + bin_x[1:bins+1]) / 2,
        gaussian((bin_x[:bins] + bin_x[1:bins+1]) / 2, popt[0], popt[1], popt[2]),
        '--',
        label='fit',
    )
    if logy:
        plt.yscale('log')
    plt.xlim(-xlimscale,xlimscale)
    plt.ylim(1,max(bin_y*ylimsclae))
    plt.grid(True)
    plt.text(-xlimscale+0.5,max(bin_y),'mean = %f\n width = %f'%(popt[1],popt[2]))
    plt.xlabel(r'Significance($\sigma$)')
    plt.ylabel("entries")
    plt.legend()
    if region_name is not None and Modelname is not None and name is not None:
        plt.savefig(f"{libdir}/../res/{region_name}/{Modelname}/hist_sig_{name}.pdf")
        plt.savefig(f"{libdir}/../res/{region_name}/{Modelname}/hist_sig_{name}.png",dpi=300)

def getsigmap(region_name, Modelname, mymap,i=0,signif=17,res=False,name="J1908", alpha=None):
    """put in a smooth map and get a sig map.

        Args:
        Returns:
            sigmap: healpix
    """
    if len(mymap) == 1:
        i=0
        imap=mymap[0]
    else:
        imap=mymap[i]

    if res:
        scale=(imap[3]+imap[5])/(imap[6]+imap[8])
        ON=imap[3]*scale
        BK=imap[5]*scale
        name+="_res"
    else:
        scale=(imap[3]+imap[4])/(imap[6]+imap[7])
        ON=imap[3]*scale
        BK=imap[4]*scale

    if alpha is not None:
        alpha = alpha
    else:
        alpha = imap[10]

    if signif==5:
        S=(ON-BK)/np.sqrt(ON+alpha*BK)
    elif signif==9:
        S=(ON-BK)/np.sqrt(ON*alpha+BK)
    elif signif==17:
        S=np.sqrt(2.)*np.sqrt(ON*np.log((1.+alpha)/alpha*ON/(ON+BK/alpha))+BK/alpha*np.log((1.+alpha)*BK/alpha/(ON+BK/alpha)))
        S[ON<BK] *= -1
    else:
        S=(ON-BK)/np.sqrt(BK)
    getsig1D(S, region_name, Modelname, name)
    return S

def write_resmap(region_name, Modelname, WCDA, roi, maptree, response, ra1, dec1, data_radius, outname,
                 active_sources=[], binc="all", ifrunllh=True, detector="WCDA", jc=10, sn=1000, s=None, e=None, llhmode=None):
    """Write residual map to skymap root file."""
    log.info(outname + "_res")

    # Healpix region info
    colat = np.radians(90 - dec1)
    lon = np.radians(ra1)
    vec = hp.ang2vec(colat, lon)
    holepixid = hp.query_disc(1024, vec, np.radians(data_radius))
    pixid = roi.active_pixels(1024)
    npix = hp.nside2npix(1024)

    # Ensure binc is a list of integers
    if binc == "all":
        binc = [x for x in WCDA._active_planes]  # Convert to integers

    # Output file path
    outroot = f"{libdir}/../res/{region_name}/{Modelname}/{outname}.root"

    # Read BinInfo with PyROOT (uproot doesn't support cloning complex objects)
    forg_pyroot = ROOT.TFile.Open(maptree, 'read')
    bininfo = forg_pyroot.Get("BinInfo")

    # Write BinInfo to output file with PyROOT
    fout_pyroot = ROOT.TFile.Open(outroot, 'recreate')
    bininfoout = bininfo.CloneTree(0)
    for entry in bininfo:
        if str(entry.name) in binc:
            bininfoout.Fill()
    bininfoout.Write()
    fout_pyroot.Close()  # Close PyROOT file

    # Active source structure
    npt = WCDA._likelihood_model.get_number_of_point_sources() if WCDA._likelihood_model else 0
    next = WCDA._likelihood_model.get_number_of_extended_sources() if WCDA._likelihood_model else 0
    pta, exta = np.zeros(npt), np.zeros(next)

    for source in active_sources:
        ids = find_source_id(WCDA._likelihood_model, source)
        if ids[0] is not None:
            pta[ids[0]] = 1
        if ids[1] is not None:
            exta[ids[1]] = 1

    # Open input file with uproot
    with uproot.open(maptree) as forg_uproot:
        for bin in binc:
            bin_int = int(bin)  # Ensure bin is an integer
            log.info(f'Processing bin nHit0{bin_int:02d}')
             
            # Construct model map (point and extended sources)
            model = WCDA._get_model_map(bin, 0, 0).as_dense()
            for idx, keep in enumerate(pta):
                if not keep:
                    model += WCDA._get_model_map(bin, idx + 1, 0).as_dense()
                    if idx != 0:
                        model -= WCDA._get_model_map(bin, idx, 0).as_dense()
            for idx, keep in enumerate(exta):
                if not keep:
                    model += WCDA._get_model_map(bin, 0, idx + 1).as_dense()
                    if idx != 0:
                        model -= WCDA._get_model_map(bin, 0, idx).as_dense()
            # active_bin = WCDA._maptree._analysis_bins[bin_int]  # Use integer bin
            # Read data and background with uproot
            try:
                tdata = forg_uproot[f"nHit{bin_int:02d}/data"].arrays(library="np")
                tbkg = forg_uproot[f"nHit{bin_int:02d}/bkg"].arrays(library="np")
                data_counts = tdata["count"]
                bkg_counts = tbkg["count"]
            except KeyError as e:
                log.error(f"Failed to read TTree for bin nHit{bin_int:02d}: {e}")
                continue

            # Vectorized computation of residual map
            val_m = bkg_counts.copy()
            roi_mask = np.isin(np.arange(npix), pixid)
            roi_indices = np.searchsorted(pixid, np.arange(npix)[roi_mask])
            val_m[roi_mask] += model[roi_indices]

            # Write to output file with uproot
            with uproot.update(outroot) as fout_uproot:
                fout_uproot[f"nHit{bin_int:02d}/data"] = {"count": data_counts}
                fout_uproot[f"nHit{bin_int:02d}/bkg"] = {"count": val_m}

    # Add additional info with external tool
    os.system(f'{libdir}/tools/llh_skymap/Add_UserInfo {outroot} {binc[0]} {binc[-1]}')

    # Run significance skymap
    if ifrunllh:
        try:
            from hawc_hal import HealpixConeROI
        except ImportError:
            from WCDA_hal import HealpixConeROI
        if llhmode is None:
            roi2 = HealpixConeROI(ra=ra1, dec=dec1, data_radius=data_radius, model_radius=data_radius + 1)
            if s is None:
                s = binc[0]
            if e is None:
                e = binc[-1]
            runllhskymap(roi2, outroot, response, ra1, dec1, data_radius, outname, Modelname,
                        detector=detector, ifres=1, s=s, e=e, jc=jc, sn=sn)
        elif llhmode == "mp":
            roi2 = HealpixConeROI(ra=ra1, dec=dec1, data_radius=data_radius, model_radius=data_radius + 1)
            if s is None:
                s = binc[0]
            if e is None:
                e = binc[-1]
            map = runllhskymap_mp(roi, outroot, response, ra1, dec1, data_radius, outname, Modelname,
                            detector=detector, ifres=1, s=s, e=e, jc=jc)
            return map
        elif llhmode == "ipy":
    return outname + "_res"


def getllhskymap(inname, region_name, Modelname, ra1, dec1, data_radius, detector="WCDA", ifsave=True, ifdraw=False, drawfullsky=False, tofits=False):
    """
        搜集显著性天图作业的结果并保存fits文件

        Parameters:
        
        Returns:
            >>> healpix
    """ 
    import glob
    import os
    folder_path = f"{libdir}/tools/llh_skymap/sourcetxt/{detector}_{inname}"
    if not os.path.exists(folder_path):
        log.info(f"Bad path: {folder_path}")
        pass
    name = folder_path.replace("./","")
    all_files = glob.glob(os.path.join(folder_path, '*'))
    nside=1024
    npix=hp.nside2npix(nside)
    skymap=hp.UNSEEN*np.ones(npix)
    for file in all_files:
        datas = np.load(file, allow_pickle=True)[0]
        for dd in datas:
            if dd != []:
                dd2 = np.array(dd)
                if len(dd2) >0:
                    skymap[dd2[:,0].astype(np.int)]=dd2[:,1]
                    # decs = hp.pix2ang(nside, dd2[:,0].astype(np.int)[dd2[:,1]!=hp.UNSEEN])[0]
                    # decs = 90-decs*180/np.pi
                    # print(file, np.nanmean(decs)) #)
    print("done")
    skymap=hp.ma(skymap)
    if ifsave:
        print("save")
        hp.write_map(f"{libdir}/../res/{region_name}/{Modelname}/{detector}_{inname}.fits.gz", skymap, overwrite=True)
    if ifdraw:
        print("Draw")
        sources={}
        drawmap(region_name, Modelname, sources, skymap, ra1, dec1, rad=2*data_radius, contours=[10000],save=ifsave, savename=inname,
                cat={ "LHAASO": [0, "P"],"TeVCat": [0, "s"],"PSR": [0, "*"],"SNR": [0, "o"],"3FHL": [0, "D"], "size": 20,"markercolor": "grey","labelcolor": "black","angle": 60,"catext": 1 }, color="Fermi"
                  )
    if drawfullsky:
        print("Draw_fullsky")
        fig = mt.hpDraw("region_name", "Modelname", skymap,0,0,skyrange=(0,360,-20,80),
                    colorlabel="Significance", contours=[1000], save=False, cat={}, color="Milagro", xsize=2048)
    if tofits:
        print("Draw_fits")
        plt.figure()
        heal2fits(skymap, f"{libdir}/../res/{region_name}/{Modelname}/{detector}_{inname}.fits", ra1-data_radius/np.cos(np.radians(dec1)), ra1+data_radius/np.cos(np.radians(dec1)), 0.01/np.cos(np.radians(dec1)), dec1-data_radius, dec1+data_radius, 0.01, ifplot=1, ifnorm=0)
    return skymap


# import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
# import healpy as hp
def healpix_to_fits_pixels(healpix_pixels, nside, wcs):
    """
    将Healpix像素转换为FITS文件中的像素坐标 (x, y)。

    Parameters:
        healpix_pixels (list of int): Healpix格式的像素列表
        nside (int): Healpix的nside参数
        wcs (WCS): FITS文件的WCS信息

    Returns:
        list of tuple: 对应的FITS像素坐标列表
    """
    # 将Healpix像素转换为天球坐标 (RA, Dec)
    theta, phi = hp.pix2ang(nside, healpix_pixels)
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)

    # 将天球坐标 (RA, Dec) 转换为FITS像素坐标 (x, y)
    x, y = wcs.wcs_world2pix(ra, dec, 0)

    return list(zip(x.astype(int), y.astype(int)))

# def calculate_pixel_solid_angle(wcs, x, y):
#     """计算FITS文件中给定像素的立体角（以弧度为单位）。"""
#     # 获取像素四个角的天球坐标 (RA, Dec)
#     ra_dec_corners = wcs.pixel_to_world([[x-0.5, y-0.5],
#                                          [x+0.5, y-0.5],
#                                          [x+0.5, y+0.5],
#                                          [x-0.5, y+0.5]])

#     # 将天球坐标转换为球坐标 (theta, phi)
#     theta = np.radians(90 - ra_dec_corners.dec.value)
#     phi = np.radians(ra_dec_corners.ra.value)

#     # 使用球面多边形来估算像素的立体角
#     solid_angle = hp.pixelfunc.query_polygon(nside=8192, vertices=np.vstack([theta, phi]).T, inclusive=False)
    
#     return np.sum(solid_angle) * hp.pixelfunc.nside2pixarea(8192)

def calculate_pixel_solid_angle(wcs, x, y):
    """
    计算FITS文件中给定像素的立体角（以弧度为单位）。
    兼容赤道坐标系(RA/Dec)和银道坐标系(l/b)。
    
    参数:
        wcs: WCS对象
        x, y: 像素坐标
        
    返回:
        立体角（以弧度为单位）
    """
    # 获取像素中心的天球坐标
    coord = wcs.pixel_to_world(x, y)
    
    # 获取WCS的坐标系类型
    ctype1 = wcs.wcs.ctype[0].upper()
    ctype2 = wcs.wcs.ctype[1].upper()
    
    # 判断坐标系类型
    is_galactic = ('GLON' in ctype1 and 'GLAT' in ctype2)
    
    # 计算RA和Dec方向上像素的角大小（假设它们是小量）
    pixel_scale = wcs.pixel_scale_matrix
    delta_lon = np.abs(pixel_scale[0, 0])  # 经度方向像素的角大小，单位：度
    delta_lat = np.abs(pixel_scale[1, 1])  # 纬度方向像素的角大小，单位：度
    
    # 转换为弧度
    delta_lon_rad = np.radians(delta_lon)
    delta_lat_rad = np.radians(delta_lat)
    
    # 根据坐标系类型获取纬度值
    if is_galactic:
        # 银道坐标系，使用b(银纬)
        lat_value = coord.b.value
    else:
        # 赤道坐标系，使用dec(赤纬)
        try:
            lat_value = coord.dec.value
        except AttributeError:
            # 如果dec属性不存在，尝试其他可能的名称
            lat_value = getattr(coord, 'lat', getattr(coord, 'latitude', 0)).value
    
    # 使用cos(lat)因子计算立体角
    solid_angle = delta_lon_rad * delta_lat_rad * np.cos(np.radians(lat_value))
    
    # 调试输出
    if solid_angle == 0:
        print(f"Warning: Zero solid angle at pixel ({x}, {y})")
        print(f"Delta_lon: {delta_lon_rad}, Delta_lat: {delta_lat_rad}")
        print(f"Latitude value: {lat_value}, cos(lat): {np.cos(np.radians(lat_value))}")
    
    return solid_angle

def fits_pixel_to_healpix(ra, dec, nside):
    """将FITS像素的天球坐标 (RA, Dec) 转换为Healpix像素索引。"""
    # theta = np.radians(90 - dec)
    # phi = np.radians(ra)
    return hp.ang2pix(nside, ra, dec, lonlat=True)

def normalize_fits_within_healpix_roi(fits_file, output_file, healpix_pixels=None, nside=None):
    """
    对给定ROI内的FITS像素进行按立体角积分归一化,并保存为新的FITS文件,同时屏蔽掉ROI以外的像素。

    Parameters:
        fits_file (str): 输入的FITS文件路径
        healpix_pixels (set of int): Healpix格式的ROI像素集合
        nside (int): Healpix的nside参数
        output_file (str): 输出的FITS文件路径

    Returns:
        None
    """
    from decimal import Decimal
    # 读取FITS文件和数据
    hdu = fits.open(fits_file)
    data = hdu[0].data
    header = hdu[0].header

    # 获取WCS信息
    wcs = WCS(header)

    # 创建mask并初始化为True
    mask = np.ones(data.shape, dtype=bool)

    # 初始化归一化参数
    total_flux = 0.0
    total_solid_angle = 0.0

    S = np.zeros(data.shape, dtype=bool)
    # 遍历FITS文件的每一个像素
    pixel_solid_angles = []
    total_fluxs = []
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            # 将FITS像素转换为天球坐标 (RA, Dec)
            ra, dec = wcs.wcs_pix2world(x, y, 0)
            # print(ra, dec)
            # 判断该Healpix像素是否在ROI内
            if healpix_pixels is not None:
                # 将天球坐标转换为对应的Healpix像素索引
                healpix_index = fits_pixel_to_healpix(ra, dec, nside)
                if healpix_index in healpix_pixels:
                    # 计算当前FITS像素的立体角
                    pixel_solid_angle = calculate_pixel_solid_angle(wcs, x, y)

                    # 计算该像素的贡献
                    # print(pixel_solid_angle)
                    flux = data[y, x]
                    if np.isnan(pixel_solid_angle):
                        print("solid_angle_nan")
                    if pixel_solid_angle==0:
                        print("solid_angle_zero")
                    if np.isnan(flux):
                        continue
                    S[y,x] = pixel_solid_angle
                    pixel_solid_angles.append(pixel_solid_angle)
                    total_fluxs.append(flux * pixel_solid_angle)
                    # total_flux = total_flux + flux * Decimal(pixel_solid_angle)
                    # total_solid_angle = total_solid_angle + Decimal(pixel_solid_angle)
                    mask[y, x] = False  # 在ROI内的像素不被mask
            else:
                # 计算当前FITS像素的立体角
                pixel_solid_angle = calculate_pixel_solid_angle(wcs, x, y)

                # 计算该像素的贡献
                # print(pixel_solid_angle)
                flux = data[y, x]
                if np.isnan(pixel_solid_angle):
                    print("solid_angle_nan")
                if pixel_solid_angle==0:
                    print("solid_angle_zero")
                if np.isnan(flux):
                    continue
                S[y,x] = pixel_solid_angle
                pixel_solid_angles.append(pixel_solid_angle)
                total_fluxs.append(flux * pixel_solid_angle)
                # total_flux = total_flux + flux * Decimal(pixel_solid_angle)
                # total_solid_angle = total_solid_angle + Decimal(pixel_solid_angle)
                mask[y, x] = False  # 在ROI内的像素不被mask
    total_flux = math.fsum(total_fluxs)
    total_solid_angle = math.fsum(pixel_solid_angles)

    # if total_solid_angle == 0:
    #     print(total_solid_angle)
    #     raise ValueError("总立体角为0,可能ROI像素列表为空。")

    normalization_factor = total_flux #/ total_solid_angle
    print(normalization_factor)

    # 对ROI内的像素进行归一化处理
    normalized_data = data.copy()
    normalized_data[~mask] /= normalization_factor

    # 将ROI以外的像素屏蔽掉
    normalized_data[mask] = 0

    # 保存为新的FITS文件
    hdu[0].data = normalized_data
    hdu.writeto(output_file, overwrite=True)
    print(f"归一化后的FITS文件已保存为 {output_file}")

def process_pixel(args):
    x, y, data, wcs, healpix_pixels, nside = args
    ra, dec = wcs.wcs_pix2world(x, y, 0)
    if healpix_pixels is not None:
        healpix_index = fits_pixel_to_healpix(ra, dec, nside)
        if healpix_index in healpix_pixels:
            pixel_solid_angle = calculate_pixel_solid_angle(wcs, x, y)
            flux = data[y, x]
            if np.isnan(pixel_solid_angle) or pixel_solid_angle == 0 or np.isnan(flux):
                return None
            return (x, y, flux * pixel_solid_angle, pixel_solid_angle)
    else:
        pixel_solid_angle = calculate_pixel_solid_angle(wcs, x, y)
        flux = data[y, x]
        if np.isnan(pixel_solid_angle) or pixel_solid_angle == 0 or np.isnan(flux):
            return None
        return (x, y, flux * pixel_solid_angle, pixel_solid_angle)
    return None

def normalize_fits_within_healpix_roi_mp(fits_file, output_file, healpix_pixels=None, nside=None):
    import multiprocessing as mp
    hdu = fits.open(fits_file)
    data = hdu[0].data
    header = hdu[0].header
    wcs = WCS(header)
    mask = np.ones(data.shape, dtype=bool)

    args = [(x, y, data, wcs, healpix_pixels, nside) for y in range(data.shape[0]) for x in range(data.shape[1])]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_pixel, args)

    total_fluxs = []
    pixel_solid_angles = []
    for result in results:
        if result is not None:
            x, y, flux_solid_angle, solid_angle = result
            total_fluxs.append(flux_solid_angle)
            pixel_solid_angles.append(solid_angle)
            mask[y, x] = False

    total_flux = math.fsum(total_fluxs)
    total_solid_angle = math.fsum(pixel_solid_angles)
    normalization_factor = total_flux

    normalized_data = data.copy()
    normalized_data[~mask] /= normalization_factor
    normalized_data[mask] = 0

    hdu[0].data = normalized_data
    hdu.writeto(output_file, overwrite=True)
    print(f"归一化后的FITS文件已保存为 {output_file}, 归一化因子为 {normalization_factor}")


def generate_wcs_fits_with_range(coord_range, coordinate_system='equatorial', 
                                 projection='CAR', image_shape=(1024, 512), save=False, output_fits_path=None):
    """
    生成一个FITS文件，包含给定坐标系和投影方式的WCS，使用RA/Dec或l/b的坐标范围。

    参数:
        output_fits_path (str): 输出FITS文件路径。
        coord_range (dict): 图像的坐标范围，形式为 {'ra_range': (min, max), 'dec_range': (min, max)} 或 {'l_range': (min, max), 'b_range': (min, max)}。
        coordinate_system (str): 目标坐标系，'galactic' 表示银道坐标系，'equatorial' 表示赤道坐标系。
        projection (str): 投影类型，默认为'CAR'（正交投影），可选'AIT'（艾托夫投影）、'MOL'（莫尔维德投影）。
        image_shape (tuple): 图像尺寸，形式为 (宽度, 高度)，默认为 (1024, 512)。
    
    返回:
        WCS对象：包含生成的WCS信息。
    """
    # 创建WCS对象
    wcs = WCS(naxis=2)

    # 设置坐标系类型和投影方式
    if coordinate_system == 'galactic':
        wcs.wcs.ctype = [f'GLON-{projection}', f'GLAT-{projection}']  # 银道坐标
        lon_range = coord_range['l_range']
        lat_range = coord_range['b_range']
    elif coordinate_system == 'equatorial':
        wcs.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']  # 赤道坐标
        lon_range = coord_range['ra_range']
        lat_range = coord_range['dec_range']
    else:
        raise ValueError("无效的目标坐标系，必须是 'galactic' 或 'equatorial'")

    # 计算每像素的分辨率
    lon_size = lon_range[1] - lon_range[0]
    lat_size = lat_range[1] - lat_range[0]

    # 计算每像素的度数
    cdelt_lon = lon_size / image_shape[0]  # 经度方向的每像素度数
    cdelt_lat = lat_size / image_shape[1]  # 纬度方向的每像素度数

    # 设置参考像素为图像左下角，并将其对应的天球坐标设置为lon_range[0], lat_range[0]
    wcs.wcs.crpix = [1, 1]  # 左下角像素
    wcs.wcs.crval = [lon_range[0], lat_range[0]]  # 左下角的坐标

    # 设置像素尺寸
    wcs.wcs.cdelt = [cdelt_lon, cdelt_lat]  # 经度和纬度方向的每像素度数

    if save:
        # 将WCS写入HDR文件
        header = wcs.to_header()
        with open(output_fits_path, 'w') as hdr_file:
            hdr_file.write(header.tostring())

        print(f"WCS HDR file saved to {output_fits_path}")
    return wcs

def reproject_fits(input_fits_path, output_fits_path, to_system='galactic', target_wcs=None, projection='CAR'):
    """
    将FITS文件重新投影到不同的坐标系，支持赤道坐标系和银道坐标系的相互转换，适用于小区域和全天图像。

    参数:
        input_fits_path (str): 输入FITS文件路径
        output_fits_path (str): 输出FITS文件路径
        to_system (str): 目标坐标系，'galactic' 表示转换为银道坐标系，'equatorial' 表示转换为赤道坐标系
        target_wcs (WCS): 目标的WCS投影。如果为None，将根据目标坐标系生成默认WCS。
        projection (str): 投影类型，默认为'CAR'（正交投影），可选'AIT'（艾托夫投影）或'MOL'（莫尔维德投影）。
    """
    from reproject import reproject_interp
    from astropy.wcs.utils import proj_plane_pixel_scales
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u
    
    # 打开FITS文件并获取WCS信息和数据
    with fits.open(input_fits_path) as hdul:
        wcs_input = WCS(hdul[0].header)
        data = hdul[0].data

        # 如果没有提供目标WCS，则创建默认的目标WCS
        if target_wcs is None:
            # 获取FITS图像的像素尺寸
            pixel_scale = np.mean(proj_plane_pixel_scales(wcs_input)) * u.deg

            # 获取图像中心的天球坐标
            sky_center = wcs_input.pixel_to_world(data.shape[1] // 2, data.shape[0] // 2)

            # 设置目标WCS
            target_wcs = WCS(naxis=2)
            if to_system == 'galactic':
                sky_center_gal = sky_center.transform_to('galactic')
                target_wcs.wcs.ctype = [f'GLON-{projection}', f'GLAT-{projection}']  # 银道坐标系
                target_wcs.wcs.crval = [sky_center_gal.l.deg, sky_center_gal.b.deg]  # 中心坐标
            elif to_system == 'equatorial':
                sky_center_eq = sky_center.transform_to('icrs')
                target_wcs.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']  # 赤道坐标系
                target_wcs.wcs.crval = [sky_center_eq.ra.deg, sky_center_eq.dec.deg]  # 中心坐标
            else:
                raise ValueError("无效的目标坐标系，必须是 'galactic' 或 'equatorial'")

            # 定义目标WCS的参考像素为图像中心
            target_wcs.wcs.crpix = [data.shape[1] // 2, data.shape[0] // 2]
            target_wcs.wcs.cdelt = [-pixel_scale.to(u.deg).value, pixel_scale.to(u.deg).value]  # 每像素度数

        # 使用reproject进行图像重新投影
        print("重新投影进行中...")
        reprojected_data, _ = reproject_interp((data, wcs_input), target_wcs, shape_out=data.shape)

        # 创建新的FITS文件，并保存重新投影后的数据和WCS信息
        new_header = target_wcs.to_header()
        hdu = fits.PrimaryHDU(reprojected_data, header=new_header)
        hdu.writeto(output_fits_path, overwrite=True)

    print(f"FITS file reprojected to {to_system} coordinates and saved to {output_fits_path}")


def crop_fits_by_coords(input_fits_path, output_fits_path, ra_min, ra_max, dec_min, dec_max):
    # 打开FITS文件
    with fits.open(input_fits_path) as hdul:
        # 获取第一个扩展的WCS信息和图像数据
        wcs = WCS(hdul[0].header)
        data = hdul[0].data

        # 初始化裁剪后的数据
        cropped_data = None

        if ra_min > ra_max:
            # 第一部分：从ra_min到360度
            x_min1, y_min1 = wcs.all_world2pix(ra_min, dec_min, 0)
            x_max1, y_max1 = wcs.all_world2pix(360, dec_max, 0)
            x_min1, x_max1 = int(np.floor(x_min1)), int(np.ceil(x_max1))
            y_min1, y_max1 = int(np.floor(y_min1)), int(np.ceil(y_max1))

            cropped_data_1 = data[y_min1:y_max1, x_min1:x_max1]

            # 第二部分：从0度到ra_max
            x_min2, y_min2 = wcs.all_world2pix(0, dec_min, 0)
            x_max2, y_max2 = wcs.all_world2pix(ra_max, dec_max, 0)
            x_min2, x_max2 = int(np.floor(x_min2)), int(np.ceil(x_max2))
            y_min2, y_max2 = int(np.floor(y_min2)), int(np.ceil(y_max2))

            cropped_data_2 = data[y_min2:y_max2, x_min2:x_max2]

            # 拼接两部分
            cropped_data = np.hstack((cropped_data_1, cropped_data_2))

            # 更新WCS信息，分别处理两部分的CRPIX
            new_wcs = wcs.deepcopy()
            new_wcs.wcs.crpix[0] -= x_min1  # RA方向，第一部分
            new_wcs.wcs.crpix[1] -= y_min1  # Dec方向

            # 更新WCS信息，第二部分需要对RA方向进行偏移，保持连续性
            # 第二部分的CRPIX调整基于x_min2和cropped_data_1的宽度
            new_wcs.wcs.crpix[0] += cropped_data_1.shape[1]  # 第二部分RA相对于第一部分

        else:
            # 常规情况，RA没有跨越0度
            x_min, y_min = wcs.all_world2pix(ra_min, dec_min, 0)
            x_max, y_max = wcs.all_world2pix(ra_max, dec_max, 0)
            x_min, x_max = int(np.floor(x_min)), int(np.ceil(x_max))
            y_min, y_max = int(np.floor(y_min)), int(np.ceil(y_max))

            cropped_data = data[y_min:y_max, x_min:x_max]

            # 更新裁剪后的WCS信息
            new_wcs = wcs.deepcopy()
            new_wcs.wcs.crpix[0] -= x_min  # 更新RA方向的CRPIX1
            new_wcs.wcs.crpix[1] -= y_min  # 更新Dec方向的CRPIX2

        # 检查裁剪数据是否成功
        if cropped_data is None:
            raise ValueError("裁剪失败，可能由于输入坐标范围错误或数据问题")

        # 转换为header
        new_header = new_wcs.to_header()

        # 创建新的FITS文件，并保存裁剪后的数据和新的WCS信息
        hdu = fits.PrimaryHDU(cropped_data, header=new_header)
        hdu.writeto(output_fits_path, overwrite=True)

    print(f"Cropped FITS file saved to {output_fits_path}")

def get_fits_roi_bounds(fits_file):
    # Step 1: Load the FITS file and extract WCS and data
    with fits.open(fits_file) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        data = hdu.data
    
    # Step 2: Find the bounds of the non-zero (or non-NaN) data in FITS
    non_zero_y, non_zero_x = np.where(data > 0)  # or np.isfinite(data) for NaN check

    # Step 3: Get the min and max pixel coordinates where the data is valid
    xmin, xmax = non_zero_x.min(), non_zero_x.max()
    ymin, ymax = non_zero_y.min(), non_zero_y.max()

    # Step 4: Convert these pixel coordinates to sky coordinates (RA, Dec)
    ra_min, dec_min = wcs.pixel_to_world_values(xmin, ymin)
    ra_max, dec_max = wcs.pixel_to_world_values(xmax, ymax)
    
    # Return the sky bounds in terms of RA and Dec
    return (ra_min, ra_max, dec_min, dec_max)

def process_healpix_pixel(args):
    pix, nside, wcs, data, ra_min, ra_max, dec_min, dec_max, reverse = args
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    
    # if ra_min <= ra <= ra_max and dec_min <= dec <= dec_max:
    x, y = wcs.world_to_pixel_values(ra, dec)
    if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
        fits_value = data[int(y), int(x)]
        return pix, fits_value
    return pix, 0.0

def fits2healpix(fits_file, nside, reverse=False):
    import multiprocessing as mp
    ra_min, ra_max, dec_min, dec_max = get_fits_roi_bounds(fits_file)
    
    with fits.open(fits_file) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        data = hdu.data

    if reverse:
        data[data==1] = 5
        data[data==0] = 1
        data[data==5] = 0
    
    npix = hp.nside2npix(nside)
    healpix_map = np.zeros(npix, dtype=np.float64)
    
    args = [(pix, nside, wcs, data, ra_min, ra_max, dec_min, dec_max, reverse) for pix in range(npix)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_healpix_pixel, args)
    
    for pix, value in results:
        healpix_map[pix] = value
    
    return healpix_map

def find_brightest_pixel(healpix_map):
    """
    找到 Healpix 图像中最亮位置的索引和对应的值，忽略 NaN 值和 mask。

    参数:
    healpix_map (numpy.ndarray): Healpix 图像数据。

    返回:
    tuple: (最亮位置的索引, 最亮位置的值)
    """
    # 使用 np.nanmax 找到最大值，并忽略 NaN 和 mask
    max_value = np.nanmax(healpix_map[np.isfinite(healpix_map)])
    
    # 找到最大值的索引，使用 np.isnan 和 mask 进行筛选
    max_index = np.where(healpix_map == max_value)[0][0]
    
    return max_index, max_value
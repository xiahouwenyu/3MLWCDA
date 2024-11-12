import subprocess
import os 

import Mycoord
import inspect

libdir = os.path.dirname(os.path.dirname(inspect.getfile(Mycoord)))
# libdir = subprocess.run("pwd -P", shell=True, capture_output=True, text=True).stdout.replace("\n","")

def runllhskymap(roi, maptree, response, ra1, dec1, data_radius, region_name, detector="WCDA", ifres=False, jc=10, sn=1000, s=None,e=None):
    """
        交作业跑显著性天图, 结果用Mysigmap里面的getllhskymap查看

        Parameters:
            ifres: 是否标记为残差显著性天图
            jc: 每一个作业进程数
            sn: 每一个作业跑多少个pixel?
        Returns:
            >>> None
    """ 
    parts = int(len(roi.active_pixels(1024))/sn)+1
    if ifres:
        region_name=region_name+"_res"
    if detector=="WCDA":
        if s is None:
            s=0
        if e is None:
            e=6
        os.system(f"cd ./tools/llh_skymap/; rm -rf ./output/*; ./runwcdaall.sh {libdir}/{maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
        print(f"cd ./tools/llh_skymap/; rm -rf ./output/*; ./runwcdaall.sh {libdir}/{maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
    elif detector=="KM2A":
        if s is None:
            s=4
        if e is None:
            e=13
        os.system(f"cd ./tools/llh_skymap/; rm -rf ./output/*; ./runkm2aall.sh {libdir}/{maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
        print(f"cd ./tools/llh_skymap/; rm -rf ./output/*; ./runkm2aall.sh {libdir}/{maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn} {s} {e} {response} {libdir}")
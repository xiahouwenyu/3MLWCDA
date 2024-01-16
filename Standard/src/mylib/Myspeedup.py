import subprocess
import os 

libdir = subprocess.run("pwd -P", shell=True, capture_output=True, text=True).stdout.replace("\n","")

def runllhskymap(roi, maptree, ra1, dec1, data_radius, region_name, detector="WCDA", ifres=False, jc=10, sn=1000):
    parts = int(len(roi.active_pixels(1024))/sn)+1
    if ifres:
        region_name=region_name+"_res"
    if detector=="WCDA":
        os.system(f"cd ./tools/llh_skymap/; rm -rf ./output/*; ./runwcdaall.sh {libdir}/{maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn}")
    elif detector=="KM2A":
        os.system(f"cd ./tools/llh_skymap/; rm -rf ./output/*; ./runkm2aall.sh {libdir}/{maptree} {ra1} {dec1} {data_radius} {region_name} {parts} {libdir}/tools/llh_skymap {jc} {sn}")
#!/bin/bash

#SBATCH --job-name=km2a_pixfitting          # 作业名称
#SBATCH --ntasks=1                           # 1个任务
#SBATCH --mail-type=end
#SBATCH --mail-user=caowy@mail.ustc.edu.cn
#SBATCH --nodes=1
#SBATCH --partition=debug         # 替换为其他可用分区

source activate 3MLhal

# dirnow=dir
srcdir=${dirsrc}/
exe=${srcdir}tools/llh_skymap/pixfitting_spec_KM2A.py
cd $srcdir

# response=/data/home/cwy/Science/3MLWCDA/data/KM2A1234full_mcpsf_DRfinal.root
# map=$1
# ra=$2
# dec=$3
# radius=$4
# name=$5
# part=$6
# rm -rf ./output/*

time python3.9 ${exe} -m ${map} -r ${response} -ra ${ra} -dec ${dec} -radius ${radius} --s ${s} --e ${e} --name ${name} --jc ${jc} --sn ${sn} -part ${part} --o ${outdir}
source activate 3ML

exe=/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_KM2A.py

response=/data/home/cwy/Science/3MLWCDA/data/KM2A1234full_mcpsf_DRfinal.root
# map=$1
# ra=$2
# dec=$3
# radius=$4
# name=$5
# part=$6
# rm -rf ./output/*

time python3.9 ${exe} -m ${map} -r ${response} -ra ${ra} -dec ${dec} -radius ${radius} --s 0 --e 5 --name ${name} --jc ${jc} --sn ${sn} -part ${part} --o ${outdir}
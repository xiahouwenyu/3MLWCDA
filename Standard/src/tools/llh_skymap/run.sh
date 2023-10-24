# conda init bash
# conda activate 3ML
# dir=/home/lhaaso/caowy/0_Tools/sig_3ml
exe=./pixfitting_spec.py

# map=../../data/residual_all.root
# map=../../data/J0248.root
# map=/home/lhaaso/caowy/6_Source/data/gcd_new.root
# map=../../data/resall.root
# map=../../data/resall_gauss.root
# map=../../data/resall_releasebkg.root

# response=/home/lhaaso/caowy/6_Source/data/WCDA_DR_psf.root

map=/data/home/cwy/Science/0_Source/data/20210305_20230731_ihep_goodlist.root
# map=/data/home/cwy/Science/0_Source/Standard/res/J0248/gaus+2pt+fixDGE/J0248resall_DGE_cdiff.root
response=/data/home/cwy/Science/0_Source/data/DR_ihep_mk_MC2.root

name=Virgo
rm -rf *${name}*

python3.9 ${exe} -m ${map} -r ${response} --actBin 0 --name ${name}
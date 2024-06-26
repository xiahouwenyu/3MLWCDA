# conda init bash
# conda activate 3ML
source activate 3ML
# dir=/home/lhaaso/caowy/0_Tools/sig_3ml
exe=/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/pixfitting_spec_fullsky.py

# map=../../data/residual_all.root
# map=../../data/J0248.root
# map=/home/lhaaso/caowy/6_Source/data/gcd_new.root
# map=../../data/resall.root
# map=../../data/resall_gauss.root
# map=../../data/resall_releasebkg.root

# response=/home/lhaaso/caowy/6_Source/data/WCDA_DR_psf.root

# map=/data/home/cwy/Science/3MLWCDA0.91/data/20210305_20230731_ihep_goodlist.root
# map=/data/home/cwy/Science/3MLWCDA0.91/Standard/res/J0248/test/J0248res_J0248.root
# map=/data/home/cwy/Science/0_Source/Standard/res/J0248/gaus+2pt+fixDGE/J0248resall_DGE_cdiff.root
# response=/data/home/cwy/Science/3MLWCDA0.91/data/DR_ihep_MK2_newpsf.root

map=/data/home/cwy/Science/3MLWCDA/data/WCDA_20240131_out.root
response=/data/home/cwy/Science/3MLWCDA/data/DR_ihep_20240131_Crabpsf_mc.root

name=fullsky
rm -rf *${name}*

python3.9 ${exe} -m ${map} -r ${response} --actBin 0 --name ${name}
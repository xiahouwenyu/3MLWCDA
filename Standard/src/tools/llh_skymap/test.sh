# map=/data/home/cwy/Science/3MLWCDA/data/20210305-20230731_trans_fromhsc.root
map=/data/home/cwy/Science/3MLWCDA/Standard/res/J0057/1ext_freeDGE_0-5/J0057-diff.root
ra=14
dec=63.5
radius=6
name=J0057
# part=100
part=35
outdir=/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap
# ./runwcda.sh $map $ra $dec $radius $name $part $outdir
./runwcdaall.sh $map $ra $dec $radius $name $part $outdir
# rm -rf ./skytxt2/*txt
# rm -rf ./output/*.sh*
for i in {0..768}
do
qsub -v no=$i ./pixrun_noi_lm.sh -o ./output/ -e ./output/
done
rm -rf ./skytxt2/*txt
rm -rf ./output/*.sh*
for i in {0..768}
do
sbatch --export=no=$i ./pixrun_noi.sh -o ./output/ -e ./output/
done
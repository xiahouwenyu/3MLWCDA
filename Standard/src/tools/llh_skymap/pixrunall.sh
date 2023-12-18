for i in {0..768}
do
qsub -v no=$i ./pixrun_noi.sh -o ./output/ -e ./output/
done
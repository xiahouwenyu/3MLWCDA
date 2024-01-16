map=$1
ra=$2
dec=$3
radius=$4
name=$5
parts=$6
outdir=$7
jc=$8
sn=$9
rm -rf ./sourcetxt/WCDA_${name}*
for ((i=0; i<=1000; i++))
do
    if [ $i -gt $parts ]; then
        break
    fi
    qsub -v map=$map,ra=$ra,dec=$dec,radius=$radius,name=$name,part=$i,outdir=${outdir},jc=${jc},sn=${sn} -o ./output -e ./output -l nodes=5 ./runwcda.sh
done
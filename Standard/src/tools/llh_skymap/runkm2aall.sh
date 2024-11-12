map=$1
ra=$2
dec=$3
radius=$4
name=$5
parts=$6
outdir=$7
jc=$8
sn=$9
s=${10}
e=${11}
response=${12}
dirsrc=${13}

rm -rf ./sourcetxt/KM2A_${name}*
for i in $(seq 0 1000)
do
    if [ $i -gt $parts ]; then
        break
    fi
    # qsub -v map=$map,ra=$ra,dec=$dec,radius=$radius,name=$name,part=$i,outdir=${outdir},jc=${jc},sn=${sn},s=${s},e=${e},response=${response},dirsrc=${dirsrc} -o ./output/output${i}.log -e ./output/err${i}.log -l nodes=5 ./runkm2a.sh
    sbatch --export=map=$map,ra=$ra,dec=$dec,radius=$radius,name=$name,part=$i,outdir=${outdir},jc=${jc},sn=${sn},s=${s},e=${e},response=${response},dirsrc=${dirsrc} --output=./output/output${i}.log --error=./output/err${i}.log --nodes=3 ./runkm2a.sh
done
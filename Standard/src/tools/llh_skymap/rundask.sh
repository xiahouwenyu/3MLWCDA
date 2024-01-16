#!/bin/bash
rm -rf *log
qsub ./testdask.sh -o ./output -e ./output
file_path="./log.log"

while [ ! -f "$file_path" ]; do
    sleep 1
done

for i in {0..2}
do
    qsub ./testdask2.sh -o ./output -e ./output
done
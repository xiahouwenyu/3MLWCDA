#!/bin/bash
# rm -rf ./skytxt3/*txt

# 目标文件
file_path="/data/home/cwy/Science/3MLWCDA/Standard/src/tools/llh_skymap/skytxt3/list00.txt"

# 逐行读取文件
while IFS= read -r line
do
    # 对每行数字进行处理，这里只是打印出来，可以根据需要进行修改
    # echo "$line"
    qsub -v no=$line ./pixrun_noi_lm.sh -o ./output/ -e ./output/
done < "$file_path"

# 
# done
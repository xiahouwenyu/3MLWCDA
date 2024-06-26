#!/bin/bash

# 目标文件夹
folder_path="./"

# 列出包含2000个以上“0 0”的文件，并且只保留文件名中的数字
for file in "$folder_path"/*.txt; do
    count=$(grep -o "0 0" "$file" | wc -l)
    if [ "$count" -gt 2000 ]; then
        filename=$(basename "$file")
        number=$(echo "$filename" | grep -o -E '[0-9]+')
        echo "$number" >> list00.txt
    fi
done

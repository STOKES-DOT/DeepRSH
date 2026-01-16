#!/bin/bash

# 删除编号1295到3764的mol2文件
for i in {1295..3764}; do
    filename="${i}.mol2"
    [ -f "$filename" ] && rm "$filename"
done

echo "删除操作完成"
#!/bin/bash

# 遍历当前目录（.）下所有以 "episode" 开头的子文件夹
# -maxdepth 1 确保只查找当前目录下的子文件夹，不深入嵌套
# -type d 确保只选择目录（文件夹）
find . -maxdepth 1 -type d -name "episode*" | while read dir; do
    # 进入找到的子文件夹
    cd "$dir" || { echo "无法进入目录 $dir"; continue; }
    
    # 在子文件夹中执行 python ../convert.py
    # ../ 会引用到上一级目录，即包含这个脚本的目录
    echo "进入 $dir 并执行 python ../convert.py"
    python ../convert.py
    
    # 执行完毕后返回到原始目录，以便继续查找下一个文件夹
    cd ..
done

echo "所有操作已完成。"

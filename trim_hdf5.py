#!/usr/bin/env python3
"""
HDF5 数据修剪脚本
读取HDF5文件，将指定数据集的前x个数据保存到新文件中
"""

import h5py
import sys
import os
from pathlib import Path


def trim_hdf5(input_file, output_file, length):
    """
    修剪HDF5文件，保留前length个样本
    
    Args:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径
        length: 保留的数据长度
    """
    datasets_to_trim = [
        "/action",
        "/observations/images/cam_high",
        "/observations/qpos"
    ]
    
    with h5py.File(input_file, 'r') as f_in:
        with h5py.File(output_file, 'w') as f_out:
            # 复制所有数据集
            def copy_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # visititems返回的name不包含开头的/，需要加上
                    full_name = '/' + name
                    if full_name in datasets_to_trim:
                        # 需要修剪的数据集
                        print(f"  修剪 {full_name}: {len(obj)} -> {min(length, len(obj))}")
                        f_out.create_dataset(name, data=obj[:length])
                    else:
                        # 其他数据集直接复制
                        f_out.create_dataset(name, data=obj)
                elif isinstance(obj, h5py.Group):
                    # 创建组
                    f_out.create_group(name)
            
            f_in.visititems(copy_dataset)
            
            # 复制属性
            if f_in.attrs:
                for key, value in f_in.attrs.items():
                    f_out.attrs[key] = value


def main():
    # 读取命令行参数
    if len(sys.argv) > 1:
        length = int(sys.argv[1])
    else:
        length = 600
    
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    else:
        input_file = None
    
    print(f"修剪长度: {length}")
    
    if input_file:
        # 处理单个文件
        if not os.path.exists(input_file):
            print(f"错误: 文件不存在 {input_file}")
            return
        
        # 生成输出文件名
        path = Path(input_file)
        output_file = path.parent / f"{path.stem}-trimed{path.suffix}"
        
        print(f"处理文件: {input_file}")
        trim_hdf5(input_file, str(output_file), length)
        print(f"已保存到: {output_file}")
    else:
        # 处理当前目录的所有HDF5文件
        hdf5_files = list(Path('.').glob('*.hdf5'))
        
        if not hdf5_files:
            print("当前目录没有找到.hdf5文件")
            return
        
        for input_file in hdf5_files:
            output_file = input_file.parent / f"{input_file.stem}-trimed{input_file.suffix}"
            print(f"\n处理文件: {input_file}")
            trim_hdf5(str(input_file), str(output_file), length)
            print(f"已保存到: {output_file}")


if __name__ == '__main__':
    main()

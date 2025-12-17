# 将提供的hdf5的文件中的"arm/jointStatePosition/pika"下的数据保存到新的hdf5文件的"observations/qpos"中，以及"action"中
# 将"camera/color/pikaDepthCamera"下的数据保存到"observations/images/cam_high_path"中
# 将cam_high_path中的值对应的图片的内容读取并保存为ndarray，保存到"observations/images/cam_high"中
# 然后保存新的hdf5文件到本地

import h5py
import numpy as np
from PIL import Image
import io
import os

# 定义函数来打印HDF5文件结构
def print_hdf5_structure(name, obj):
    print(f"{name}: {type(obj)}")

# 打开原始hdf5文件
hdf5_path = 'data.hdf5'
hdf5_dir = os.path.dirname(os.path.abspath(hdf5_path))
with h5py.File(hdf5_path, 'r') as src_file:
    print("HDF5文件结构：")
    src_file.visititems(print_hdf5_structure)
    print("\n开始转换...\n")
    
    # 创建新的hdf5文件
    with h5py.File('data_converted.hdf5', 'w') as dst_file:
        # 创建observations组
        obs_grp = dst_file.create_group('observations')
        
        # 复制arm/jointStatePosition/pika数据到observations/qpos
        if 'arm/jointStatePosition/pika' in src_file:
            data_qpos = src_file['arm/jointStatePosition/pika'][:]
            obs_grp.create_dataset('qpos', data=data_qpos)
            print(f"已保存 'observations/qpos' 数据，形状: {data_qpos.shape}")
        
        # 复制camera/color/pikaDepthCamera数据到observations/images/cam_high
        if 'camera/color/pikaDepthCamera' in src_file:
            cam_high_paths = src_file['camera/color/pikaDepthCamera'][:]
            img_subgrp = obs_grp.create_group('images')
            
            # 保存图片路径
            img_subgrp.create_dataset('cam_high_path', data=cam_high_paths)
            print(f"已保存 'observations/images/cam_high_path' 数据，形状: {cam_high_paths.shape}")
            
            # 读取图片并保存为ndarray
            images_list = []
            for i, img_path in enumerate(cam_high_paths):
                # 处理字节路径
                if isinstance(img_path, bytes):
                    img_path = img_path.decode('utf-8')
                
                # 构建完整的图片路径
                full_img_path = os.path.join(hdf5_dir, img_path)
                
                try:
                    # 读取图片
                    if os.path.exists(full_img_path):
                        img = Image.open(full_img_path)
                        img_array = np.array(img)
                        images_list.append(img_array)
                        if i % 100 == 0:
                            print(f"已读取图片 {i}: {full_img_path}，形状: {img_array.shape}")
                    else:
                        print(f"警告: 图片路径不存在 {full_img_path}")
                except Exception as e:
                    print(f"错误: 无法读取图片 {full_img_path}, 错误: {e}")
            
            # 保存图片数组
            if images_list:
                images_array = np.array(images_list)
                img_subgrp.create_dataset('cam_high', data=images_array)
                print(f"已保存 'observations/images/cam_high' 数据，形状: {images_array.shape}")
        
        # 保存action数据，与qpos使用相同的数据源
        if 'arm/jointStatePosition/pika' in src_file:
            data_action = src_file['arm/jointStatePosition/pika'][:]
            dst_file.create_dataset('action', data=data_action)
            print(f"已保存 'action' 数据，形状: {data_action.shape}")

print("新的hdf5文件已保存为 'data_converted.hdf5'")
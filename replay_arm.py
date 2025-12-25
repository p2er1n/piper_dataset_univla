#!/usr/bin/env python3
"""
通过piper_sdk在Piper机械臂上回放收集的HDF5数据。
将数据集中的action数据转换为机械臂控制命令。
"""

import h5py
import time
import argparse
from pathlib import Path
from piper_sdk import C_PiperInterface_V2


def convert_joint_angles(radian_angles):
    """
    将关节角度从弧度转换为0.001度单位。
    
    参数:
        radian_angles: 6个关节角度列表，单位为弧度
        
    返回:
        6个关节角度列表，单位为0.001度
    """
    factor = 180 / 3.14159265359 * 1000  # 弧度转0.001度
    return [round(angle * factor) for angle in radian_angles]


def convert_gripper_angle(meter_angle):
    """
    将夹爪角度从米转换为0.001毫米单位。
    
    参数:
        meter_angle: 夹爪角度，单位为米
        
    返回:
        夹爪角度，单位为0.001毫米
    """
    factor = 1000000  # 米转0.001毫米
    return round(abs(meter_angle) * factor)


def replay_episode(hdf5_path, device="can0", episode_idx=0, speed=100, fps=30):
    """
    从HDF5文件中回放单个回合的数据到机械臂。
    
    参数:
        hdf5_path: HDF5文件路径
        device: CAN设备名称 (默认: "can0")
        episode_idx: 回放第几个回合 (默认: 0)
        speed: 运动速度参数 (默认: 100)
        fps: 回放帧率 (默认: 30)
    """
    # 打开HDF5文件
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # 获取action和observation数据
        actions = hdf5_file['action'][()]  # [episode_len, 7]
        episode_len = len(actions)
        
        print(f"已加载回合，包含 {episode_len} 个步骤")
        print(f"动作数据形状: {actions.shape}")
        
        # 初始化机械臂接口
        piper = C_PiperInterface_V2(device)
        piper.ConnectPort()
        
        # 启用机械臂
        max_retries = 100
        retry_count = 0
        while not piper.EnablePiper() and retry_count < max_retries:
            time.sleep(0.01)
            retry_count += 1
        
        if retry_count >= max_retries:
            print("多次尝试后启用机械臂失败")
            return
        
        print("机械臂启用成功")
        
        # 初始化夹爪
        piper.GripperCtrl(0, 1000, 0x01, 0)
        
        # 计算每步的睡眠时间
        sleep_time = 1.0 / fps
        
        # 回放每一步
        for step_idx, action in enumerate(actions):
            # 提取关节角度 (前6个自由度) 和夹爪 (第7个自由度)
            joint_angles_rad = action[:6]  # 单位：弧度
            gripper_angle_m = action[6]     # 单位：米
            
            # 转换单位
            joint_angles_ctrl = convert_joint_angles(joint_angles_rad)
            gripper_angle_ctrl = convert_gripper_angle(gripper_angle_m)
            
            # 控制机械臂
            piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
            piper.JointCtrl(
                joint_angles_ctrl[0],
                joint_angles_ctrl[1],
                joint_angles_ctrl[2],
                joint_angles_ctrl[3],
                joint_angles_ctrl[4],
                joint_angles_ctrl[5]
            )
            piper.GripperCtrl(gripper_angle_ctrl, 1000, 0x01, 0)
            
            # 获取并打印状态
            status = piper.GetArmStatus()
            print(f"步骤 {step_idx}/{episode_len}: 关节角度={action[:6]}, 夹爪={action[6]:.4f}")
            
            # 控制频率
            time.sleep(sleep_time)
        
        print("回放完成")


def list_episodes(hdf5_path):
    """
    列出HDF5文件中可用的所有回合。
    
    参数:
        hdf5_path: HDF5文件路径
    """
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        print(f"HDF5文件: {hdf5_path}")
        print(f"顶层键: {list(hdf5_file.keys())}")
        
        if 'action' in hdf5_file:
            print(f"动作数据形状: {hdf5_file['action'].shape}")
        
        if 'observations' in hdf5_file:
            obs = hdf5_file['observations']
            print(f"观测数据键: {list(obs.keys())}")
            if 'images' in obs:
                print(f"  图像键: {list(obs['images'].keys())}")
            if 'qpos' in obs:
                print(f"  关节位置形状: {obs['qpos'].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="在Piper机械臂上回放HDF5数据集"
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="HDF5数据集文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="can0",
        help="CAN设备名称 (默认: can0)"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        help="运动速度参数 (默认: 100)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="回放帧率，即一秒钟执行多少步 (默认: 30)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出数据集结构并退出"
    )
    
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_file)
    
    if not hdf5_path.exists():
        print(f"错误: 文件不存在: {hdf5_path}")
        return
    
    if args.list:
        list_episodes(hdf5_path)
    else:
        try:
            replay_episode(
                hdf5_path,
                device=args.device,
                speed=args.speed,
                fps=args.fps
            )
        except KeyboardInterrupt:
            print("\n用户中断回放")
        except Exception as e:
            print(f"回放过程中出错: {e}")


if __name__ == "__main__":
    main()

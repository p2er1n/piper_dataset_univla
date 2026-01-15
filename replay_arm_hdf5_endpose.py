#!/usr/bin/env python3
"""
Replay HDF5 end-effector states on a Piper arm using EndPoseCtrl.

Expected HDF5 datasets:
  arm/endPose/pika_end: [x, y, z, rx, ry, rz, ...]
  arm/jointStatePosition/pika: gripper value (optional, uses last dim)
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from piper_sdk import C_PiperInterface_V2


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def state_to_endpose(state, pos_scale, rot_scale):
    x, y, z, rx, ry, rz = state[:6]
    X = round(x * pos_scale)
    Y = round(y * pos_scale)
    Z = round(z * pos_scale)
    RX = round(rx * rot_scale)
    RY = round(ry * rot_scale)
    RZ = round(rz * rot_scale)
    return X, Y, Z, RX, RY, RZ


def gripper_to_ctrl(gripper):
    return round(float(gripper) * 1000000.0)


def load_endpose(hdf5_path, gripper_index):
    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "arm/endPose/pika_end" not in hdf5_file:
            raise KeyError("missing dataset: arm/endPose/pika_end")
        end_pose = hdf5_file["arm/endPose/pika_end"][:]
        if end_pose.ndim == 1:
            end_pose = end_pose[:, None]
        if end_pose.shape[1] < 6:
            raise ValueError("endPose has fewer than 6 dims")

        gripper = None
        if "arm/jointStatePosition/pika" in hdf5_file:
            qpos = hdf5_file["arm/jointStatePosition/pika"][:]
            if qpos.ndim == 1:
                gripper = qpos
            else:
                gripper = qpos[:, gripper_index]
        elif end_pose.shape[1] >= 7:
            gripper = end_pose[:, 6]

    if gripper is None:
        gripper = np.zeros(end_pose.shape[0], dtype=np.float32)

    steps = min(end_pose.shape[0], gripper.shape[0])
    return end_pose[:steps], gripper[:steps]


def list_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hdf5_file:
        print(f"HDF5 file: {hdf5_path}")
        print(f"top-level keys: {list(hdf5_file.keys())}")
        if "arm" in hdf5_file:
            print(f"arm keys: {list(hdf5_file['arm'].keys())}")
        if "arm/endPose/pika_end" in hdf5_file:
            print(f"endPose shape: {hdf5_file['arm/endPose/pika_end'].shape}")
        if "arm/jointStatePosition/pika" in hdf5_file:
            print(
                "jointStatePosition shape: "
                f"{hdf5_file['arm/jointStatePosition/pika'].shape}"
            )


def replay_episode(
    hdf5_path,
    device,
    speed,
    fps,
    pos_scale,
    rot_scale,
    gripper_index,
    dry_run,
):
    end_pose, gripper = load_endpose(hdf5_path, gripper_index)

    piper = None
    if not dry_run:
        piper = C_PiperInterface_V2(device)
        piper.ConnectPort()

        max_retries = 100
        retry_count = 0
        while not piper.EnablePiper() and retry_count < max_retries:
            time.sleep(0.01)
            retry_count += 1
        if retry_count >= max_retries:
            raise RuntimeError("Failed to enable Piper after retries")

        piper.GripperCtrl(0, 1000, 0x01, 0)

    sleep_time = 1.0 / fps if fps > 0 else 0.0
    for step_idx in range(end_pose.shape[0]):
        pose = end_pose[step_idx].astype(np.float32)
        X, Y, Z, RX, RY, RZ = state_to_endpose(pose, pos_scale, rot_scale)
        gripper_ctrl = gripper_to_ctrl(gripper[step_idx])

        if not dry_run:
            piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
            piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            piper.GripperCtrl(abs(gripper_ctrl), 1000, 0x01, 0)

        print(
            f"step {step_idx}: "
            f"endpose=({X}, {Y}, {Z}, {RX}, {RY}, {RZ}) "
            f"endpose_raw=({pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}, "
            f"{pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}) "
            f"gripper_raw={float(gripper[step_idx]):.6f} "
            f"gripper={gripper_ctrl}"
        )

        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 end-effector states using EndPoseCtrl."
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "--device",
        default="can0",
        help="CAN device name (default: can0).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        help="Motion speed (default: 100).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Replay rate in steps per second (default: 30).",
    )
    parser.add_argument(
        "--pos_scale",
        type=float,
        default=1000000.0,
        help="Position scale to EndPoseCtrl units (default: 1000000).",
    )
    parser.add_argument(
        "--rot_scale",
        type=float,
        default=57295.7795,
        help="Rotation scale to EndPoseCtrl units (default: 57295.7795).",
    )
    parser.add_argument(
        "--gripper_index",
        type=int,
        default=-1,
        help="Index to read gripper from jointStatePosition (default: -1).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only, do not control the arm.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List HDF5 structure and exit.",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_file)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"hdf5 file not found: {hdf5_path}")

    if args.list:
        list_hdf5(hdf5_path)
        return

    replay_episode(
        hdf5_path=hdf5_path,
        device=args.device,
        speed=args.speed,
        fps=args.fps,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        gripper_index=args.gripper_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

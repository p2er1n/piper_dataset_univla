#!/usr/bin/env python3
"""
Replay RLDS dataset end-effector states on a Piper arm using EndPoseCtrl.

Expected RLDS step format (from convert_endpose.py):
  observation/state = [x, y, z, rx, ry, rz, padding, gripper]

gripper is in [-1, 1], where -1 = open, 1 = close.
The gripper range is 0 - 0.07 m, and GripperCtrl expects distance in 0.01 mm.

Replay modes:
- state: use observation/state for every step (default).
- action_delta: initialize from observation/state[0], then apply
  action = [dx, dy, dz, rx, ry, rz, ...] each step to update pose.
"""

import argparse
import time

import numpy as np
import tensorflow_datasets as tfds
from piper_sdk import C_PiperInterface_V2


def iter_steps(steps):
    if isinstance(steps, dict):
        keys = list(steps.keys())
        first_key = keys[0] if keys else None
        if first_key is None:
            return
        length = len(steps[first_key])
        for i in range(length):
            yield {k: steps[k][i] for k in keys}
        return
    if isinstance(steps, np.ndarray):
        for item in steps:
            yield item
        return
    for item in steps:
        yield item


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


def gripper_to_ctrl(gripper, max_range_m, gripper_scale):
    g = float(gripper)
    g = clamp(g, -1.0, 1.0)
    t = (g + 1.0) / 2.0
    distance_m = max_range_m * (1.0 - t)
    return round(distance_m * gripper_scale)


def apply_action_delta(state, action, gripper_index):
    if action.shape[0] < 6:
        raise ValueError("action has fewer than 6 dims; expected dx dy dz rx ry rz")

    next_state = state.copy()
    next_state[0:3] = next_state[0:3] + action[0:3]
    next_state[3:6] = action[3:6]
    if action.shape[0] >= 7 and gripper_index is not None:
        if not (-action.shape[0] <= gripper_index < action.shape[0]):
            raise IndexError("action_gripper_index out of range for action shape")
        next_state[7] = action[gripper_index]
    return next_state


def get_episode(builder, split, episode_idx):
    ds = builder.as_dataset(split=split)
    for idx, episode in enumerate(tfds.as_numpy(ds)):
        if idx == episode_idx:
            return episode
    raise IndexError(f"episode_idx {episode_idx} out of range")


def list_episodes(builder, split, max_print):
    ds = builder.as_dataset(split=split)
    count = 0
    for episode in tfds.as_numpy(ds):
        if count < max_print:
            steps = episode.get("steps", [])
            step_count = 0
            for _ in iter_steps(steps):
                step_count += 1
            print(f"episode {count}: steps={step_count}")
        count += 1
    print(f"total episodes: {count}")


def replay_episode(
    rlds_dir,
    split,
    episode_idx,
    device,
    speed,
    fps,
    pos_scale,
    rot_scale,
    max_gripper_range_m,
    gripper_scale,
    mode,
    action_gripper_index,
    dry_run,
):
    builder = tfds.builder_from_directory(rlds_dir)
    episode = get_episode(builder, split, episode_idx)
    steps = episode["steps"]

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
    step_idx = 0

    if mode == "action_delta":
        step_iter = iter_steps(steps)
        first_step = next(step_iter, None)
        if first_step is None:
            return
        current_state = first_step["observation"]["state"].astype(np.float32)

        step = first_step
        while step is not None:
            action = step["action"].astype(np.float32)
            current_state = apply_action_delta(
                current_state,
                action,
                action_gripper_index,
            )
            X, Y, Z, RX, RY, RZ = state_to_endpose(
                current_state, pos_scale, rot_scale
            )
            gripper_ctrl = gripper_to_ctrl(
                current_state[7],
                max_gripper_range_m,
                gripper_scale,
            )

            if not dry_run:
                piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
                piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                piper.GripperCtrl(abs(gripper_ctrl), 1000, 0x01, 0)

            print(
                f"step {step_idx}: "
                f"action=({action[0]:.6f}, {action[1]:.6f}, {action[2]:.6f}, "
                f"{action[3]:.6f}, {action[4]:.6f}, {action[5]:.6f}) "
                f"endpose=({X}, {Y}, {Z}, {RX}, {RY}, {RZ}) "
                f"state_raw=({current_state[0]:.6f}, {current_state[1]:.6f}, "
                f"{current_state[2]:.6f}, {current_state[3]:.6f}, "
                f"{current_state[4]:.6f}, {current_state[5]:.6f}) "
                f"gripper_raw={current_state[7]:.6f} "
                f"gripper={gripper_ctrl}"
            )
            step_idx += 1
            if sleep_time > 0:
                time.sleep(sleep_time)
            step = next(step_iter, None)
    else:
        for step in iter_steps(steps):
            obs = step["observation"]
            state = obs["state"].astype(np.float32)
            X, Y, Z, RX, RY, RZ = state_to_endpose(state, pos_scale, rot_scale)
            gripper_ctrl = gripper_to_ctrl(
                state[7],
                max_gripper_range_m,
                gripper_scale,
            )

            if not dry_run:
                piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
                piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
                piper.GripperCtrl(abs(gripper_ctrl), 1000, 0x01, 0)

            print(
                f"step {step_idx}: "
                f"endpose=({X}, {Y}, {Z}, {RX}, {RY}, {RZ}) "
                f"endpose_raw=({state[0]:.6f}, {state[1]:.6f}, {state[2]:.6f}, {state[3]:.6f}, {state[4]:.6f}, {state[5]:.6f}) "
                f"gripper_raw={state[7]:.6f} "
                f"gripper={gripper_ctrl}"
            )
            step_idx += 1
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="Replay RLDS end-effector states using EndPoseCtrl."
    )
    parser.add_argument(
        "--rlds_dir",
        required=True,
        help="Path to TFDS dataset directory (contains dataset_info.json).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="TFDS split name (default: train).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay (default: 0).",
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
        "--gripper_max_m",
        type=float,
        default=0.07,
        help="Gripper max opening in meters (default: 0.07).",
    )
    parser.add_argument(
        "--gripper_scale",
        type=float,
        default=1000000.0,
        help="Meters to GripperCtrl units (0.01mm -> 1000000).",
    )
    parser.add_argument(
        "--mode",
        choices=("state", "action_delta"),
        default="state",
        help="Replay mode: state or action_delta (default: state).",
    )
    parser.add_argument(
        "--action_gripper_index",
        type=int,
        default=-1,
        help="Action index to read gripper when using action_delta (default: -1).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List episode counts and exit.",
    )
    parser.add_argument(
        "--list_max",
        type=int,
        default=5,
        help="Max episodes to print when listing (default: 5).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only, do not control the arm.",
    )
    args = parser.parse_args()

    builder = tfds.builder_from_directory(args.rlds_dir)
    if args.list:
        list_episodes(builder, args.split, args.list_max)
        return

    replay_episode(
        rlds_dir=args.rlds_dir,
        split=args.split,
        episode_idx=args.episode,
        device=args.device,
        speed=args.speed,
        fps=args.fps,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        max_gripper_range_m=args.gripper_max_m,
        gripper_scale=args.gripper_scale,
        mode=args.mode,
        action_gripper_index=args.action_gripper_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

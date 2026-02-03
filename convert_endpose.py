# 将提供的hdf5的文件中的"arm/jointStatePosition/pika"下的数据用于构建 action
# action 的前六个部分从 "arm/endPose/piper_end" 中获取，其余部分沿用 qpos
# 将 "camera/color/pikaDepthCamera" 下的图片读取并 resize 到 256x256，保存到 RLDS 的 observation/image
# 将 "arm/endPose/piper_end" 前六维 + padding + gripper 保存到 observation/state
# qpos 不再保存
# 指令通过脚本参数提供，language_instruction 需要为每一步保存，形状为 (episode_len,)

import argparse
import os

import h5py
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
from etils import epath


def debug_print(enabled, message):
    if enabled:
        print(message)


def print_hdf5_structure(name, obj):
    print(f"{name}: {type(obj)}")


def load_action(src_file, sample_interval=1, debug=False):
    if "arm/jointStatePosition/pika" not in src_file:
        raise KeyError("未找到 'arm/jointStatePosition/pika'")

    data_action = src_file["arm/jointStatePosition/pika"][:]
    action = data_action.copy()
    original_steps = action.shape[0]

    if "arm/endPose/piper_end" in src_file:
        end_pose = src_file["arm/endPose/piper_end"][:]
        if action.shape[0] != end_pose.shape[0]:
            print("警告: endPose 与 qpos 的时间步长度不一致，按最小长度对齐")
            min_steps = min(action.shape[0], end_pose.shape[0])
            action = action[:min_steps]
            end_pose = end_pose[:min_steps]
        if sample_interval > 1:
            action = action[::sample_interval]
            end_pose = end_pose[::sample_interval]
            debug_print(
                debug,
                f"先采样再计算差值: sample_interval={sample_interval}, 步长变为 {action.shape[0]}",
            )

        if action.ndim == end_pose.ndim:
            slice_len = min(6, action.shape[-1], end_pose.shape[-1])
            action[..., :slice_len] = end_pose[..., :slice_len]
            debug_print(debug, f"已用 endPose 替换 action 前 {slice_len} 个维度")
            if slice_len >= 3 and action.shape[0] >= 2:
                for i in range(action.shape[0] - 1):
                    diff_xyz = end_pose[i + 1, :3] - end_pose[i, :3]
                    debug_print(
                        debug,
                        "差值索引: {prev_idx} -> {next_idx}, "
                        "prev_xyz={prev_xyz}, next_xyz={next_xyz}, diff_xyz={diff_xyz}".format(
                            prev_idx=i,
                            next_idx=i + 1,
                            prev_xyz=end_pose[i, :3],
                            next_xyz=end_pose[i + 1, :3],
                            diff_xyz=diff_xyz,
                        )
                    )
                diff_xyz = end_pose[1:, :3] - end_pose[:-1, :3]
                action[:-1, :3] = diff_xyz
                if slice_len > 3:
                    if slice_len >= 6:
                        diff_rpy = end_pose[1:, 3:6] - end_pose[:-1, 3:6]
                        action[:-1, 3:6] = diff_rpy
                        if slice_len > 6:
                            action[:-1, 6:slice_len] = end_pose[1:, 6:slice_len]
                    else:
                        action[:-1, 3:slice_len] = end_pose[1:, 3:slice_len]
                if action.shape[-1] > slice_len:
                    action[:-1, slice_len:] = action[1:, slice_len:]
                action = action[:-1]
                if slice_len >= 6:
                    for i in range(action.shape[0]):
                        debug_print(
                            debug,
                            "新步索引 {new_idx} 的 action[3:6]={new_vals} 来自原索引 {src_idx} 的 end_pose 差值 {diff_vals}".format(
                                new_idx=i,
                                src_idx=i + 1,
                                new_vals=action[i, 3:6],
                                diff_vals=end_pose[i + 1, 3:6] - end_pose[i, 3:6],
                            )
                        )
                debug_print(debug, "已将 endPose 前三维替换为相邻步差值，其余部分使用下一步数据，并丢弃最后一步")
            else:
                print("警告: endPose 维度不足或步长过短，未计算前三维差值")
        else:
            print("警告: endPose 与 qpos 维度不一致，未替换 action 前六个部分")
    else:
        print("警告: 未找到 'arm/endPose/piper_end'，action 将与 qpos 相同")

    if action.ndim == 1:
        action = action[:, None]

    if action.shape[0] != original_steps:
        debug_print(debug, f"action 步长对齐: {original_steps} -> {action.shape[0]}")

    return action


def binarize_gripper_top10(action, gripper_index=-1, quantile=0.1, debug=False):
    gripper = action[:, gripper_index].astype(np.float32)
    threshold = np.quantile(gripper, quantile)
    binarized = np.where(gripper <= threshold, 1.0, -1.0)
    for i in range(action.shape[0]):
        debug_print(
            debug,
            "gripper 索引 {idx}: 原值 {raw_val} -> 二值化 {bin_val}".format(
                idx=i,
                raw_val=gripper[i],
                bin_val=binarized[i],
            )
        )
    action[:, gripper_index] = binarized
    return action


def load_state(
    src_file, action_len, sample_interval=1, gripper_index=-1, debug=False
):
    if "arm/endPose/piper_end" not in src_file:
        raise KeyError("未找到 'arm/endPose/piper_end'")
    if "arm/jointStatePosition/pika" not in src_file:
        raise KeyError("未找到 'arm/jointStatePosition/pika'")

    end_pose = src_file["arm/endPose/piper_end"][:]
    qpos = src_file["arm/jointStatePosition/pika"][:]
    min_steps = min(end_pose.shape[0], qpos.shape[0])
    if min_steps == 0:
        raise ValueError("endPose 或 qpos 为空，无法构建 state")
    if end_pose.shape[0] != qpos.shape[0]:
        debug_print(debug, "警告: endPose 与 qpos 的时间步长度不一致，按最小长度对齐")
    end_pose = end_pose[:min_steps]
    qpos = qpos[:min_steps]
    if sample_interval > 1:
        end_pose = end_pose[::sample_interval]
        qpos = qpos[::sample_interval]
        debug_print(
            debug,
            f"state 先采样再对齐: sample_interval={sample_interval}, 步长变为 {end_pose.shape[0]}",
        )

    state_steps = min(action_len, end_pose.shape[0] - 1)
    if state_steps <= 0:
        raise ValueError("action 长度不足，无法构建 state")

    state = np.zeros((state_steps, 8), dtype=np.float32)
    pose_dim = min(6, end_pose.shape[-1])
    state[:, :pose_dim] = end_pose[:state_steps, :pose_dim]
    state[:, 6] = 0.0

    if qpos.ndim == 1:
        gripper = qpos[:state_steps]
    else:
        gripper = qpos[:state_steps, gripper_index]
    state[:, 7] = gripper.astype(np.float32)
    return state


def load_images(
    src_file, hdf5_dir, sample_interval=1, resize_hw=(256, 256), debug=False
):
    if "camera/color/pikaDepthCamera" not in src_file:
        raise KeyError("未找到 'camera/color/pikaDepthCamera'")

    cam_high_paths = src_file["camera/color/pikaDepthCamera"][:]
    if sample_interval > 1:
        cam_high_paths = cam_high_paths[::sample_interval]
        debug_print(
            debug,
            f"图片路径先采样: sample_interval={sample_interval}, 数量变为 {len(cam_high_paths)}",
        )
    images_list = []
    placeholder = np.zeros((resize_hw[1], resize_hw[0], 3), dtype=np.uint8)

    for i, img_path in enumerate(cam_high_paths):
        if isinstance(img_path, bytes):
            img_path = img_path.decode("utf-8")

        full_img_path = os.path.join(hdf5_dir, img_path)
        try:
            if os.path.exists(full_img_path):
                img = Image.open(full_img_path).convert("RGB")
                img = img.resize(resize_hw, Image.BILINEAR)
                img_array = np.array(img)
                images_list.append(img_array)
                if i % 100 == 0:
                    debug_print(debug, f"已读取图片 {i}: {full_img_path}，形状: {img_array.shape}")
            else:
                print(f"警告: 图片路径不存在 {full_img_path}，使用占位图")
                images_list.append(placeholder)
        except Exception as e:
            print(f"错误: 无法读取图片 {full_img_path}, 错误: {e}，使用占位图")
            images_list.append(placeholder)

    return images_list


class EndposeRLDSDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    @classmethod
    def _get_pkg_dir_path(cls):
        return epath.Path(os.path.dirname(os.path.abspath(__file__)))

    def __init__(
        self,
        hdf5_paths,
        instruction,
        action_dim,
        state_dim,
        image_shape,
        sample_interval=1,
        gripper_quantile=0.1,
        episode_instructions=None,
        debug=False,
        **kwargs,
    ):
        self._hdf5_paths = list(hdf5_paths)
        self._instruction = instruction
        self._episode_instructions = (
            list(episode_instructions) if episode_instructions is not None else None
        )
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._image_shape = image_shape
        self._sample_interval = max(int(sample_interval), 1)
        self._gripper_quantile = gripper_quantile
        self._debug = debug
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=self._image_shape,
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(self._state_dim,),
                                        dtype=np.float32,
                                        doc="End-effector state with padding and gripper.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(self._action_dim,),
                                dtype=np.float32,
                                doc="Robot action vector.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language instruction for each step.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for episode_idx, hdf5_path in enumerate(self._hdf5_paths):
            hdf5_dir = os.path.dirname(os.path.abspath(hdf5_path))
            with h5py.File(hdf5_path, "r") as src_file:
                if self._debug:
                    print("HDF5文件结构：")
                    src_file.visititems(print_hdf5_structure)
                    print("\n开始转换...\n")

                action = load_action(
                    src_file, sample_interval=self._sample_interval, debug=self._debug
                )
                action = binarize_gripper_top10(
                    action,
                    quantile=self._gripper_quantile,
                    debug=self._debug,
                )
                state = load_state(
                    src_file,
                    action.shape[0],
                    sample_interval=self._sample_interval,
                    debug=self._debug,
                )
                state = binarize_gripper_top10(
                    state,
                    gripper_index=7,
                    quantile=self._gripper_quantile,
                    debug=self._debug,
                )
                images = load_images(
                    src_file,
                    hdf5_dir,
                    sample_interval=self._sample_interval,
                    debug=self._debug,
                )

            episode_len = min(len(images), action.shape[0], state.shape[0])
            if len(images) != action.shape[0] or action.shape[0] != state.shape[0]:
                print("警告: action/state/图片数量不一致，按最小长度对齐")
            action = action[:episode_len]
            state = state[:episode_len]
            images = images[:episode_len]

            if self._episode_instructions is not None:
                instruction = self._episode_instructions[episode_idx]
            else:
                instruction = self._instruction

            steps = []
            for i in range(episode_len):
                steps.append(
                    {
                        "observation": {"image": images[i], "state": state[i]},
                        "action": action[i].astype(np.float32),
                        "discount": 1.0,
                        "reward": float(i == (episode_len - 1)),
                        "is_first": i == 0,
                        "is_last": i == (episode_len - 1),
                        "is_terminal": i == (episode_len - 1),
                        "language_instruction": instruction,
                    }
                )

            sample = {
                "steps": steps,
                "episode_metadata": {"file_path": hdf5_path},
            }

            yield f"episode_{episode_idx:06d}", sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", default="data_converted_endpose_rlds")
    parser.add_argument("--instruction")
    parser.add_argument("--instruction_file")
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--gripper_quantile", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if bool(args.instruction) == bool(args.instruction_file):
        raise ValueError("必须且只能提供 --instruction 或 --instruction_file 之一")

    episode_instructions = None
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            episode_instructions = [line.rstrip("\n") for line in f]
        if len(episode_instructions) != len(args.hdf5_paths):
            raise ValueError("instruction_file 行数必须与 hdf5_paths 数量一致")

    with h5py.File(args.hdf5_paths[0], "r") as src_file:
        action = load_action(
            src_file, sample_interval=args.sample_interval, debug=args.debug
        )
        state = load_state(
            src_file, action.shape[0], sample_interval=args.sample_interval, debug=args.debug
        )

    action_dim = action.shape[1]
    state_dim = state.shape[1]
    image_shape = (256, 256, 3)

    builder = EndposeRLDSDataset(
        hdf5_paths=args.hdf5_paths,
        instruction=args.instruction or "",
        action_dim=action_dim,
        state_dim=state_dim,
        image_shape=image_shape,
        sample_interval=args.sample_interval,
        gripper_quantile=args.gripper_quantile,
        episode_instructions=episode_instructions,
        debug=args.debug,
        data_dir=args.output_dir,
    )

    builder.download_and_prepare()
    print(f"RLDS 数据集已保存到: {builder.data_dir}")


if __name__ == "__main__":
    main()

from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

from vertiencoder.utils.helpers import to_tensor, read_patch, read_map
from vertiencoder.utils.robot import RobotUtilities


class TvertiDatasetBase(Dataset):

    def __init__(
        self, root: str, stats: str, train: bool = True, block_size: int = 20, f_size: int = 7, height_diff: int = 0.5
    ):
        # footprint is the cropped patch underneath the robot
        # elev_map is the whole elevation map
        # self.metadata['data'][idx] -> idx is the bag index
        # within each bag, we have 6 keys:
        # data keys: dict_keys(['cmd_vel', 'elevation_map', 'footprint', 'pose', 'motor_speed', 'dt'])
        # lentgh of each key, depends on the bag length, so it varies
        # cmd_vel: (2, ) -> (v, w). ex: [0.125, 0.0880130325229122]
        # elevation_map: (str) relative whole elev_map address. ex: 'elev_map/mocap_s_122_07-18-14-14_2.npy'
        # footprint: (str) relative patch address. ex: 'footprint/mocap_s_122_07-18-14-14_0.npy'
        # pose: (6, ) -> (x, y, z, roll, pitch, yaw). ex: [3.09371, -0.0438, 0.40175, -0.00134, -0.0699, 2.9726913334990965]
        # motor_speed: (int) motor speed, multiple of 8. ex: 8x
        # dt: (float) . ex: 0.1085062026977539
        self.block_size = block_size
        self.height_diff = height_diff
        self.r_utils = RobotUtilities()
        self.root = Path(root)  # 20 bags
        with open(self.root, "rb") as f:
            metadata = pickle.load(f)

        with open(stats, "rb") as f:
            self.stats = pickle.load(f)
            print(f"Loaded stats from {self.stats['name']}!")

        self.metadata = defaultdict(list)
        num_samples = 0
        self.f_size = f_size
        for i, bag_name in enumerate(metadata["bag_name"]):
            num_samples += len(metadata["data"][i]["cmd_vel"])
            cmd_vel = np.array(metadata["data"][i]["cmd_vel"], dtype=np.float32)
            cmd_vel[:, 0] = np.convolve(
                cmd_vel[:, 0], np.ones(self.f_size) / self.f_size, mode="same"
            )
            cmd_vel[:, 1] = np.convolve(
                cmd_vel[:, 1], np.ones(self.f_size) / self.f_size, mode="same"
            )

            pose_diff = np.array(metadata["data"][i]["pose_diff"], dtype=np.float32)
            for j in range(6):
                pose_diff[:, j] = np.convolve(
                    pose_diff[:, j], np.ones(self.f_size) / self.f_size, mode="same"
                )

            self.metadata["cmd_vel"].extend(cmd_vel.tolist())
            # self.metadata["elevation_map"].extend(metadata["data"][i]["elevation_map"])
            self.metadata["footprint"].extend(metadata["data"][i]["footprint"])
            self.metadata["pose"].extend(metadata["data"][i]["pose"])
            self.metadata["motor_speed"].extend(metadata["data"][i]["motor_speed"])
            self.metadata["dt"].extend(metadata["data"][i]["dt"])
            self.metadata["pose_diff"].extend(pose_diff.tolist())
            self.metadata["time"].extend(metadata["data"][i]["time"])
                # self.r_utils.pose_difference(metadata["data"][i]["pose"])
        assert num_samples == len(
            self.metadata["cmd_vel"]
        ), "The number of samples does not match"
        self.transform = v2.Compose(
            [
                v2.Resize(size=(40, 40), antialias=True),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TvertiDatasetAE(TvertiDatasetBase):
    def __init__(self, root: str, stats: str, train: bool = True):
        super().__init__(root=root, stats=stats, train=train)
        if train:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(40, 40), antialias=True),
                    v2.ToDtype(torch.float32, scale=False),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    # v2.GaussianNoise(mean=0.0, sigma=0.001),
                    # v2.RandomRotation(degrees=(45, 275)),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(40, 40), antialias=True),
                    v2.ToDtype(torch.float32, scale=False),
                ]
            )

    def __len__(self):
        return len(self.metadata["cmd_vel"])

    def __getitem__(self, idx):
        """Return a sample in the form: (patch, )"""
        patch = self.transform(
            read_patch(self.root.parents[0] / self.metadata["footprint"][idx],
                       self.metadata["pose"][idx][2], self.height_diff)
        )
        patch = (patch - self.stats['footprint_mean']) / self.stats['footprint_std']
        return patch


class TvertiDatasetAENextToken(TvertiDatasetBase):
    def __init__(self, root: str, stats: str, train: bool = True, f_size: int = 7,  height_diff: int = 0.5):
        super().__init__(root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff)

    def __len__(self):
        return len(self.metadata["cmd_vel"]) - self.block_size - 16

    def __getitem__(self, idx):
        """Return a sample in the form: (patch, next_patch)"""
        patch = self.transform(
            read_patch(self.root.parents[0] / self.metadata["footprint"][idx],
                       self.metadata["pose"][idx][2], self.height_diff)
        )
        patch = (patch - self.stats['footprint_mean']) / self.stats['footprint_std']
        # next patch
        next_patch = self.transform(
            read_patch(self.root.parents[0] / self.metadata["footprint"][idx + self.block_size + 15],
                       self.metadata["pose"][idx + self.block_size + 15][2], self.height_diff)
        )
        next_patch = (next_patch - self.stats['footprint_mean']) / self.stats['footprint_std']
        next_cmd_vel = torch.stack(
            [
                (to_tensor(self.metadata["cmd_vel"][i]) - self.stats['cmd_vel_mean']) / self.stats['cmd_vel_std']
                for i in range(idx + self.block_size, idx + self.block_size + 15)
            ],
            dim=0,
        )
        current_cmd_vel = to_tensor(self.metadata["cmd_vel"][idx])
        return patch, next_patch, current_cmd_vel, next_cmd_vel


class TvertiDatasetMP(TvertiDatasetBase):
    def __init__(self, root: str, stats: str, train: bool = True, block_size: int = 20, f_size: int = 7, height_diff: int = 0.5):
        super().__init__(root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff)
        self.block_size = block_size

    def __len__(self):
        return len(self.metadata["pose"]) - self.block_size - 1

    def __getitem__(self, idx: int):
        # patch input to the model
        patch = torch.stack(
            [
                (self.transform(
                    read_patch(self.root.parents[0] / self.metadata["footprint"][i],
                               self.metadata["pose"][i][2], self.height_diff)
                ) - self.stats['footprint_mean']) / self.stats['footprint_std']
                for i in range(idx, idx + self.block_size)
            ],
            dim=0,
        )
        # next patch the model should predict
        # next_patch = torch.stack([self.transform(
        #     read_patch(self.root.parents[0] / self.metadata["footprint"][i],
        #                self.metadata["pose"][i][2], self.height_diff)
        # ) for i in range(idx + 1, idx + 1 + self.block_size)], dim=0)
        next_patch = self.transform(
            read_patch(self.root.parents[0] / self.metadata["footprint"][idx + self.block_size + 1],
                       self.metadata["pose"][idx + self.block_size + 1][2], self.height_diff)
        )
        next_patch = (next_patch - self.stats['footprint_mean']) / self.stats['footprint_std']
        cmd_vel = torch.stack(
            [
                (to_tensor(self.metadata["cmd_vel"][i]) - self.stats['cmd_vel_mean']) / self.stats['cmd_vel_std']
                for i in range(idx, idx + self.block_size)
            ],
            dim=0,
        )
        # next patch the model should predict
        # next_cmd_vel = torch.stack([to_tensor(self.metadata['cmd_vel'][i]) for i in range(idx + 1, idx + 1 + self.block_size)], dim=0)
        pose = torch.stack(
            [
                ((to_tensor(self.metadata["pose_diff"][i]) / self.metadata['dt'][i]) - self.stats['pose_diff_mean']) / self.stats['pose_diff_std']
                for i in range(idx, idx + self.block_size)
            ],
            dim=0,
        )

        next_pose = ((to_tensor(self.metadata["pose_diff"][idx + self.block_size + 1]) / self.metadata['dt'][idx + self.block_size + 1]) - self.stats['pose_diff_mean']) / self.stats['pose_diff_std']
        # motor_speed = [to_tensor(self.metadata['motor_speed'][i]) for i in range(idx, idx + self.block_size)]
        # dt = [to_tensor(self.metadata['dt'][i]) for i in range(idx, idx + self.block_size)]

        return (
            patch,
            next_patch,
            cmd_vel,
            pose,
            next_pose
        )


class TvertiDownStream(TvertiDatasetBase):
    def __init__(
        self, root: str, stats: str, train: bool = True, block_size: int = 20, task: str = "pose", f_size: int = 7, height_diff: int = 0.5
    ):
        super().__init__(root=root, stats=stats, train=train, f_size=f_size, height_diff=height_diff)
        self.block_size = block_size
        self.task = task

    def __len__(self):
        return len(self.metadata["pose"]) - self.block_size - 1 if self.task != 'reconstruction' else len(self.metadata["cmd_vel"]) - self.block_size - 16

    def __getitem__(self, idx: int):
        # patch input to the model
        patch = torch.stack(
            [
                (self.transform(
                    read_patch(self.root.parents[0] / self.metadata["footprint"][i],
                               self.metadata["pose"][i][2], self.height_diff)
                ) - self.stats['footprint_mean']) / self.stats['footprint_std']
                for i in range(idx, idx + self.block_size)
            ],
            dim=0,
        )
        cmd_vel = torch.stack(
            [
                (to_tensor(self.metadata["cmd_vel"][i]) - self.stats['cmd_vel_mean']) / self.stats['cmd_vel_std']
                for i in range(idx, idx + self.block_size)
            ],
            dim=0,
        )
        if self.task == "fkd":
            pose = to_tensor(
                to_tensor(self.metadata["pose_diff"][idx + self.block_size + 1]) / self.metadata["dt"][idx + self.block_size + 1]
            )  # next pose
            pose = (pose - self.stats['pose_diff_mean']) / self.stats['pose_diff_std']
            return patch, cmd_vel, pose
        elif self.task == "bc":
            next_cmd = (to_tensor(self.metadata["cmd_vel"][idx + self.block_size + 1]) - self.stats['cmd_vel_mean']) / self.stats['cmd_vel_std']
            return (
                patch,
                cmd_vel,
                next_cmd,
            )  # next cmd_vel
        elif self.task == "ikd":
            next_cmd = to_tensor(self.metadata["cmd_vel"][idx + self.block_size + 1])
            next_cmd = (next_cmd - self.stats['cmd_vel_mean']) / self.stats['cmd_vel_std']
            next_pose = to_tensor(self.metadata["pose_diff"][idx + self.block_size + 1])
            next_pose = (next_pose - self.stats['pose_diff_mean']) / self.stats['pose_diff_std']
            return patch, cmd_vel, (next_cmd, next_pose)
        elif self.task == "reconstruction":
            next_patch = self.transform(
                read_patch(self.root.parents[0] / self.metadata["footprint"][idx + self.block_size + 15],
                           self.metadata["pose"][idx + self.block_size + 15][2], self.height_diff)
            )
            next_patch = (next_patch - self.stats['footprint_mean']) / self.stats['footprint_std']
            next_cmd_vel = torch.stack(
                [
                    (to_tensor(self.metadata["cmd_vel"][i]) - self.stats['cmd_vel_mean']) / self.stats['cmd_vel_std']
                    for i in range(idx + self.block_size, idx + self.block_size + 15)
                ],
                dim=0,
            )
            return patch, cmd_vel, (next_patch, next_cmd_vel)


if __name__ == "__main__":
    pass

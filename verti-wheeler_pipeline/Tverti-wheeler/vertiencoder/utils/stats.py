from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import v2

from helpers import to_tensor, read_patch, read_map
from robot import RobotUtilities


def calculate_stats(root: str, f_size: int = 7, height_diff: int = 0.5):
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
    root = Path(root)
    with open(root, "rb") as f:
        metadata = pickle.load(f)

    data = defaultdict(list)
    transform = v2.Compose(
        [
            v2.Resize(size=(40, 40), antialias=True),
            v2.ToDtype(torch.float32, scale=False),
            # v2.Normalize(mean=self.stats["footprint_mean"].squeeze(0), std=self.stats["footprint_std"].squeeze(0)),
        ]
    )
    num_samples = 0
    for i, bag_name in enumerate(metadata["bag_name"]):
        num_samples += len(metadata["data"][i]["cmd_vel"])

        data["cmd_vel"].extend(to_tensor(metadata["data"][i]["cmd_vel"]))
        # self.metadata["elevation_map"].extend(metadata["data"][i]["elevation_map"])
        data["footprint"].extend(metadata["data"][i]["footprint"])
        data["pose"].extend(to_tensor(metadata["data"][i]["pose"]))
        data["motor_speed"].extend(to_tensor(metadata["data"][i]["motor_speed"]))
        data["dt"].extend(to_tensor(metadata["data"][i]["dt"]))
        # pose_diff = np.array(metadata["data"][i]["pose_diff"], dtype=np.float32)
        # for j in range(6):
        #     pose_diff[:, j] = np.convolve(
        #         pose_diff[:, j], np.ones(f_size) / f_size, mode="same"
        #     )
        # print(f"{to_tensor(metadata['data'][i]['pose_diff']).shape = }")
        data["pose_diff"].extend(to_tensor(metadata["data"][i]["pose_diff"]) / to_tensor(metadata["data"][i]["dt"]).unsqueeze(dim=1))
        data["time"].extend(to_tensor(metadata["data"][i]["time"]))

    data["cmd_vel"] = torch.stack(data["cmd_vel"], dim=0)
    # data["footprint"] = torch.stack(data["footprint"], dim=0)
    data["footprint"] = torch.stack([transform(read_patch(root.parents[0] / footprint, pose[2]))
                                     for footprint, pose in zip(data["footprint"], data['pose'])], dim=0)
    data["pose"] = torch.stack(data["pose"], dim=0)
    data["motor_speed"] = torch.stack(data["motor_speed"], dim=0)
    data["dt"] = torch.stack(data["dt"], dim=0)
    data["pose_diff"] = torch.stack(data["pose_diff"], dim=0)
    data["time"] = torch.stack(data["time"], dim=0)
    # print(f"{num_samples = }")
    # print(f"{data['cmd_vel'].shape = }")
    # print(f"{data['footprint'].shape = }")
    # print(f"{data['pose'].shape = }")
    # print(f"{data['motor_speed'].shape = }")
    # print(f"{data['dt'].shape = }")
    # print(f"{data['pose_diff'].shape = }")
    # print(f"{data['time'].shape = }")
    stats = {'name': root.parent.name}
    for key, value in data.items():
        # print(f"{key = }")
        if key in ['footprint', 'elevation_map']:
            stats[key + '_mean'] = torch.mean(value)
            stats[key + '_var'] = torch.var(value)
            stats[key + '_std'] = torch.std(value)
            stats[key + '_max'] = torch.max(value)
            stats[key + '_min'] = torch.min(value)
        else:
            stats[key + '_mean'] = torch.mean(value, dim=0)
            stats[key + '_var'] = torch.var(value, dim=0)
            stats[key + '_std'] = torch.std(value, dim=0)
            stats[key + '_max'] = torch.max(value, dim=0).values
            stats[key + '_min'] = torch.min(value, dim=0).values

    for key, value in stats.items():
        print(f"{key}: {value}")

    # print(f"{stats['footprint_mean'].shape = }, {stats['footprint_var'].shape = }, {stats['footprint_std'].shape = }")
    # print(f"{stats['footprint_mean'] = }, {stats['footprint_var'] = }, {stats['footprint_std'] = }")
    # print(f"{stats['footprint_max'] = }, {stats['footprint_min'] = }")
    #
    # print(f"{stats['cmd_vel_mean'].shape = }, {stats['cmd_vel_var'].shape = }, {stats['cmd_vel_std'].shape = }")
    # print(f"{stats['cmd_vel_mean'] = }, {stats['cmd_vel_var'] = }, {stats['cmd_vel_std'] = }")
    #
    # print(f"{stats['pose_mean'].shape = }, {stats['pose_std'].shape = }, {stats['pose_std'].shape = }")
    # print(f"{stats['pose_mean'] = }, {stats['pose_var'] = }, {stats['pose_std'] = }")
    #
    # print(f"{stats['motor_speed_mean'].shape = }, {stats['motor_speed_var'].shape = }, {stats['motor_speed_std'].shape = }")
    # print(f"{stats['motor_speed_mean'] = }, {stats['motor_speed_var'] = }, {stats['motor_speed_std'] = }")
    #
    # print(f"{stats['dt_mean'].shape = }, {stats['dt_var'].shape = }, {stats['dt_std'].shape = }")
    # print(f"{stats['dt_mean'] = }, {stats['dt_var'] = }, {stats['dt_std'] = }")
    #
    # print(f"{stats['pose_diff_mean'].shape = }, {stats['pose_diff_var'].shape = }, {stats['pose_diff_std'].shape = }")
    # print(f"{stats['pose_diff_mean'] = }, {stats['pose_diff_var'] = }, {stats['pose_diff_std'] = }")
    #
    # print(f"{stats['time_mean'].shape = }, {stats['time_var'].shape = }, {stats['time_std'].shape = }")
    # print(f"{stats['time_mean'] = }, {stats['time_var'] = }, {stats['time_std'] = }")

    with open(root.parents[0] / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    calculate_stats("vertiencoder/data/train/data_train_filtered.pickle")

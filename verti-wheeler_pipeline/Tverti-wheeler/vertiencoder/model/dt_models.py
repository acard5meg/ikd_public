# ATC EDITS
import sys
import os
# cur_directory = os.path.abspath('./Tverti_wheeler/vertiencoder/utils')
# current = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# utils_dir = os.path.join(current,"utils")
# sys.path.append()
# print(cur_directory)
# print(utils_dir)
# sys.path.append(utils_dir)

import torch
import torch.nn as nn

# ATC EDITS
# from vertiencoder.utils.nn import make_mlp
# from vertiencoder.model.tverti import load_model

# in the utils directory which is in a totally different directory 

# from utils.nn import make_mlp
# ATC EDITS
# from verti_wheeler_pipeline.Tverti_wheeler.vertiencoder.utils.nn import make_mlp
from nn import make_mlp
# from verti_wheeler_pipeline.Tverti_wheeler.vertiencoder.model.tverti import load_model
from tverti import load_model


class FKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.pose_encoder = nn.Sequential(nn.Linear(6, cfg.dims[0]), nn.LeakyReLU(),)
        # cfg.pose.dims.insert(0, 6)
        # cfg.pose.dims.append(cfg.fc.dims[0] // 4)
        # self.pose_encoder = make_mlp(**cfg.pose)
        # cfg.pose.dims[0] = 2
        # self.cmd_encoder = make_mlp(**cfg.pose)
        # cfg.fc.dims[0] = cfg.fc.dims[0] + (cfg.fc.dims[0] // 2)
        # cfg.dims.insert(0, cfg.dims[0] * 3)
        cfg.fc.dims.append(6)  # append the task output to the end
        self.fc = make_mlp(**cfg.fc)

    def forward(self, x: torch.Tensor, cmd: torch.Tensor, pose: torch.Tensor = None) -> torch.Tensor:
        # pose = self.pose_encoder(pose)
        # cmd = self.cmd_encoder(cmd)
        # x = torch.cat((x, pose, cmd), dim=-1)
        return self.fc(x)


class BehaviorCloning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.dims.append(2)  # append the task output to the end
        self.fc = make_mlp(**cfg)

    def forward(self, x: torch.Tensor, g_pose: torch.Tensor = None) -> torch.Tensor:
        return self.fc(x)


class IKD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = make_mlp(**cfg.pose)
        cfg.pose.dims[0] = 6
        self.goal_pose = make_mlp(**cfg.pose)
        self.curr_pose = nn.Sequential(nn.Linear(6, cfg.pose.dims[-1]), nn.LeakyReLU(), )
        cfg.fc.dims.insert(0, 3 * cfg.pose.dims[-1])
        cfg.fc.dims.append(2)  # append the task output to the end
        self.fc = make_mlp(**cfg.fc)

    def forward(self, z: torch.Tensor, g_pose: torch.Tensor = None, curr_pose: torch.Tensor = None) -> torch.Tensor:
        z = self.encoder(z)
        g_pose = self.goal_pose(g_pose)
        curr_pose = self.curr_pose(curr_pose)
        pose = torch.cat([z, g_pose, curr_pose], dim=-1)
        return self.fc(pose)


class DeployedModel(nn.Module):
    def __init__(self, encoder: nn.Module, downstream: nn.Module, task: str = 'fkd'):
        super().__init__()
        self.encoder = encoder
        self.downstream = downstream
        self.task = task

    @torch.inference_mode()
    def forward(self, patches: torch.Tensor, actions: torch.Tensor, curr_pose: torch.Tensor = None, g_pose: torch.Tensor = None) -> torch.Tensor:
        ctx = self.encoder(patches, actions, curr_pose)
        if self.task == 'fkd':
            actions_in = actions[:, -1]
            curr_pose_in = curr_pose[:, -1]
            # ctx = torch.zeros_like(ctx)
            # actions_in = torch.randn_like(actions_in)
            # curr_pose_in = torch.randn_like(curr_pose_in)
            return self.downstream(ctx, actions_in, curr_pose_in)
        elif self.task == 'ikd':
            return self.downstream(ctx, g_pose, curr_pose)
        else:
            return self.downstream(ctx, g_pose)


def get_model(cfg):
    tverti, _ = load_model(cfg.tverti)
    tverti_weight = torch.load(cfg.tverti_weight, map_location=torch.device('cpu'))['pretext_model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(tverti_weight.items()):
        if k.startswith(unwanted_prefix):
            tverti_weight[k[len(unwanted_prefix):]] = tverti_weight.pop(k)

    tverti.load_state_dict(tverti_weight)
    tverti = tverti.eval()
    tverti.requires_grad_(False)
    if cfg.task == 'bc':
        cfg.bc.dims.insert(0, cfg.tverti.transformer_layer.d_model)
        bc = BehaviorCloning(cfg.bc)
        bc_weight = torch.load(cfg.bc_weight, map_location=torch.device('cpu'))['dt_model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(bc_weight.items()):
            if k.startswith(unwanted_prefix):
                bc_weight[k[len(unwanted_prefix):]] = bc_weight.pop(k)

        bc.load_state_dict(bc_weight)
        bc = bc.eval()
        bc.requires_grad_(False)
        model = DeployedModel(tverti, bc, cfg.task)
    elif cfg.task == 'fkd':
        cfg.fkd.fc.dims.insert(0, cfg.tverti.transformer_layer.d_model)
        fkd = FKD(cfg.fkd)
        fkd_weight = torch.load(cfg.fkd_weight, map_location=torch.device('cpu'))['dt_model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(fkd_weight.items()):
            if k.startswith(unwanted_prefix):
                fkd_weight[k[len(unwanted_prefix):]] = fkd_weight.pop(k)
        fkd.load_state_dict(fkd_weight)
        fkd = fkd.eval()
        fkd.requires_grad_(False)
        model = DeployedModel(tverti, fkd, cfg.task)
    elif cfg.task == 'ikd':
        cfg.ikd.dims.insert(0, cfg.tverti.transformer_layer.d_model)
        ikd = IKD(cfg.ikd)
        ikd_weight = torch.load(cfg.ikd_weight, map_location=torch.device('cpu'))['dt_model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(ikd_weight.items()):
            if k.startswith(unwanted_prefix):
                ikd_weight[k[len(unwanted_prefix):]] = ikd_weight.pop(k)

        ikd.load_state_dict(ikd_weight)
        ikd = ikd.eval()
        ikd.requires_grad_(False)
        model = DeployedModel(tverti, ikd, cfg.task)
    else:
        raise NotImplementedError(f"Unknown task {cfg.task}")

    if cfg.compile:
        model = torch.compile(model)
    return model
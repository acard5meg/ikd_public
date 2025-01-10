import gc
from pathlib import Path
from datetime import datetime
import sys
import argparse
from functools import partial

try:
    sys.path.append(str(Path(".").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project to the path")

from comet_ml.integration.pytorch import log_model, watch
from omegaconf import OmegaConf, ListConfig
from rich import print
import numpy as np
from tqdm import tqdm
from icecream import ic, install
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torcheval.metrics import PeakSignalNoiseRatio
import matplotlib
import matplotlib.pyplot as plt

from model.tverti import Tverti, load_model
from model.swae import SWAutoencoder, Encoder, Decoder
from model.dataloader import TvertiDatasetMP
from utils.nn import check_grad_norm, init_weights, op_counter, init_optimizer
from utils.io import save_checkpoint, load_checkpoint
from utils.helpers import (
    get_conf,
    timeit,
    init_logger,
    init_device,
    to_tensor,
    fix_seed,
)
from utils.loss import sliced_wasserstein_distance


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file
        self.cfg = get_conf(cfg_dir)
        # set the name for the model
        self.cfg.directory.model_name = (
            f"{self.cfg.logger.experiment_name}-{self.cfg.model.swae.latent_dim}D"
        )
        self.cfg.directory.model_name += f"-{datetime.now():%m-%d-%H-%M}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        # if debugging True, set a few rules
        if self.cfg.train_params.debug:
            install()
            ic.enable()
            ic.configureOutput(prefix=lambda: f"{datetime.now():%H:%M:%S} |> ")
            torch.autograd.set_detect_anomaly(True)
            self.cfg.logger.disabled = True
            matplotlib.use("TkAgg")
        else:
            ic.disable()
            torch.autograd.set_detect_anomaly(True)
        # initialize the logger and the device
        self.logger = init_logger(self.cfg)
        self.device = init_device(self.cfg)
        # torch.set_default_device(self.device)
        ic(torch.get_default_device())
        # fix the seed for reproducibility
        fix_seed(self.cfg.train_params.seed)
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")
        self.model, self.swae = load_model(self.cfg.model)
        self.model = self.model.to(self.device)
        self.swae = self.swae.to(self.device)
        if self.cfg.train_params.compile:
            self.model = torch.compile(self.model)
        # log the model gradients, weights, and activations in comet
        watch(self.model)
        self.logger.log_code(folder="./vertiencoder/model")
        # initialize the optimizer
        self.optimizer, self.scheduler = init_optimizer(
            self.cfg, self.model.parameters(), self.cfg.train_params.optimizer
        )
        num_params = [x.numel() for x in self.model.parameters()]
        trainable_params = [
            x.numel() for x in self.model.parameters() if x.requires_grad
        ]
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of parameters: {sum(num_params) / 1e6:.2f}M"
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of trainable parameters: {sum(trainable_params) / 1e6:.2f}M"
        )
        # define loss function
        self.criterion = torch.nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio()
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
            )
            for data in bar:
                self.iteration += 1
                (loss_data), t_train = self.forward_batch(data)
                t_train /= self.data.batch_size
                running_loss.append(loss_data["loss"])

                bar.set_postfix(
                    loss=loss_data["loss"],
                    Grad_Norm=loss_data["grad_norm"],
                    Time=t_train,
                )

                self.logger.log_metrics(
                    {
                        "batch_loss": loss_data["loss"],
                        "grad_norm": loss_data["grad_norm"],
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )
                self.logger.log_image(
                    loss_data["samples"],
                    f"train_E{self.epoch}",
                    step=self.iteration,
                    image_channels="first",
                )

            bar.close()
            self.scheduler.step()

            # validate on val set
            (val_loss), t = self.validate()
            t /= len(self.val_data.dataset)

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: "
                + f"[green]{self.e_loss[-1]:.2f}[/green] \t| Val loss: [red]{val_loss:.2f}[/red] \t| "
                +
                # f"Patch loss: [red]{loss_patch:.2f}[/red] \t| Action loss: [red]{loss_action:.2f}[/red] \t| " +
                f"PSNR: [red]{self.psnr.compute().item():.2f}[/red] "
                + f"\t| time: {t:.3f} seconds\n"
            )

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    # "val_loss_patch": loss_patch,
                    # "val_loss_action": loss_action,
                    "PSNR": self.psnr.compute().item(),
                    "time": t,
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.psnr.compute().item() > self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()

            gc.collect()
            self.epoch += 1

        # elev_map, patch, next_patch, cmd_vel, next_cmd_vel, pose = next(iter(self.data))
        # patch = torch.randn(
        #     size=(256, 20, 1, 40, 40), dtype=torch.float32, device=self.device
        # )
        # actions = torch.randn(
        #     size=(256, 20, 2), dtype=torch.float32, device=self.device
        # )
        # macs, params = op_counter(self.model, sample=(patch, actions))
        # print("macs = ", macs, " | params = ", params)
        # self.logger.log_metrics(
        #     {"GFLOPS": float(macs[:-1]), "#Params": float(params[:-1])}
        # )
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        # move data to device
        patch, next_patch, cmd_vel, pose, next_pose = data
        ic(patch.shape)
        ic(next_patch.shape)
        ic(cmd_vel.shape)
        # ic(next_cmd_vel.shape)
        ic(pose.shape)
        # ic((patch[0][1:] == next_patch[0][:1]).all())
        # elevation_map -> (B, 1, 360, 360)
        # patch -> (B, 1, 40, 100)
        patch = patch.to(self.device)
        next_patch = next_patch.to(self.device)
        cmd_vel = cmd_vel.to(self.device)
        next_pose = next_pose.to(self.device)
        # next_cmd_vel = next_cmd_vel.to(self.device)
        ##### debugging
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # grid = make_grid(next_patch[55], nrow=4, padding=2, value_range=(-1, 1), normalize=True, scale_each=True)
        # # plt.imshow(grid.cpu().permute(1, 2, 0))
        # ax[0].imshow(grid.cpu().permute(1, 2, 0))
        # # plt.show()
        # next_patch_embed = torch.stack([self.model.patch_encoder(next_patch[:, i, :])
        #                                 for i in range(next_patch.shape[1])], dim=1)  # (B, T, L)
        # next_patch_embed_re = torch.stack([self.swae(next_patch_embed[:, i, :])
        #                                 for i in range(next_patch.shape[1])], dim=1)  # (B, T, L)
        # grid = make_grid(next_patch_embed_re[55], nrow=4, padding=2, value_range=(-1, 1), normalize=True, scale_each=True)
        #
        #
        # ax[1].imshow(grid.cpu().permute(1, 2, 0))
        # plt.show()
        # plt.imshow((next_patch[55, 0].cpu().squeeze() + 1) / 2., cmap='gray')
        # plt.show()
        # temp = self.model.patch_encoder(next_patch[55, ...])
        # ic(temp.shape)
        # temp = self.swae(temp)
        # plt.imshow(temp[0].cpu().squeeze(), cmap='gray')
        # plt.show()
        # plt.imshow(self.swae(torch.randn(size=(1, 128), device=self.device)).cpu().squeeze(), cmap='gray')
        # plt.show()
        # # plt.plot(cmd_vel[0, :, 0].cpu(), cmd_vel[0, :, 1].cpu())
        # # plt.show()
        # sys.exit(0)
        ##### end debugging
        # embed the next patch
        next_patch_embed = self.swae.encode(next_patch)
        # next_patch_embed = torch.stack([self.swae.encode(next_patch[:, i, :])
        #                                 for i in range(next_patch.shape[1])], dim=1)  # (B, T, L)
        # forward, backward
        pred_patch, pred_pose = self.model(patch, cmd_vel)
        ic(pred_patch.shape)
        loss = self.criterion(pred_patch, next_patch_embed) + self.criterion(pred_pose, next_pose)
        # loss_actions = self.criterion(pred_action, next_cmd_vel)
        self.optimizer.zero_grad()
        # loss  = loss_patches + loss_actions
        loss.backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        self.optimizer.step()
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)
        recon_samples_hat = self.swae.decode(pred_patch[:64]).cpu()
        # recon_samples = torch.stack([self.swae.decode(next_patch[:64, i, :].to(self.device, non_blocking=True)).cpu()
        #                                  for i in range(next_patch.shape[1])], dim=1)  # (B, T, L)
        log_samples = make_grid(
            [
                make_grid(
                    [gt, pred],
                    nrow=2,
                    value_range=(-1, 1),
                    normalize=True,
                    scale_each=True,
                )
                for gt, pred in zip(next_patch[:64].cpu(), recon_samples_hat)
            ],
            nrow=8,
        )

        return {
            "loss": loss.detach().item(),
            "grad_norm": grad_norm,
            "samples": log_samples,
        }

    @timeit
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = []
        # running_loss_patch = []
        # running_loss_action = []
        self.psnr.reset()
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
        )
        for data in bar:
            # move data to device
            patch, next_patch, cmd_vel, pose, next_pose = data
            patch = patch.to(self.device)
            next_patch = next_patch.to(self.device)
            cmd_vel = cmd_vel.to(self.device)
            next_pose = next_pose.to(self.device)
            # next_cmd_vel = next_cmd_vel.to(self.device)
            # embed the next patch
            next_patch_embed = self.swae.encode(next_patch)
            # next_patch_embed = torch.stack([self.swae.encode(next_patch[:, i, :])
            #                    for i in range(next_patch.shape[1])], dim=1)  # (B, T, L)
            # forward, backward
            pred_patch, pred_pose = self.model(patch, cmd_vel)

            loss = self.criterion(
                pred_patch, next_patch_embed
            )  + self.criterion(pred_pose, next_pose) # compare only the last patch
            # loss_action = self.criterion(pred_action, next_cmd_vel[:, [-1], :])  # compare only the last action
            # loss = loss_patch + loss_action
            recon_samples_hat = self.swae.decode(pred_patch)
            self.psnr.update(recon_samples_hat, next_patch)

            log_patch_samples = make_grid(
                [
                    make_grid(
                        [gt, pred],
                        nrow=2,
                        value_range=(-1, 1),
                        normalize=True,
                        scale_each=True,
                    )
                    for gt, pred in zip(
                        next_patch[:64].cpu(), recon_samples_hat[:64].cpu()
                    )
                ],
                nrow=8,
            )

            # log_action_samples = make_grid([make_grid([gt, pred], nrow=2, value_range=(-1, 1), normalize=True, scale_each=True)
            #                          for gt, pred in zip(next_cmd_vel[:64, -1].cpu(),
            #                                              pred_action[:64].cpu())], nrow=8)

            running_loss.append(loss.item())
            # running_loss_patch.append(loss_patch.item())
            # running_loss_action.append(loss_action.item())
            bar.set_postfix(loss=loss.item(), PSNR=self.psnr.compute().item())
            self.logger.log_image(
                log_patch_samples,
                f"val_E{self.epoch}",
                step=self.iteration,
                image_channels="first",
            )
        bar.close()
        # average loss
        loss = np.mean(running_loss)
        # loss_patch = np.mean(running_loss_patch)
        # loss_action = np.mean(running_loss_action)

        return loss  # , loss_patch, loss_action

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        # defining the dataset interface
        dataset = TvertiDatasetMP(**self.cfg.dataset)
        self.cfg.dataset.update(self.cfg.val_dataset)
        val_dataset = TvertiDatasetMP(**self.cfg.dataset)
        # creating dataloader
        data = DataLoader(dataset, **self.cfg.dataloader)

        self.cfg.dataloader.update({"shuffle": False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        # log dataset status
        self.logger.log_parameters(
            {"train_len": len(dataset), "val_len": len(val_dataset)}
        )
        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples."
        )

        return data, val_data

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = -np.inf
            self.e_loss = []

        self.logger.set_epoch(self.epoch)

    def save(self, name=None):
        model = self.model
        if isinstance(self.model, torch.nn.DataParallel):
            model = model.module

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "lr_scheduler": self.scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}-E{self.epoch}"
        else:
            save_name = name

        if self.psnr.compute().item() > self.best:
            self.best = self.psnr.compute().item()
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/transformer", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()

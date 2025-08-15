import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from roma import roma
from torch import nn
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.datasets import BaseDataset
from src.entities.gaussian_model import GaussianModel, MiniGaussianModel
from src.entities.losses import l1_loss
from src.entities.mw2loss import SinkhornDistance, simplify_gmm_vectorized
from src.utils.align_utils import (compute_camera_opt_params,
                                   multiply_quaternions, render_mini_gaussian)
from src.utils.gaussian_model_utils import build_rotation
from src.utils.utils import get_render_settings, render_gaussian_model


class GaussianAligner(object):

    def __init__(self, config: dict,
                 dataset: BaseDataset,
                 device: torch.device,
                 nopo_encoder=None,
                 nopo_decoder=None,
                 nopo_losses=None,) -> None:
        self.dataset = dataset
        self.config = config
        self.filter_alpha = self.config["filter_alpha"]
        self.alpha_thre = self.config["alpha_thre"]
        self.soft_alpha = self.config["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["mask_invalid_depth"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(
            description="Training script parameters"))
        self.device = device
        self.nopo_encoder = nopo_encoder
        self.nopo_decoder = nopo_decoder
        self.nopo_losses = nopo_losses

    def compute_align_losses(self, total_iter: int,
                             iter: int,
                             frame_id_i: int,
                             model: GaussianModel,
                             mini: MiniGaussianModel,
                             w2c_i: torch.Tensor,
                             opt_gs_scale: torch.Tensor,
                             opt_cam_rot: torch.Tensor,
                             opt_cam_trans: torch.Tensor,
                             gt_color_i: torch.Tensor,
                             main_component: tuple,
                             mini_component: tuple,
                             mw2_loss_flag: bool = True) -> tuple:
        width = self.dataset.width
        height = self.dataset.height
        intrinsics = self.dataset.intrinsics
        device = self.device

        # 计算相对位移
        rel_R = build_rotation(F.normalize(opt_cam_rot[None])).squeeze(0)
        rel_T = opt_cam_trans

        main_xyz = model.get_xyz()
        main_xyz = (rel_R @ main_xyz.transpose(1, 0)).transpose(1, 0) + rel_T

        main_rotation = model.get_rotation()
        quat = F.normalize(opt_cam_rot[None])
        main_rotation = multiply_quaternions(
            main_rotation, quat.unsqueeze(0)).squeeze(0)

        # 渲染全局高斯
        render_settings = get_render_settings(
            width, height, intrinsics, w2c_i)
        render_dict = render_gaussian_model(model, render_settings,
                                            override_means_3d=main_xyz,
                                            override_rotations=main_rotation)

        main_depth_i = render_dict["depth"]
        main_color_i = render_dict["color"]
        main_alpha_i = render_dict["alpha"]

        scaled_mini = mini.scale_gaussians(opt_gs_scale)

        # 渲染nopo子图的高斯
        render_settings = get_render_settings(
            width, height, intrinsics, torch.eye(4, dtype=torch.float32, device=device))
        render_dict = render_mini_gaussian(scaled_mini, render_settings)

        mini_depth_i = render_dict["depth"]
        mini_color_i = render_dict["color"]

        # 深度损失项目
        depth_mask = (main_depth_i > 0.5) & (mini_depth_i > 0.5)
        depth_loss = l1_loss(main_depth_i, mini_depth_i, agg="none")
        depth_loss = depth_loss * depth_mask
        depth_loss = depth_loss.sum()

        alpha_mask = main_alpha_i > self.alpha_thre

        tracking_mask = torch.ones_like(alpha_mask).bool()

        if self.filter_alpha:
            tracking_mask &= alpha_mask

        color_loss = l1_loss(main_color_i, gt_color_i, agg="none")

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask

        color_loss = color_loss.sum()

        if iter + 1 == total_iter:
            self.vis_align_result(
                frame_id_i, iter+1, gt_color_i, main_color_i, main_depth_i, mini_color_i, mini_depth_i)

        if mw2_loss_flag:
            mw2loss = self.compute_mw2_loss(
                main_component, mini_component, w2c_i, opt_gs_scale, opt_cam_rot, opt_cam_trans)
        else:
            mw2loss = torch.tensor(0.0).to(device)

        return color_loss, depth_loss, mw2loss

    @torch.no_grad()
    def vis_align_result(self, frame_id: int,
                         iter: int,
                         gt_color: torch.Tensor,
                         main_color: torch.Tensor,
                         main_depth: torch.Tensor,
                         mini_color: torch.Tensor,
                         mini_depth: torch.Tensor):

        output_dir = Path(self.config["output_path"])
        output_dir = output_dir / "vis_align"
        output_dir.mkdir(exist_ok=True, parents=True)

        n_rows = 1
        n_cols = 5
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        gt_color = gt_color.clamp(0, 1).permute(
            1, 2, 0).detach().cpu().numpy()
        main_color = main_color.clamp(0, 1).permute(
            1, 2, 0).detach().cpu().numpy()
        main_depth = main_depth[0].detach().cpu().numpy()
        mini_color = mini_color.clamp(0, 1).permute(
            1, 2, 0).detach().cpu().numpy()
        mini_depth = mini_depth[0].detach().cpu().numpy()

        axs[0].imshow(gt_color)
        axs[0].set_title("GT Color", fontsize=16)
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(main_color)
        axs[1].set_title("Main Color", fontsize=16)
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        axs[2].imshow(main_depth, cmap="jet", vmin=0, vmax=6)
        axs[2].set_title("Main Depth", fontsize=16)
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        axs[3].imshow(mini_color)
        axs[3].set_title("Mini Color", fontsize=16)
        axs[3].set_xticks([])
        axs[3].set_yticks([])

        axs[4].imshow(mini_depth, cmap="jet", vmin=0, vmax=6)
        axs[4].set_title("Mini Depth", fontsize=16)
        axs[4].set_xticks([])
        axs[4].set_yticks([])

        fig_title = f"Frame {frame_id:04d} Iter {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)

        plt.subplots_adjust(top=0.90)
        fig.tight_layout()
        plt.savefig(output_dir / f"{frame_id:04d}.png",
                    dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()

    def compute_mw2_loss(self, main_component: tuple, mini_component: tuple,
                         w2c_i: torch.Tensor,
                         opt_scale: torch.Tensor,
                         opt_cam_rot: torch.Tensor,
                         opt_cam_trans: torch.Tensor) -> torch.Tensor:

        w2c_r = w2c_i[:3, :3]
        w2c_t = w2c_i[:3, 3]

        rel_R = build_rotation(F.normalize(opt_cam_rot[None])).squeeze(0)
        rel_T = opt_cam_trans

        total_R = w2c_r @ rel_R
        total_T = rel_T @ w2c_r.T + w2c_t

        mini_mu, mini_cov, mini_weight = mini_component

        scaled_mini_mu = mini_mu * opt_scale
        scaled_mini_cov = mini_cov * (opt_scale ** 2)

        main_mu, main_cov, main_weight = main_component
        transformed_main_mu = (main_mu @ total_R.T) + total_T
        transformed_main_cov = torch.einsum(
            'ij,bjk,lk->bil', total_R, main_cov, total_R)

        sinkhorn = SinkhornDistance(epsilon=0.1)
        loss = sinkhorn(
            scaled_mini_mu, scaled_mini_cov, mini_weight,
            transformed_main_mu, transformed_main_cov, main_weight
        )
        return loss

    @torch.no_grad()
    def get_main_gmm_component(self, model: GaussianModel, num_cluster: int):
        mu = model.get_xyz()
        weight = torch.softmax(model.get_opacity().squeeze(), dim=0)
        cov = model.get_full_covariance()
        component = (simplify_gmm_vectorized(mu, cov, weight, num_cluster))
        return component

    @torch.no_grad()
    def get_mini_gmm_component(self, model: MiniGaussianModel, num_cluster: int):
        mu = model.xyz
        weight = torch.softmax(torch.sigmoid(model.opacities).squeeze(), dim=0)
        scales = torch.exp(model.scales)
        rotations = torch.nn.functional.normalize(model.rotations)
        rotations = roma.quat_wxyz_to_xyzw(rotations)
        R = roma.unitquat_to_rotmat(rotations)
        S = scales.diag_embed()
        L = R @ S
        cov = L @ L.transpose(1, 2)
        component = (simplify_gmm_vectorized(mu, cov, weight, num_cluster))
        return component

    @torch.no_grad()
    def compute_scale_params(self, model: GaussianModel,
                             mini: MiniGaussianModel,
                             c2w_i: torch.Tensor) -> nn.Parameter:

        width = self.dataset.width
        height = self.dataset.height
        intrinsics = self.dataset.intrinsics
        w2c_i = torch.inverse(c2w_i)

        render_settings = get_render_settings(
            width, height, intrinsics, w2c_i)
        render_dict = render_gaussian_model(model, render_settings)

        pseudo_depth_i = render_dict["depth"]

        render_settings = get_render_settings(
            width, height, intrinsics, np.eye(4))
        render_dict = render_mini_gaussian(mini, render_settings)

        mini_depth_i = render_dict["depth"]

        init_scale = torch.median(pseudo_depth_i) / torch.median(mini_depth_i)
        return nn.Parameter(init_scale.clone().requires_grad_(True))

    @torch.no_grad()
    def print_align_message(self, frame_id_i,
                            frame_id_j,
                            min_loss: torch.Tensor,
                            c2w_i: torch.Tensor,
                            gs_scale: torch.Tensor,
                            rel_c2w_j: torch.Tensor,
                            normalize_scale: torch.Tensor) -> None:
        device = self.device
        gt_c2w_i = self.dataset[frame_id_i][3]
        gt_c2w_j = self.dataset[frame_id_j][3]
        gt_c2w_i = torch.tensor(gt_c2w_i, dtype=torch.float32, device=device)
        gt_c2w_j = torch.tensor(gt_c2w_j, dtype=torch.float32, device=device)
        gt_rel_c2w_j = torch.inverse(gt_c2w_i) @ gt_c2w_j

        abs_scale = torch.norm(
            rel_c2w_j[0:3, 3]) / torch.norm(gt_rel_c2w_j[0:3, 3])

        gt_rel_quat = roma.rotmat_to_unitquat(gt_rel_c2w_j[0:3, 0:3])
        rel_quat = roma.rotmat_to_unitquat(rel_c2w_j[0:3, 0:3])
        rel_quat_error = torch.sum(torch.abs(rel_quat - gt_rel_quat)).mean()

        quat_i = roma.rotmat_to_unitquat(c2w_i[0:3, 0:3])
        gt_quat_i = roma.rotmat_to_unitquat(gt_c2w_i[0:3, 0:3])
        quat_error_i = torch.sum(torch.abs(quat_i - gt_quat_i)).mean()
        trans_err_i = torch.abs(c2w_i[0:3, 3] - gt_c2w_i[0:3, 3]).mean()

        final_scale = normalize_scale * gs_scale
        print()
        print(f"Min Loss: {min_loss:.2f}")
        print(f"Absolute Scale: {abs_scale.item():.5f}")
        print(f"Relative Scale: {gs_scale.item():.5f}")
        print(f"Relative Quat Error: {rel_quat_error.item():.5f}")
        print(f"Quat Error: {quat_error_i.item():.5f}")
        print(f"Trans Error: {trans_err_i.item():.5f}")
        print(f"Normalize Scale: {normalize_scale.item():.5f}")
        print(f"Final Scale: {final_scale.item():.5f}")
        print()

    def align_sub_gaussians(self, frame_id_i: int,
                            frame_id_j: int,
                            model: GaussianModel,
                            mini: MiniGaussianModel,
                            init_c2w_i: torch.Tensor,
                            rel_c2w_j: torch.Tensor) -> tuple[MiniGaussianModel, torch.Tensor, torch.Tensor]:
        """
        estimated_c2w_i [4 4]
        rel_c2w_j [4 4]
        """
        assert frame_id_i != 0
        device = self.device

        init_w2c_i = torch.inverse(init_c2w_i)
        init_rel_c2w = torch.eye(4, dtype=torch.float32)
        init_rel_w2c = torch.inverse(init_rel_c2w)

        reference_w2c = init_w2c_i
        opt_cam_rot, opt_cam_trans = compute_camera_opt_params(
            init_rel_w2c.detach().cpu().numpy())
        opt_gs_scale = self.compute_scale_params(model, mini, init_c2w_i)
        if torch.isnan(opt_gs_scale).any() or torch.isinf(opt_gs_scale).any():
            print(f"opt_gs_scale invalid")
            print(
                f"Value: {opt_gs_scale.detach().cpu()}")
            raise Exception()

        model.training_setup_scale_rot_trans(
            opt_gs_scale, opt_cam_rot, opt_cam_trans, self.config)

        gt_color_i = self.dataset[frame_id_i][1]
        gt_color_j = self.dataset[frame_id_j][1]
        gt_color_i = self.transform(gt_color_i).to(device)
        gt_color_j = self.transform(gt_color_j).to(device)

        num_iters = self.config["iterations"] * 2
        current_min_loss = float("inf")

        print(f"\nAligning sub gaussians {frame_id_i:04d}")
        pbar = tqdm(range(num_iters), desc="Aligning...",
                    dynamic_ncols=True)

        color_loss_weight = 1.0
        depth_loss_weight = 0.1
        mw2_weight = 0.1

        num_cluster = 1000
        main_component = self.get_main_gmm_component(model, num_cluster)
        mini_component = self.get_mini_gmm_component(mini, num_cluster)
        print(
            f"Color Weight {color_loss_weight}, Depth Loss Weight {depth_loss_weight}, MW2 Loss Weight {mw2_weight}")

        color_loss_list = []
        depth_loss_list = []
        mw2_loss_list = []
        total_loss_list = []
        mw2_loss_history = np.array([float("inf")] * 10)
        mw2_loss_flag = True

        for iter in pbar:

            color_loss, depth_loss, mw2loss = self.compute_align_losses(
                num_iters, iter, frame_id_i,
                model, mini, reference_w2c, opt_gs_scale, opt_cam_rot, opt_cam_trans, gt_color_i,
                main_component, mini_component, mw2_loss_flag)

            mw2_loss_history[iter % len(mw2_loss_history)] = mw2loss.item()
            if mw2_loss_flag and iter > 10 and np.max(mw2_loss_history) <= mw2loss.item():
                mw2_loss_flag = False

            total_loss = color_loss_weight * color_loss + \
                depth_loss_weight * depth_loss + mw2_weight * mw2loss
            total_loss.backward()
            if torch.isnan(opt_gs_scale.grad).any() or torch.isinf(opt_gs_scale.grad).any():
                print(f"opt_gs_scale.grad invalid")
                print(
                    f"Value: {opt_gs_scale.detach().cpu()} Grad: {opt_gs_scale.grad.detach().cpu()}")
                print(
                    f"Color Loss {color_loss.detach().cpu()}, Depth Loss {depth_loss.detach().cpu()}, MW2 Loss {mw2loss.detach().cpu()}")
                raise Exception()
            model.optimizer.step()
            model.optimizer.zero_grad()

            with torch.no_grad():
                # pbar.set_postfix_str(
                #     f"Color Loss: {color_loss.item():.0f}, Depth Loss: {depth_loss.item():.0f}, MW2 Loss: {mw2loss:.0f}")
                if total_loss.item() < current_min_loss:
                    current_min_loss = total_loss.item()
                    best_w2c = torch.eye(4, dtype=torch.float32, device=device)
                    best_w2c[:3, :3] = build_rotation(F.normalize(
                        opt_cam_rot[None].clone().detach()))[0]
                    best_w2c[:3, 3] = opt_cam_trans.clone().detach()
                    best_gs_scale = opt_gs_scale.detach().clone()

                color_loss_list.append(color_loss.item())
                depth_loss_list.append(depth_loss.item())
                mw2_loss_list.append(mw2loss.item())
                total_loss_list.append(total_loss.item())

        final_c2w_i = torch.inverse(reference_w2c @ best_w2c)
        final_c2w_i[3, :].fill_(0.0)
        final_c2w_i[3, 3] = 1.0

        # compute pose j
        rel_c2w_j[0:3, 3] = rel_c2w_j[0:3, 3] * best_gs_scale
        c2w_j = final_c2w_i @ rel_c2w_j
        # update scale to gaussians
        normalize_scale = mini.normalize_scale
        mini = mini.scale_gaussians(best_gs_scale)
        self.print_align_message(
            frame_id_i, frame_id_j, current_min_loss, final_c2w_i, best_gs_scale, rel_c2w_j, normalize_scale)

        # save_result = {
        #     "frame_id": frame_id_i,
        #     "current_min_loss": current_min_loss,
        #     "current_depth_loss": current_depth_loss,
        #     "current_color_loss": current_color_loss,
        #     "current_mw2_loss": current_mw2_loss,
        #     "color_loss_list": color_loss_list,
        #     "depth_loss_list": depth_loss_list,
        #     "mw2_loss_list": mw2_loss_list,
        # }
        # output_path = Path(config["output_path"])
        # with (output_path / f"align_result_{frame_id_i:04d}.json").open("w") as f:
        #     json.dump(save_result, f, indent=4)
        return mini, final_c2w_i, c2w_j

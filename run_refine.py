import argparse
from argparse import ArgumentParser
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.datasets import BaseDataset, get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.losses import l1_loss, ssim
from src.utils.io_utils import load_config
from src.utils.utils import (batch_search_faiss, get_render_settings,
                             render_gaussian_model, setup_seed)


def get_args():
    parser = argparse.ArgumentParser(
        description='Arguments to compute the mesh')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Checkpoint path', default="output/re10k/000c3ab189999a83")
    parser.add_argument('--config_path', type=str,
                        help='Config path', default="")
    return parser.parse_args()


def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal


class Refinement(object):

    def __init__(self, checkpoint_path, config_path, config=None) -> None:
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        setup_seed(self.config["seed"])

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda")
        self.dataset = get_dataset(self.config["dataset_name"])(
            {**self.config["data"], **self.config["cam"]})
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = np.array(self.dataset.poses)
        self.fx = self.dataset.intrinsics[0, 0]
        self.fy = self.dataset.intrinsics[1, 1]
        self.cx = self.dataset.intrinsics[0, 2]
        self.cy = self.dataset.intrinsics[1, 2]
        self.width, self.height = self.dataset.width, self.dataset.height

        self.estimated_c2ws = torch.load(
            self.checkpoint_path / "estimated_c2w.ckpt")
        self.submaps_paths = sorted(
            list((self.checkpoint_path / "submaps").glob('*')))

        self.cameras_extent = self.compute_camera_extent(self.estimated_c2ws)[
            1]

        sample_rate = self.config["sample_rate"]
        n_views = self.config["n_views"]
        n_frames = len(self.dataset)
        frame_ids = np.arange(n_frames)
        test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]
        remain_frame_ids = np.array(
            [i for i in frame_ids if i not in test_frame_ids])
        train_frame_ids = remain_frame_ids[np.linspace(
            0, remain_frame_ids.shape[0] - 1, n_views).astype(int)]

        self.test_frame_ids = test_frame_ids
        self.train_frame_ids = train_frame_ids
        print(f"Training Frames: {train_frame_ids}")
        print(f"Eval Frames: {test_frame_ids}")

    def compute_camera_extent(self, estimated_c2w):
        cam_centers = []
        for i in range(len(estimated_c2w)):
            cam_center = estimated_c2w[i][:3, 3:4]
            cam_centers.append(cam_center)
        center, diag = get_center_and_diag(cam_centers)
        radius = diag * 1.1
        translate = -center
        return translate, radius

    def merge_submaps(self, submaps_paths, radius: float = 0.0001) -> GaussianModel:
        model = GaussianModel(3)
        device = self.device
        pts_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 500, faiss.METRIC_L2))
        pts_index.nprobe = 5

        for submap_path in tqdm(submaps_paths, desc="Merging submaps"):
            gaussian_params = torch.load(submap_path)
            sh_degree = gaussian_params["active_sh_degree"]
            xyz = gaussian_params["xyz"].to(
                device).float().contiguous()
            features_dc = gaussian_params["features_dc"].to(
                device).float().contiguous()
            features_rest = gaussian_params["features_rest"].to(
                device).float().contiguous()
            scales = gaussian_params["scaling"].to(
                device).float().contiguous()
            rotations = gaussian_params["rotation"].to(
                device).float().contiguous()
            opacities = gaussian_params["opacity"].to(
                device).float().contiguous()

            pts_index.train(xyz.cpu())
            distances, _ = batch_search_faiss(pts_index, xyz, 8)
            neighbor_num = (distances < radius).sum(axis=1).int()

            ids_to_include = torch.where(neighbor_num == 0)[0]
            pts_index.add(xyz[ids_to_include].cpu())

            model.add_gaussians({
                "xyz": xyz[ids_to_include],
                "features_dc": features_dc[ids_to_include],
                "features_rest": features_rest[ids_to_include],
                "scales": scales[ids_to_include],
                "rotations": rotations[ids_to_include],
                "opacities": opacities[ids_to_include],
                "sh_degree": sh_degree
            })

        total_xyz = model.get_xyz()
        max_pts_num = 1_000_000
        if total_xyz.shape[0] > max_pts_num:
            n = total_xyz.shape[0]
            mask = torch.ones(n, dtype=torch.bool)
            indices = torch.randperm(n)[:max_pts_num]
            mask[indices] = False
            model.prune_points(mask)

        return model

    def refine_global_map(self,
                          model: GaussianModel,
                          dataset: BaseDataset,
                          train_frame_ids: list,
                          max_iterations: int,
                          output_dir,
                          save_iteration=False) -> GaussianModel:
        width = self.dataset.width
        height = self.dataset.height
        intrinsics = self.dataset.intrinsics
        device = self.device

        estimated_c2ws = self.estimated_c2ws

        transform = torchvision.transforms.ToTensor()

        opt_params = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        model.active_sh_degree = 0
        model.training_setup(opt_params)
        iteration = 0

        for iteration in tqdm(range(max_iterations), desc="Refinement"):
            idx = np.random.randint(0, len(train_frame_ids))
            frame_id = train_frame_ids[idx]
            gt_color = transform(dataset[frame_id][1]).to(device)
            model.update_learning_rate(iteration)
            if iteration > 0 and iteration % 1000 == 0:
                model.oneupSHdegree()

            c2w = estimated_c2ws[idx]
            w2c = torch.inverse(c2w)
            render_settings = get_render_settings(
                width, height, intrinsics, w2c.cpu().numpy()
            )

            render_dict = render_gaussian_model(model, render_settings)
            rendered_color = render_dict["color"]
            radii = render_dict["radii"]
            visibility_filter = radii > 0.0
            means2D = render_dict["means2D"]

            color_l1_loss = l1_loss(rendered_color, gt_color)
            color_ssim_loss = 1.0 - ssim(rendered_color, gt_color)
            color_loss = (1.0 - opt_params.lambda_dssim) * color_l1_loss \
                + opt_params.lambda_dssim * color_ssim_loss

            total_loss = color_loss
            total_loss.backward()

            with torch.no_grad():
                opacity_reset_interval = 3000
                densify_until_iter = 15_000
                densify_from_iter = 500
                densification_interval = 100
                densify_grad_threshold = 0.0002
                if (iteration + 1) < densify_until_iter:

                    # Keep track of max radii in image-space for pruning
                    model.max_radii2D[visibility_filter] = torch.max(
                        model.max_radii2D[visibility_filter], radii[visibility_filter])
                    model.add_densification_stats(means2D, visibility_filter)

                    if (iteration + 1) > densify_from_iter and (iteration + 1) % densification_interval == 0:
                        size_threshold = 20 if (
                            iteration + 1) > opacity_reset_interval else None
                        model.densify_and_prune(
                            densify_grad_threshold, 0.005, self.cameras_extent, size_threshold, radii)

                    if (iteration + 1) % opacity_reset_interval == 0 or ((iteration + 1) == densify_from_iter):
                        model.reset_opacity()

                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

                if save_iteration and (iteration + 1) % 1000 == 0:
                    save_path = output_dir / "gaussians" / \
                        f"iter_{iteration + 1}_gs.ply"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save_ply(str(save_path))
            iteration += 1
        return model

    def run_global_refinement(self):
        print("Running global refinement...")

        merged_gaussians = self.merge_submaps(
            self.submaps_paths)
        ply_path = Path(self.checkpoint_path) / \
            "gaussians" / "before_refined_gs.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        merged_gaussians.save_ply(str(ply_path))

        iter_count = 30000
        refined_gaussians = self.refine_global_map(
            merged_gaussians, self.dataset, self.train_frame_ids, iter_count, self.checkpoint_path, save_iteration=False)

        ply_path = Path(self.checkpoint_path) / \
            "gaussians" / "global_refined_gs.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        refined_gaussians.save_ply(str(ply_path))

    def run(self) -> None:

        print("Starting refinement...🍺")

        self.run_global_refinement()


if __name__ == "__main__":
    args = get_args()
    if args.config_path == "":
        args.config_path = Path(args.checkpoint_path) / "config.yaml"

    refinement = Refinement(Path(args.checkpoint_path),
                            Path(args.config_path))
    refinement.run()

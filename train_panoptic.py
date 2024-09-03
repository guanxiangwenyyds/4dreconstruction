from scene.PanopticSportsDataset import PanopticSportsDataset, transform_into_ply
from scene.deformation_model import DeformationModel
from scene.gaussian_model import GaussianModel
from scene.pcd import fetchPly, BasicPointCloud, load_point_cloud_from_npz
from scene.math_utils import o3d_knn
from arguments import ModelParams, PipelineParams, PanopticParams
from scene.data_management import Delta_v2
from gaussian_renderer import render
from loss import l1_loss, compute_psnr2, rigidity_loss, mse_loss

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import argparse
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def setup_parser():
    parser = argparse.ArgumentParser()
    return parser


def main():

    parser = setup_parser()

    pan_configs = PanopticParams(parser)
    pipeline_configs = PipelineParams(parser)

    args = parser.parse_args()

    pan_params = pan_configs.extract(args)
    pipe_params = pipeline_configs.extract(args)


    # model path
    save_gaussian = f"output/result/{pan_params.dataset}{pan_params.identifier}/gaussian.ply"
    save_model = f"output/result/{pan_params.dataset}{pan_params.identifier}/model.ckpt"
    save_init_gs = f"output/result/{pan_params.dataset}{pan_params.identifier}/gaussian_init.ply"


    # data path
    datadir = f"data/{pan_params.dataset}/"
    metafile = 'train_meta.json'
    meta_npz_path = f"data/{pan_params.dataset}/init_pt_cld.npz"

    # Create Gaussian Point Group
    gaussian = GaussianModel()
    is_fg = gaussian.is_fg_point(meta_npz_path)
    transform_into_ply(datadir)
    pcd = fetchPly(f'data/{pan_params.dataset}/pointd3D.ply')
    gaussian.create_from_pcd(pcd)

    # Create Dataset
    scene = PanopticSportsDataset(datadir=datadir, metafile=metafile)


    # Create Deformation Model
    fv = gaussian.get_feature_vector(t=0.0, pe_t=pan_params.pe_t, pe_xyz=pan_params.pe_xyz, only_xyz_t=pan_params.only_xyz_t)
    input_dim = fv.shape[1]
    model = DeformationModel(input_dim, pan_params.depth, pan_params.common_width).to(device)

    # Parameter for rendering
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Optimizer
    optimizer_gaussian = torch.optim.Adam(gaussian.parameters(), lr=pan_params.gaussian_lr_init)
    optimizer_deformation = torch.optim.Adam(model.parameters(), lr=pan_params.deformation_lr)
    scheduler_gaussian = torch.optim.lr_scheduler.StepLR(optimizer_gaussian, step_size=pan_params.gaussian_lr_scheduler, gamma=0.1)
    scheduler_deformation = torch.optim.lr_scheduler.StepLR(optimizer_deformation, step_size=pan_params.deformation_lr_scheduler, gamma=0.1)


    # Train initial Gaussian Model
    if not pan_params.skip_init_gaussian:
        print(" Training initial Gaussian Model... ...")
        print(" Loading 1st frame data... ... ")
        scene.readPanopticmeta(index=0, load_mask=None)
        train_cams = scene.cam_infos
        train_camera_loader = DataLoader(train_cams, batch_size=1, shuffle=True, num_workers=0, collate_fn=list)
        iterations = 0
        pbar_gaussian = tqdm(range(pan_params.num_epoch_4_init), desc="Training Initial Gaussian Model", total=pan_params.num_epoch_4_init, unit="epoch", leave=True)
        for e in pbar_gaussian:

            loss_list = []
            psnr_list = []
            pbar_inner = tqdm(train_camera_loader, desc="Camera Processing", unit="camera", leave=False)
            for camera in pbar_inner:

                optimizer_gaussian.zero_grad()

                if iterations % pan_params.densification_inter_iteration == 0 and iterations != 0:
                    ob_densification = True
                else:
                    ob_densification = False

                rendered_imgs = []
                gt_imgs = []
                cam = camera[0]

                stage = 'coarse'
                delta = None

                render_pkg = render(cam, gaussian, pipe_params, background, scaling_modifier=1.0,
                                    override_color=None, stage=stage, cam_type="PanopticSports", delta=delta)

                # rendered image
                rendered_img = render_pkg['render']
                rendered_imgs.append(rendered_img.unsqueeze(0))
                rendered_img_tensor = torch.cat(rendered_imgs, 0).to(device)

                # ground truth image
                gt_img = cam['image']
                gt_imgs.append(gt_img.unsqueeze(0))
                gt_images_tensor = torch.cat(gt_imgs, 0).to(device)

                # mask
                # mask = cam['mask']

                # Loss Computation
                l1 = l1_loss(rendered_img_tensor, gt_images_tensor)

                loss = l1
                loss_list.append(loss.item())

                # psnr
                rendered_img_tensor_p = rendered_img_tensor.float()
                gt_images_tensor_p = gt_images_tensor.float()
                psnr = compute_psnr2(rendered_img_tensor_p, gt_images_tensor_p)
                psnr_list.append(psnr.item())

                loss.backward()

                optimizer_gaussian.step()

                if e >= pan_params.densification_from_epoch_in_coarse and e < pan_params.densification_till_epoch_in_coarse and gaussian.get_xyz.shape[0] < pan_params.max_points:
                    with torch.no_grad():
                        radii_list = []
                        visibility_filter_list = []
                        viewspace_point_tensor_list = []

                        visibility_filter = render_pkg["visibility_filter"]
                        visibility_filter_list.append(visibility_filter.unsqueeze(0))
                        visibility_filter = torch.cat(visibility_filter_list).any(dim=0).to(device)

                        radii = render_pkg["radii"]

                        viewspace_point_tensor = render_pkg["viewspace_points"]
                        viewspace_point_tensor_list.append(viewspace_point_tensor)
                        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
                        for idx in range(0, len(viewspace_point_tensor_list)):
                            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

                        gaussian.max_radii2D[visibility_filter] = torch.max(gaussian.max_radii2D[visibility_filter],
                                                                            radii[visibility_filter])
                        gaussian.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                        if ob_densification == True:
                            gaussian.densify(scene.nerf_normalization["radius"])
                            optimizer_gaussian = torch.optim.Adam(gaussian.parameters(), lr=pan_params.gaussian_lr)

                iterations += 1
                pbar_inner.set_description(f"Iterating 1st frame of all train_cameras, Loss: {loss:.4f}, PSNR: {psnr:.4f}")

            avg_loss = sum(loss_list) / len(loss_list)
            avg_psnr = sum(psnr_list) / len(psnr_list)
            num_points = gaussian.get_xyz.shape[0]
            pbar_gaussian.set_description(f"Training initial Gaussian Points: {e + 1}/{pan_params.num_epoch_4_init} Avg Loss: {avg_loss:.4f} Avg PSNR: {avg_psnr:.4f} Gaussian Points: {num_points} Iterations:{iterations}")


        gaussian.save_ply(save_init_gs)
        print(f"Initial Gaussian Points have been saved at：{save_init_gs}")
    else:
        print(f"Training initial Gaussian and Loading trained initial gaussian points... ... ")
        gaussian.load_ply(save_init_gs)
        pass

    # find neighbour for rigidity loss
    fg_pts = gaussian.get_xyz[is_fg]
    neighbor_sq_dist, neighbor_indices = o3d_knn(fg_pts.detach().cpu().numpy(), 20)
    neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)


    # Train Deformation Model
    if pan_params.train_deformation_model:
        print("Training Deformation Model... ...")

        if pan_params.fix_gaussian:
            for param in gaussian.parameters():
                param.requires_grad = False
            print("Parameters of Gaussian Points are fixed in Deformation Training")
        else:
            optimizer_gaussian = torch.optim.Adam(gaussian.parameters(), lr=pan_params.gaussian_lr)

        iterations = 0
        pbar_de = tqdm(range(pan_params.num_epoch_4_deformation), desc="Training Deformation Model", total=pan_params.num_epoch_4_deformation, unit="epoch", leave=True)
        for e in pbar_de:

            loss_list = []
            psnr_list = []

            prev_fi_rot = []
            prev_fi_offset = []

            frame_idx = list(range(0, 150))
            pbar_frame = tqdm(frame_idx, desc="Loading frames... ...", leave=False)
            for fi in pbar_frame:

                loss_list_fi = []
                loss_mse_fi = []
                loss_rg_fi =[]
                psnr_list_fi = []

                curr_fi_rot = []
                curr_fi_offset = []

                scene.readPanopticmeta(index=fi, load_mask=None)
                train_cams = scene.cam_infos
                train_camera_loader = DataLoader(train_cams, batch_size=1, shuffle=False, num_workers=0, collate_fn=list)

                cam_idx = 0
                pbar_cam = tqdm(train_camera_loader, desc="Loading cameras... ...", leave=False)
                for camera in pbar_cam:

                    rendered_imgs = []
                    gt_imgs = []

                    cam = camera[0]

                    if pan_params.fix_gaussian:
                        pass
                    else:
                        optimizer_gaussian.zero_grad()
                    optimizer_deformation.zero_grad()

                    # Parameters for rendering
                    stage = "fine"
                    timestamp = cam['time']
                    fv = gaussian.get_feature_vector(t=timestamp, pe_t=pan_params.pe_t, pe_xyz=pan_params.pe_xyz, only_xyz_t=pan_params.only_xyz_t)
                    pred = model(fv)
                    delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])

                    render_pkg = render(cam, gaussian, pipe_params, background, scaling_modifier=1.0,
                                        override_color=None, stage=stage, cam_type="PanopticSports", delta=delta)

                    # rendered image
                    rendered_img = render_pkg['render']
                    rendered_imgs.append(rendered_img.unsqueeze(0))
                    rendered_img_tensor = torch.cat(rendered_imgs, 0).to(device)

                    # ground truth image
                    gt_img = cam['image']
                    gt_imgs.append(gt_img.unsqueeze(0))
                    gt_images_tensor = torch.cat(gt_imgs, 0).to(device)

                    # prepare for rigidity loss

                    fg_pts = gaussian.get_xyz[is_fg]

                    curr_offset = (fg_pts[neighbor_indices] - fg_pts[:, None]).clone().detach().requires_grad_(True)
                    curr_rot = gaussian.rotation[is_fg] + delta.rotation[is_fg]

                    # Loss Computation
                    if pan_params.w_l1 == 0.0:
                        l1 = 0
                    else:
                        l1 = l1_loss(rendered_img_tensor, gt_images_tensor)

                    if pan_params.w_mse == 0.0:
                        mse = 0
                    else:
                        mse = mse_loss(rendered_img_tensor, gt_images_tensor)

                    loss_mse_fi.append(mse)

                    # Rigidity loss, when num frame > 0
                    if fi > 0:
                        with torch.no_grad():
                            prev_offset = prev_fi_offset[cam_idx]
                            prev_rot = prev_fi_rot[cam_idx]
                        if pan_params.w_rg == 0.0:
                            rigidity = 0
                        else:
                            rigidity = rigidity_loss(curr_rot, prev_rot, curr_offset, prev_offset, neighbor_weight)
                    else:
                        rigidity = 0
                    loss_rg_fi.append(rigidity)

                    loss = pan_params.w_l1*l1 + pan_params.w_mse*mse + pan_params.w_rg*rigidity
                    loss_list.append(loss.item())
                    loss_list_fi.append(loss.item())

                    # PSNR
                    rendered_img_tensor_p = rendered_img_tensor.float()
                    gt_images_tensor_p = gt_images_tensor.float()
                    psnr = compute_psnr2(rendered_img_tensor_p, gt_images_tensor_p)
                    psnr_list.append(psnr.item())
                    psnr_list_fi.append(psnr.item())

                    loss.backward()

                    if pan_params.fix_gaussian:
                        pass
                    else:
                        optimizer_gaussian.step()
                    optimizer_deformation.step()

                    with torch.no_grad():
                        curr_fi_rot.append(curr_rot)
                        curr_fi_offset.append(curr_offset)

                    cam_idx += 1
                    iterations += 1
                    pbar_cam.set_description(f"Iterating Camera {cam_idx}. LOSS: {loss:.4f}")

                with torch.no_grad():
                    prev_fi_rot = curr_fi_rot
                    prev_fi_offset = curr_fi_offset


                avg_loss_fi = sum(loss_list_fi) / len(loss_list_fi)
                avg_psnr_fi = sum(psnr_list_fi) / len(psnr_list_fi)
                num_points = gaussian.get_xyz.shape[0]
                pbar_frame.set_description(f"Iterating frame {fi}.  Avg Loss: {avg_loss_fi:.4f} Avg PSNR: {avg_psnr_fi} Gaussian Points: {num_points}")


            scheduler_deformation.step()
            scheduler_gaussian.step()

            avg_loss = sum(loss_list) / len(loss_list)
            avg_psnr = sum(psnr_list) / len(psnr_list)
            pbar_de.set_description(
                f"Training Deformation Model: {e + 1}/{pan_params.num_epoch_4_deformation} Avg Loss: {avg_loss:.4f} Avg PSNR: {avg_psnr} Gaussian Points: {gaussian.xyz.shape[0]}")


    torch.save(model.state_dict(), save_model)
    gaussian.save_ply(save_gaussian)


if __name__ == "__main__":
    main()





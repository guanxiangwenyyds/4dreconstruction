from scene.deformation_model import DeformationModel
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, DnerfParams
from scene.DnerfDataset import Scene
from scene.data_management import Delta_v2
from gaussian_renderer import render
from loss import l1_loss, compute_psnr2

from torch.utils.data import DataLoader
from tqdm import tqdm


import argparse
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_parser():
    parser = argparse.ArgumentParser()
    return parser


def update_optimizer(epoch, gaussian, dnerf_params):
    if epoch < 90:
        lr = 0.001
    elif 90 <= epoch < 120:
        lr = 0.00016
    elif 120 <= epoch < 150:
        lr = 0.000016
    elif 150 <= epoch < 200:
        lr = 0.000008
    else:
        lr = 0.0000016

    return torch.optim.Adam([
        {'params': gaussian.xyz, 'lr': lr},
        {'params': gaussian.features_dc, 'lr': dnerf_params.feature_lr},
        {'params': gaussian.features_rest, 'lr': dnerf_params.feature_lr},
        {'params': gaussian.scaling, 'lr': dnerf_params.scaling_lr},
        {'params': gaussian.rotation, 'lr': dnerf_params.rotation_lr},
        {'params': gaussian.opacity, 'lr': dnerf_params.opacity_lr},
    ])


def main():
    parser = setup_parser()

    dnerf_configs = DnerfParams(parser)
    pipeline_configs = PipelineParams(parser)
    scene_configs = ModelParams(parser)

    args = parser.parse_args()

    dnerf_params = dnerf_configs.extract(args)
    pipe_params = pipeline_configs.extract(args)
    scene_params = scene_configs.extract(args)
    scene_params.source_path = f'data/{dnerf_params.dataset}/'

    save_init_gs = f"output/result/{dnerf_params.dataset}{dnerf_params.identifier}/gaussian_init.ply"
    save_gaussian = f'output/result/{dnerf_params.dataset}{dnerf_params.identifier}/gaussian.ply'
    save_model = f"output/result/{dnerf_params.dataset}{dnerf_params.identifier}/model.ckpt"


    # Create Gaussian Point Group
    gaussian = GaussianModel()
    scene = Scene(scene_params, gaussian)

    # Create Deformation Model
    fv = gaussian.get_feature_vector(t=1.0, pe_t=dnerf_params.pe_t, pe_xyz=dnerf_params.pe_xyz, only_xyz_t=dnerf_params.only_xyz_t)
    input_dim = fv.shape[1]
    model = DeformationModel(input_dim, dnerf_params.depth, dnerf_params.common_width).to(device)

    # Optimizer
    optimizer_gaussian = torch.optim.Adam([
        {'params': gaussian.xyz, 'lr': dnerf_params.xyz_lr},
        {'params': gaussian.features_dc, 'lr': dnerf_params.feature_lr},
        {'params': gaussian.features_rest, 'lr': dnerf_params.feature_lr},
        {'params': gaussian.scaling, 'lr': dnerf_params.scaling_lr},
        {'params': gaussian.rotation, 'lr': dnerf_params.rotation_lr},
        {'params': gaussian.opacity, 'lr': dnerf_params.opacity_lr},
    ])
    optimizer_deformation = torch.optim.Adam(model.parameters(), lr=dnerf_params.deformation_lr)

    # Params for Rendering
    train_cams = scene.getTrainCameras()
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    train_camera_loader = DataLoader(train_cams, batch_size=1, shuffle=True, num_workers=16, collate_fn=list)
    iterations = 0
    pbar_epoch = tqdm(range(dnerf_params.num_epoch), desc="Training... ...", total=dnerf_params.num_epoch, unit="epoch", leave=True)
    for e in pbar_epoch:

        loss_list = []
        psnr_list = []
        optimizer_gaussian.zero_grad()
        optimizer_deformation.zero_grad()

        for camera in train_camera_loader:

            if iterations % dnerf_params.densification_inter_iteration == 0 and iterations != 0:
                ob_densification = True
            else:
                ob_densification = False

            rendered_imgs = []
            gt_imgs = []

            camera = camera[0]
            if e < dnerf_params.num_epoch_4_init:
                delta = None
                stage = "coarse"
            else:
                timestamp = camera.time
                fv = gaussian.get_feature_vector(t=timestamp, pe_t=dnerf_params.pe_t, pe_xyz=dnerf_params.pe_xyz, only_xyz_t=dnerf_params.only_xyz_t)
                pred = model(fv)

                delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])
                stage = "fine"

            # render
            render_pkg = render(camera, gaussian, pipe_params, background, scaling_modifier=1.0,
                                override_color=None, stage=stage, cam_type=None, delta=delta)

            # rendered image
            rendered_img = render_pkg['render']
            rendered_imgs.append(rendered_img.unsqueeze(0))
            rendered_img_tensor = torch.cat(rendered_imgs, 0).to(device)

            # ground truth image
            gt_img = camera.original_image
            gt_imgs.append(gt_img.unsqueeze(0))
            gt_images_tensor = torch.cat(gt_imgs, 0).to(device)

            # Loss Computation
            l1 = l1_loss(rendered_img_tensor, gt_images_tensor)
            loss = l1
            loss_list.append(loss.item())

            # PSNR
            rendered_img_tensor_p = rendered_img_tensor.float()
            gt_images_tensor_p = gt_images_tensor.float()
            psnr = compute_psnr2(rendered_img_tensor_p, gt_images_tensor_p)
            psnr_list.append(psnr)

            loss.backward()

            optimizer_gaussian.step()
            optimizer_deformation.step()

            optimizer_gaussian.zero_grad()
            optimizer_deformation.zero_grad()

            if iterations > dnerf_params.densification_from_iteration and iterations < dnerf_params.densification_till_iteration and ob_densification and gaussian.get_xyz.shape[0] < dnerf_params.max_points:
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
                    gaussian.densify(scene.cameras_extent)

                    optimizer_gaussian = update_optimizer(e, gaussian, dnerf_params)

            iterations += 1

            if e == 90:
                optimizer_deformation = torch.optim.Adam(model.parameters(), lr=0.0005)
                optimizer_gaussian = update_optimizer(e, gaussian, dnerf_params)
            elif e == 120:
                optimizer_deformation = torch.optim.Adam(model.parameters(), lr=0.00016)
                optimizer_gaussian = update_optimizer(e, gaussian, dnerf_params)
            elif e == 150:
                optimizer_deformation = torch.optim.Adam(model.parameters(), lr=0.00008)
                optimizer_gaussian = update_optimizer(e, gaussian, dnerf_params)
            elif e == 200:
                optimizer_deformation = torch.optim.Adam(model.parameters(), lr=0.000016)
                optimizer_gaussian = update_optimizer(e, gaussian, dnerf_params)

        if e == dnerf_params.num_epoch_4_init-1:
            gaussian.save_ply(save_init_gs)

        avg_loss = sum(loss_list) / len(loss_list)
        avg_psnr = sum(psnr_list) / len(psnr_list)
        pbar_epoch.set_description(
            f"Training Epoch {e + 1}/{dnerf_params.num_epoch} Avg Loss: {avg_loss:.4f} Avg PSNR: {avg_psnr:.4f} Gaussian Points: {gaussian.get_xyz.shape[0]} Iterations: {iterations}")


    gaussian.save_ply(path=save_gaussian)
    torch.save(model.state_dict(), save_model)
    print(f"Training is complete and the model as well as the Gaussian points are saved in: output/result/{dnerf_params.dataset}{dnerf_params.identifier}/")


if __name__ == "__main__":
    main()
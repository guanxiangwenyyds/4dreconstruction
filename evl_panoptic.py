from scene.PanopticSportsDataset import PanopticSportsDataset
from scene.deformation_model import DeformationModel
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, PanopticParams
from gaussian_renderer import render
from loss import l1_loss, compute_ssim, compute_psnr2
from scene.data_management import Delta_v2
from animation_synthesis import animation_synthesis

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import os

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

    vsl = True
    evl = True

    # model path
    if pan_params.coarse_model:
        gaussian_path = f"output/result/{pan_params.dataset}{pan_params.identifier}/gaussian_init.ply"
    else:
        gaussian_path = f"output/result/{pan_params.dataset}{pan_params.identifier}/gaussian.ply"
    model_path = f"output/result/{pan_params.dataset}{pan_params.identifier}/model.ckpt"

    datadir = f"data/{pan_params.dataset}/"
    metafile = f"{pan_params.test_or_train}_meta.json"


    # Create Gaussian Point Group
    gaussian = GaussianModel()
    gaussian.load_ply(gaussian_path)
    print('num of points:', gaussian.get_xyz.shape[0])

    # Create Dataset Class
    scene = PanopticSportsDataset(datadir=datadir, metafile=metafile)

    # Load Deformation Model
    fv = gaussian.get_feature_vector(t=0.0, pe_t=pan_params.pe_t, pe_xyz=pan_params.pe_xyz, only_xyz_t=pan_params.only_xyz_t)
    input_dim = fv.shape[1]
    model = DeformationModel(input_dim, pan_params.depth, pan_params.common_width).to(device)
    if not pan_params.coarse_model:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Parameter for rendering
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    if vsl:
        print(f'Rendering Model... ...')
        i = 0
        for f_idx in range(pan_params.start_frame, pan_params.start_frame+pan_params.num_frames, pan_params.intervals_frame):

            scene.readPanopticmeta(index=f_idx)
            camera = scene.cam_infos[pan_params.cam_index]

            if pan_params.coarse_model:
                stage = 'coarse'
                delta = None
            else:
                stage = 'fine'
                timestamp = camera['time']
                fv = gaussian.get_feature_vector(t=timestamp, pe_t=pan_params.pe_t, pe_xyz=pan_params.pe_xyz, only_xyz_t=pan_params.only_xyz_t)
                pred = model(fv)
                delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])
                # print('delta.xyz:', delta.xyz.sum())
                # print('delta.scale:', delta.scale.sum())
                # print('delta.rotation:', delta.rotation.sum())
                # breakpoint()

            render_pkg = render(camera, gaussian, pipe_params, background, scaling_modifier=1.0,
                                            override_color=None, stage=stage, cam_type="PanopticSports", delta=delta)

            rendered_img = render_pkg['render']


            rendered_img_np = rendered_img.detach().cpu().numpy()
            rendered_img_np = np.transpose(rendered_img_np, (1, 2, 0))

            if rendered_img_np.dtype == np.float32 or rendered_img_np.dtype == np.float64:
                if rendered_img_np.max() > 1.0:
                    rendered_img_np = rendered_img_np / rendered_img_np.max()  # 归一化到 0..1

            save_path = f"output/result/{pan_params.dataset}{pan_params.identifier}/img/{pan_params.cam_index}/frame_{i:04d}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.imsave(save_path, rendered_img_np)
            plt.close()
            i += 1
        print(f'Rendered images saved at: output/result/{pan_params.dataset}{pan_params.identifier}/img/{pan_params.cam_index}/')
        img_path = f"output/result/{pan_params.dataset}{pan_params.identifier}/img/{pan_params.cam_index}/"
        video_name = 'Video'
        animation_synthesis(img_path, video_name)

    if evl:

        l1_list = []
        psnr_list = []
        ssim_list = []
        print(f'Evaluating on the test set... ...')
        for f_idx in range(pan_params.start_frame, pan_params.start_frame+pan_params.num_frames):
            scene.readPanopticmeta(index=f_idx)
            camera = scene.cam_infos

            for cam_idx in range(len(camera)):

                cam = camera[cam_idx]

                rendered_imgs = []
                gt_imgs = []

                timestamp = cam['time']

                fv = gaussian.get_feature_vector(t=timestamp, pe_t=pan_params.pe_t, pe_xyz=pan_params.pe_xyz, only_xyz_t=pan_params.only_xyz_t)
                pred = model(fv)
                delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])

                render_pkg = render(cam, gaussian, pipe_params, background, scaling_modifier=1.0,
                                    override_color=None, stage="fine", cam_type='PanopticSports', delta=delta)

                rendered_img = render_pkg['render']
                rendered_imgs.append(rendered_img.unsqueeze(0))
                rendered_img_tensor = torch.cat(rendered_imgs, 0).to(device)

                gt_img = cam['image']
                gt_imgs.append(gt_img.unsqueeze(0))
                gt_images_tensor = torch.cat(gt_imgs, 0).to(device)

                # Loss Computation
                l1 = l1_loss(rendered_img_tensor, gt_images_tensor)
                l_ssim = compute_ssim(rendered_img_tensor, gt_images_tensor)

                rendered_img_tensor_p = rendered_img_tensor.float()
                gt_images_tensor_p = gt_images_tensor.float()
                psnr = compute_psnr2(rendered_img_tensor_p, gt_images_tensor_p)

                l1_list.append(l1.item())
                psnr_list.append(psnr.cpu().item())
                ssim_list.append(l_ssim.item())

        l1_mean = np.mean(l1_list)
        psnr_mean = np.mean(psnr_list)
        ssim_mean = np.mean(ssim_list)

        print(f'{pan_params.dataset}.{pan_params.identifier} - average l1 loss: {l1_mean}')
        print(f'{pan_params.dataset}.{pan_params.identifier} - PSNR: {psnr_mean}')
        print(f'{pan_params.dataset}.{pan_params.identifier} - average ssim loss:{ssim_mean}')


if __name__ == "__main__":
    main()
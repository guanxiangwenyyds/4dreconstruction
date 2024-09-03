from scene.deformation_model import DeformationModel
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, DnerfParams
from scene.DnerfDataset import Scene
from gaussian_renderer import render
from loss import l1_loss, compute_ssim, compute_psnr2
from scene.data_management import Delta_v2

import numpy as np
import argparse
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_parser():
    parser = argparse.ArgumentParser()
    return parser



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

    gaussian_path = f'output/result/{dnerf_params.dataset}{dnerf_params.identifier}/gaussian.ply'
    model_path = f"output/result/{dnerf_params.dataset}{dnerf_params.identifier}/model.ckpt"


    gaussian = GaussianModel()
    scene = Scene(scene_params, gaussian)

    gaussian = GaussianModel()
    gaussian.load_ply(gaussian_path)


    fv = gaussian.get_feature_vector(t=1.0)
    input_dim = fv.shape[1]
    model = DeformationModel(input_dim, dnerf_params.depth, dnerf_params.common_width).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_cams = scene.getTestCameras()

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    l1_list=[]
    psnr_list=[]
    ssim_list=[]

    for idx in range(len(test_cams)):

        rendered_imgs = []
        gt_imgs = []

        camera = test_cams[idx]
        timestamp = camera.time
        fv = gaussian.get_feature_vector(t=timestamp)
        pred = model(fv)
        num = gaussian.get_xyz.shape[0]
        delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])

        render_pkg = render(camera, gaussian, pipe_params, background, scaling_modifier=1.0,
                                    override_color=None, stage="fine", cam_type=None, delta=delta)

        rendered_img = render_pkg['render']
        rendered_imgs.append(rendered_img.unsqueeze(0))
        rendered_img_tensor = torch.cat(rendered_imgs, 0).to(device)

        gt_img = camera.original_image
        gt_imgs.append(gt_img.unsqueeze(0))
        gt_images_tensor = torch.cat(gt_imgs, 0).to(device)

        # Loss Computation
        l1 = l1_loss(rendered_img_tensor, gt_images_tensor)
        l_ssim = compute_ssim(rendered_img_tensor, gt_images_tensor)
        lambda_ssim = 0.0
        loss = l1 + lambda_ssim * (1 - l_ssim)

        rendered_img_tensor_p = rendered_img_tensor.float()
        gt_images_tensor_p = gt_images_tensor.float()
        psnr = compute_psnr2(rendered_img_tensor_p, gt_images_tensor_p)


        l1_list.append(l1.item())
        psnr_list.append(psnr.cpu().item())
        ssim_list.append(l_ssim.item())

    print(f'{dnerf_params.dataset} - average l1 loss:{np.mean(l1_list)}')
    print(f'{dnerf_params.dataset} - average psnr:{np.mean(psnr_list)}')
    print(f'{dnerf_params.dataset} - average ssim loss:{np.mean(ssim_list)}')

if __name__ == "__main__":
    main()
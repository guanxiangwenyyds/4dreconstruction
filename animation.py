from scene.gaussian_model import GaussianModel
from scene.deformation_model import DeformationModel
from arguments import ModelParams, PipelineParams, DnerfParams
from gaussian_renderer import render
from scene.DnerfDataset import Scene
from scene.data_management import Delta_v2
from animation_synthesis import animation_synthesis

from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_parser():
    parser = argparse.ArgumentParser()
    return parser


to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


def main():

    # Parameters
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
    render_path = f'output/result/{dnerf_params.dataset}{dnerf_params.identifier}/'

    video_srd = True
    video_0_5x = True
    video_1_0x = True


    # Load Model
    gaussian = GaussianModel()
    scene = Scene(scene_params, gaussian)

    gaussian.load_ply(gaussian_path)
    video_cams = scene.getVideoCameras()

    fv = gaussian.get_feature_vector(t=1.0)
    input_dim = fv.shape[1]
    depth = 3
    common_width = 128
    model = DeformationModel(input_dim, depth, common_width).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Generate Surround View Video
    if video_srd:
        print('Composite Surround View Video')
        render_srd_path = os.path.join(render_path, 'renders_srd')
        if not os.path.exists(render_srd_path):
            os.makedirs(render_srd_path)

        for idx, view in enumerate(tqdm(video_cams, desc="Rendering progress")):

            timestamp = view.time
            fv = gaussian.get_feature_vector(t=timestamp)
            pred = model(fv)
            delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])

            render_pkg = render(view, gaussian, pipe_params, background, scaling_modifier=1.0,
                                override_color=None, stage="fine", cam_type=None, delta=delta)

            rendered_img = render_pkg['render']

            img_to_save = to8b(rendered_img).transpose(1, 2, 0)
            file_path = os.path.join(render_srd_path, f'frame_{idx:04d}.png')
            plt.imsave(file_path, img_to_save)

        print(f"Frames for 0.5x has been save at {render_srd_path} ")
        animation_synthesis(render_srd_path, 'Video_srd')


    # Generate 0.5x Video (fixed viewpoint)
    if video_0_5x:
        print('Composite 0.5x Video')
        render_0_5_path = os.path.join(render_path, 'renders_0_5x')
        if not os.path.exists(render_0_5_path):
            os.makedirs(render_0_5_path)

        num_frames = len(video_cams)
        x05_frames = 2 * num_frames
        print('Num of frames for :', x05_frames)

        for idx in tqdm(range(x05_frames), desc='Rendering frames'):

            timestamp = 1.0*idx/x05_frames
            fv = gaussian.get_feature_vector(t=timestamp)
            pred = model(fv)
            delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])

            render_pkg = render(video_cams[0], gaussian, pipe_params, background, scaling_modifier=1.0,
                                override_color=None, stage="fine", cam_type=None, delta=delta)

            rendered_img = render_pkg['render']

            img_to_save = to8b(rendered_img).transpose(1, 2, 0)
            file_path = os.path.join(render_0_5_path, f'frame_{idx:04d}.png')
            plt.imsave(file_path, img_to_save)
        print(f"Frames for 0.5x has been save at {render_0_5_path} ")
        animation_synthesis(render_0_5_path, 'Video_0_5x')

    # Generate 1.0x Video (fixed viewpoint)
    if video_1_0x:
        print('Composite 1.0x Video')
        render_1_0_path = os.path.join(render_path, 'renders_1_0x')
        if not os.path.exists(render_1_0_path):
            os.makedirs(render_1_0_path)

        num_frames = len(video_cams)
        x1_frames = num_frames
        print('Total frames:', x1_frames)

        for idx in tqdm(range(x1_frames), desc='Rendering frames'):

            timestamp = 1.0*idx/x1_frames
            fv = gaussian.get_feature_vector(t=timestamp)
            pred = model(fv)
            delta = Delta_v2(xyz=pred['xyz'], scale=pred['scale'], rotation=pred['rotation'])

            render_pkg = render(video_cams[0], gaussian, pipe_params, background, scaling_modifier=1.0,
                                override_color=None, stage="fine", cam_type=None, delta=delta)

            rendered_img = render_pkg['render']

            img_to_save = to8b(rendered_img).transpose(1, 2, 0)
            file_path = os.path.join(render_1_0_path, f'frame_{idx:04d}.png')
            plt.imsave(file_path, img_to_save)

        print(f"Frames for 1.0x has been save at {render_1_0_path} ")
        animation_synthesis(render_1_0_path, 'Video_1_0x')

if __name__ == "__main__":
    main()
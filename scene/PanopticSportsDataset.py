from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from torch.utils.data import Dataset

from plyfile import PlyData, PlyElement
from typing import NamedTuple
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import json
import os

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

class PanopticSportsDataset(Dataset):

    def __init__(self, datadir, metafile, transform=None):
        self.datadir = datadir
        with open(os.path.join(datadir, metafile)) as f:
            self.meta = json.load(f)
        self.transform = transform
        self.max_time = 0
        self.scene_radius = 0
        self.nerf_normalization = {}

    def readPanopticmeta_v2(self, randomly=True, num_frames=30, start_frame=0):
        '''
            parameters:
                randomly -> True: Randomly read frames and corresponding camera config from whole dataset
                randomly -> False: Load fixed number of frames in order from dataset for training process
                num_frames -> Number of frames in un-randomly mode
        '''

        w = self.meta['w']
        h = self.meta['h']
        self.max_time = len(self.meta['fn'])
        self.cam_infos = []

        if randomly == True:
            array = reset_array(26)
            for fn in range(150):
                time = fn / self.max_time
                # print(f"时间信息：{time}")
                index, array = draw_element(array)

                focal = self.meta['k'][fn][index]
                w2c = self.meta['w2c'][fn][index]
                # fn = self.meta['fn'][fn][index]

                filename = "{:06d}.jpg".format(fn)
                image = Image.open(os.path.join(self.datadir, "ims", str(index), filename))
                im_data = np.array(image.convert("RGBA"))
                im_data = PILtoTorch(im_data, None)[:3, :, :]

                camera = setup_camera(w, h, focal, w2c)
                self.cam_infos.append({
                    "camera": camera,
                    "time": time,
                    "image": im_data,
                    # "masked_image": im_masked,
                    # "mask": im_mask,
                })
        else:

            for fn in range(start_frame, start_frame + num_frames):
                time = fn / self.max_time
                for index in range(27):
                    focal = self.meta['k'][fn][index]
                    w2c = self.meta['w2c'][fn][index]

                    filename = "{:06d}.jpg".format(fn)
                    image = Image.open(os.path.join(self.datadir, "ims", str(index), filename))
                    im_data = np.array(image.convert("RGBA"))
                    im_data = PILtoTorch(im_data, None)[:3, :, :]

                    camera = setup_camera(w, h, focal, w2c)
                    self.cam_infos.append({
                        "camera": camera,
                        "time": time,
                        "image": im_data,
                        # "masked_image": im_masked,
                        # "mask": im_mask,
                    })

        cam_centers = np.linalg.inv(self.meta['w2c'][0])[:, :3, 3]  # Get scene radius
        self.scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

        self.nerf_normalization = {
            "radius": self.scene_radius,
            "translate": torch.tensor([0, 0, 0])
        }

        return self.cam_infos, self.max_time, self.scene_radius, self.nerf_normalization


    def readPanopticmeta(self, index, load_mask=None):
        '''
        第 index 帧中的27个相机数据组成的一个字典
        '''
        w = self.meta['w']
        h = self.meta['h']
        self.max_time = len(self.meta['fn'])
        self.cam_infos = []

        focals = self.meta['k'][index]
        w2cs = self.meta['w2c'][index]
        fns = self.meta['fn'][index]
        cam_ids = self.meta['cam_id'][index]

        time = 10 * index / len(self.meta['fn'])
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image = Image.open(os.path.join(self.datadir, "ims", fn))

            if load_mask is True:
                mask_fn = fn.replace('.jpg', '.png')
                image_mask = Image.open(os.path.join(self.datadir, "seg", mask_fn))

                mask_data = np.array(image_mask.convert("RGBA"))
                im_mask = mask_data[:, :, 0]
                im_mask = (im_mask == 255).astype(np.uint8)

                im_data = np.array(image.convert("RGBA"))
                im_masked = im_data.copy()
                im_masked[:, :, 0] *= im_mask
                im_masked[:, :, 1] *= im_mask
                im_masked[:, :, 2] *= im_mask

                im_data = PILtoTorch(im_data, None)[:3, :, :]
                im_masked = PILtoTorch(im_masked, None)[:3, :, :]

            else:
                im_mask = None
                im_masked = None
                im_data = np.array(image.convert("RGBA"))
                im_data = PILtoTorch(im_data, None)[:3, :, :]


            camera = setup_camera(w, h, focal, w2c)
            self.cam_infos.append({
                "camera": camera,
                "time": time,
                "image": im_data,
                "masked_image": im_masked,
                "mask": im_mask,
            })

        cam_centers = np.linalg.inv(self.meta['w2c'][0])[:, :3, 3]  # Get scene radius
        self.scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

        self.nerf_normalization = {
            "radius": self.scene_radius,
            "translate": torch.tensor([0, 0, 0])
        }

        return self.cam_infos, self.max_time, self.scene_radius, self.nerf_normalization


def setup_camera(w, h, k, w2c, near=0.01, far=100):

    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam


def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    if np.array(resized_image_PIL).max()!=1:
        resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def storePly(path, xyz, rgb):
    """
    点云的位置和颜色信息格式化并保存到一个PLY文件中

    """

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def transform_into_ply(datadir):
    '''
    Transform point information
    '''
    ply_path = os.path.join(datadir, "pointd3D.ply")
    plz_path = os.path.join(datadir, "init_pt_cld.npz")

    if os.path.exists(ply_path):
        print(f"File '{ply_path}' already exists.")

    data = np.load(plz_path)["data"]
    xyz = data[:, :3]
    rgb = data[:, 3:6]
    storePly(ply_path, xyz, rgb)
    print(f"PLY file created at {ply_path}")


def reset_array(len):
    return list(range(0, len))


def draw_element(arr):
    if not arr:  # 如果数组为空，则重置
        arr = reset_array(26)
    random_index = random.randint(0, len(arr) - 1)  # 生成随机索引
    element = arr.pop(random_index)  # 移除并返回该元素
    return element, arr
import os
from PIL import Image
from scene.geometry import getWorld2View2, focal2fov, fov2focal

import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData
from scene.gaussian_model import BasicPointCloud
from typing import NamedTuple
from scene.gaussian_utils import SH2RGB


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int


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


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):

    timestamp_mapper, max_time = read_timeline(path)

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    else:
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper={}):
    """
    从指定的 JSON 文件中读取相机的转换信息
    :param
        path: 存储转换信息和图像文件的路径。
        transformsfile: 包含相机转换信息的 JSON 文件名。
        white_background: 布尔值，指定是否使用白色背景合成图像。
        extension: 图像文件的后缀，默认为 .png。
        mapper: 一个映射函数或字典，用于将 JSON 文件中的时间标记转换为实际使用的时间标记。
    :return
        am_infos：包含所有帧的相机信息的列表。每个 CameraInfo 对象包含以下字段：
        uid: 帧的唯一标识符。
        R（旋转矩阵）和 T（平移向量）：描述相机在3D空间中的姿态。
        FovY 和 FovX：相机的垂直和水平视场角。
        image: 处理后的图像，可能包含白色背景。
        image_path, image_name: 图像的路径和文件名。
        width, height: 图像的尺寸。
        time: 映射后的时间值，用于确定帧的时间点。
        mask: 可选的遮罩信息，此例中未使用。
    """
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'], contents['w'])

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(frame["file_path"] + extension)
            cam_name = cam_name.replace('./', '')

            time = mapper[frame["time"]]

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image, (800, 800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.shape[1],
                                        height=image.shape[2],
                                        time=time, mask=None))

    return cam_infos


def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    """
    生成一系列相机信息，包括它们的姿态和其他相关属性
    :parameter：
        path: 存储转换信息的基本路径。
        template_transformsfile: 包含相机转换参数的JSON文件的名称。
        extension: 图像文件的扩展名，用于加载与相机关联的图像。
        maxtime: 用于定义时间范围，可能与动画或场景的时间标尺相关。
    :return
        cam_infos：一个包含所有生成相机信息的列表。每个 CameraInfo 对象包含：
        uid：唯一标识符。
        R（旋转矩阵）和 T（平移向量）：定义相机在世界坐标系中的位置和方向。
        FovY 和 FovX：相机的垂直和水平视场角。
        image：关联的图像数据。
        width 和 height：图像的尺寸。
        time：对应的时间值，归一化到0到1的范围。
        mask：可选的遮罩信息，此示例中未使用。

    """
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    def pose_spherical(theta, phi, radius):
        """
        生成相机的世界到相机坐标系的变换矩阵
        """
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w

    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])

    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(frame["file_path"] + extension)
        cam_name = cam_name.replace('./', '')

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos


def getNerfppNorm(cam_info):
    """
    根据一组相机的信息计算归一化参数
    主要帮助设置一个适当的场景坐标系
    使得相机捕捉的场景可以在一个统一且优化的空间尺度中进行处理
    :return
            translate --> translation vector
            radius --> normalized radius
    """

    def get_center_and_diag(cam_centers):
        """
        接收一组相机中心点坐标，用于计算 场景的中心 和 对角线 长度

        """
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()

    return {"translate": translate, "radius": radius}   #


def fetchPly(path):
    """
    读取指定路径的PLY文件, 提取点云数据，包括点的位置、颜色和法线信息。
    将这些数据封装成一个 BasicPointCloud 对象并返回

    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float
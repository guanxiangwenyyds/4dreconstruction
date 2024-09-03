from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.colmap_loader import read_points3D_binary, read_points3D_text
from scene.hyper_loader import format_hyper_data, Load_hyper_data
from scene.geometric import focal2fov, getNerfppNorm, getWorld2View2

import matplotlib.pyplot as plt
from typing import NamedTuple
from plyfile import PlyData
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import json
import copy
import math
import sys
import os


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
    time: float
    mask: np.array

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

# READ COLMAP PROJECT
def readColmapSceneInfo(path, images, eval, llffhold=8):

    # get extrinsic and intrinsic from path
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # com_infos which has all cameras' config data
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # breakpoint()
    # split train and test in com_infos
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # normalization vector for each cam
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Ensure .ply files are generated and accessible
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        storePly(ply_path, xyz, rgb)  # --> store xyz, rgb into .ply file
    try:
        pcd = fetchPly(ply_path)   # --> read .ply file as BasicPointCloud
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    从COLMAP项目中提取相机的 外参 和 内参 ，以及读取该相机的所有配置参数，包括其对应的单一的图像
    :return
        cam_infos --> 一个包含所有相机信息的列表，该对象封装了一个相机的所有相关数据（包括UID、旋转矩阵、平移向量、视场角、图像路径、图像名、图像宽度和高度等）。
    """
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):

        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.

        cam_infos.append(cam_info)

    sys.stdout.write('\n')

    return cam_infos

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

def fetchPly(path):
    """
    read ply from path, extract point cloud info, it contains color, position, ...
    Wrap this into BasicPointCloud object and return
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


# READ PanopticSport PROJECT
def readPanopticSportsinfos(datadir):
    print('Reading Panoptic sports infos...')
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

    # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

def readPanopticmeta(datadir, json_path):
    print('Reading:', json_path)
    with open(os.path.join(datadir, json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    print('max_time:', max_time)
    print('length of meta:',len(test_meta['fn']))

    for index in range(len(test_meta['fn'])):

        ratio = index / float(len(test_meta['fn']))

        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir, "ims")
            image_name = fn
            image = Image.open(os.path.join(datadir, "ims", fn))
            im_data = np.array(image.convert("RGBA"))
            im_data = PILtoTorch(im_data, None)[:3, :, :]
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera": camera,
                "time": time,
                "image": im_data})

    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    return cam_infos, max_time, scene_radius

def setup_camera(w, h, k, w2c, near=0.01, far=100):

    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
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


# READ NERFFILES
def readHyperDataInfos(datadir,use_bg_points):
    train_cam_infos = Load_hyper_data(datadir, 0.5, use_bg_points, split="train")
    test_cam_infos = Load_hyper_data(datadir, 0.5, use_bg_points, split="test.py")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split = "video"
    print('befor ply_path finished')

    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info

def plot_camera_orientations(cam_list, xyz):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)





def add_points(pointsclouds, xyz_min, xyz_max):

    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
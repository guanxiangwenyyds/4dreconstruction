import numpy as np

from plyfile import PlyData
from scene.gaussian_model import BasicPointCloud
from typing import NamedTuple



class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def load_point_cloud_from_npz(npz_path):

    data = np.load(npz_path)

    points = data['points']
    colors = data['colors']
    normals = data['normals']

    # 创建BasicPointCloud实例
    point_cloud = BasicPointCloud(points=points, colors=colors, normals=normals)
    return point_cloud

def fetchPly(path):

    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
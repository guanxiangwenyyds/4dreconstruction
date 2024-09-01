import numpy as np

from plyfile import PlyData
from scene.gaussian_model import BasicPointCloud
from typing import NamedTuple



class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def load_point_cloud_from_npz(npz_path):
    # 使用numpy.load加载.npz文件
    data = np.load(npz_path)

    # 从加载的数据中提取点、颜色和法线数组
    points = data['points']   # 确保这里的键与.npz文件中的相匹配
    colors = data['colors']   # 确保这里的键与.npz文件中的相匹配
    normals = data['normals'] # 确保这里的键与.npz文件中的相匹配

    # 创建BasicPointCloud实例
    point_cloud = BasicPointCloud(points=points, colors=colors, normals=normals)
    return point_cloud

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
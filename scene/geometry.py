import numpy as np
import math


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

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


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)

    return np.float32(Rt)


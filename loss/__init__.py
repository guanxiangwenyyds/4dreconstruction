import torch
import numpy as np
import torch.nn.functional as F
import open3d as o3d
from torch.autograd import Variable
from math import exp

def build_rotation(r):
    # 通过 旋转四元数 构建旋转矩阵R
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def rigidity_loss(fg_rot_curr, fg_rot_prev, curr_offset, prev_offset, neighbor_weight):

    # 计算上一帧的旋转
    fg_rot_prev[:, 1:] = -1 * fg_rot_prev[:, 1:]
    fg_rot_prev_inv = fg_rot_prev.detach()
    rel_rot = quat_mult(fg_rot_curr, fg_rot_prev_inv)
    rot = build_rotation(rel_rot)

    # 当前帧，近邻点的偏移,并转移到前一帧的坐标系
    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

    return weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offset, neighbor_weight)



def l1_loss(img1, img2, mask=None, mw=0):

    device = img1.device

    if mw == 1:
        l1 = 0
    else:
        l1 = torch.abs((img1 - img2)).mean()

    masked_l1 = 0
    if mask is not None and mw != 0:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)  # 将 numpy 数组转换为 PyTorch 张量
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = mask.to(device)
        masked_diff = torch.abs((img1 - img2)) * mask
        masked_l1 = masked_diff.sum() / mask.sum()

    return l1 * (1-mw) + masked_l1 * mw


def mse_loss(img1, img2):

    device = img1.device
    squared_diff = (img1 - img2) ** 2
    mse = squared_diff.mean()

    return mse


def compute_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def weighted_l2_loss_v2(x, y, w):
    if isinstance(w, np.ndarray):
        w = torch.tensor(w, dtype=x.dtype, device=x.device)
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

@torch.no_grad()
def compute_psnr2(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))

    return psnr.mean().double()


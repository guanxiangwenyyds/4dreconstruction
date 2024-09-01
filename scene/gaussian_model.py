from plyfile import PlyData, PlyElement
from scene.gaussian_utils import BasicPointCloud, RGB2SH, inverse_sigmoid, mkdir_p, build_rotation
from scene.math_utils import build_scaling_rotation, strip_symmetric
from scene.gaussian_utils import SH2RGB
from scene.dataset import Delta

import os
import torch
import torch.nn as nn
import numpy as np
from simple_knn._C import distCUDA2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianModel(nn.Module):

    def __init__(self):
        print('Create Gaussian Model')
        super(GaussianModel, self).__init__()
        self.percent_dense = 0.01
        self.max_sh_degree = 3

        self.xyz = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.features_dc = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.features_rest = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.scaling = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.rotation = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.opacity = nn.Parameter(torch.empty(0, dtype=torch.float32, requires_grad=True))
        self.setup_functions()

        self.pos_encodings = None
        self.max_radii2D = torch.empty(0).to(device)
        self.xyz_gradient_accum = torch.empty(0).to(device)
        self.denom = torch.empty(0).to(device)


    def forward(self):
        return {
            'xyz': self.xyz,
            'features_dc': self.features_dc,
            'features_rest': self.features_rest,
            'scaling': self.scaling,
            'rotation': self.rotation,
            'opacity': self.opacity
        }

    def is_fg_point(self, path):
        init_pt_cld = np.load(path)["data"]
        seg = init_pt_cld[:, 6]
        seg_color = np.stack((seg, np.zeros_like(seg), 1 - seg), -1)
        self.is_fg = seg_color[:, 0] > 0.5
        return self.is_fg


    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        # print('features_dc', features_dc.shape)
        # print('features_rest', features_rest.shape)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation)

    def setup_functions(self):
        # initialize mathematical model and activate function
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

    def load_ply(self, path):
        print('Load Gaussian Points with ply... ...')
        try:
            plydata = PlyData.read(path)
        except Exception as e:
            print(f"Error reading PLY file: {e}")
            return

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc(0，1，2)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # features_rest( ... ... )
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # scale
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # rotation
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def save_ply(self, path):
        ''' Save the model data as a .ply file '''

        mkdir_p(os.path.dirname(path))

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # f_dc = self.features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self.features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation),
                                    axis=1)  # -> ply file content
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        '''  --> Build a list of attributes '''
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1]*self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1]*self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def create_from_pcd(self, pcd: BasicPointCloud):
        # Get PointCloud data
        # Creating the internal state of a model from PointCloud data

        # print('reading spatial_lr_scale:')
        # self.spatial_lr_scale = spatial_lr_scale

        # get points and color
        # print('Reading points:')
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # print('Num of points:', fused_point_cloud.shape[0])

        # print('Reading fused_color:')
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())   # RGB2SH: --> Converts colour data from RGB to spherical harmonic (SH) representation
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # --> Reserve space for features according to max_sh_degree
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        # print('Shape of features:', features.shape)

        # print('Initial scaling')
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # print('Scaling shape:', scales.shape)

        # initialize rotation quaternion (1,0,0,0)
        # print('Initial rotation')
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        # print('Rotation shape:', rots.shape)

        # print('Initial opacities')
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # print('Opacities shape:', opacities.shape)

        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self.features_dc = nn.Parameter(features[:, :, 0:1].contiguous().requires_grad_(True))
        # self.features_rest = nn.Parameter(features[:, :, 1:].contiguous().requires_grad_(True))
        self.features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scaling = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def create_random_points(self):
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        self.create_from_pcd(pcd)

    def get_feature_vector(self, t=None, pe_xyz=True, pe_t=True, d_model=4, only_xyz_t=True):
        if t is None:
            t = 0.0

        num_points = self.xyz.shape[0]
        t = torch.ones((num_points, 1), device=device) * t
        xyz = self.xyz.to(device)

        opacities = self.opacity.to(device)
        scale = self.scaling.to(device)
        rotation = self.rotation.to(device)
        features_dc = self.features_dc.squeeze(1).to(device)
        features_rest = self.features_rest.view(num_points, -1).to(device)

        # print("t shape:", t.shape)
        # print("xyz shape:", xyz.shape)
        # print("opacities shape:", opacities.shape)
        # print("scale shape:", scale.shape)
        # print("rotation shape:", rotation.shape)
        # print("features_dc shape:", features_dc.shape)
        # print("features_rest shape:", features_rest.shape)

        # Apply position encoding for xyz if requested
        if pe_xyz:
            pos_encodings_xyz = self.positional_encoding_xyz(d_model)
            xyz = pos_encodings_xyz

        # Apply position encoding for time if requested
        if pe_t:
            pos_encodings_t = self.positional_encoding_t(t, d_model*3)
            t = torch.cat([t, pos_encodings_t], dim=1)  # Extend time vector with positional encoding

        # Construct the feature vector based on the given flags
        if only_xyz_t:
            fv = torch.cat([xyz, t], dim=1)
        else:
            fv = torch.cat([xyz, opacities, scale, rotation, features_dc, features_rest, t], dim=1)


        return fv

    def positional_encoding_t(self, t, d_model):

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).to(device)
        div_term = div_term.view(1, -1)

        times = t if t.dim() == 2 else t.unsqueeze(1)

        pe = torch.zeros(times.size(0), d_model, device=device)
        pe[:, 0::2] = torch.sin(times * div_term)
        pe[:, 1::2] = torch.cos(times * div_term)
        return pe

    def positional_encoding_xyz(self, d_model):

        f_operator = torch.arange(d_model, dtype=torch.float32, device=device) ** 2 * torch.pi
        gam_sin = torch.sin(self.xyz.unsqueeze(2) * f_operator)
        gam_cos = torch.cos(self.xyz.unsqueeze(2) * f_operator)

        position_encoding = torch.cat((gam_sin, gam_cos), dim=2)
        position_encoding = position_encoding.reshape(position_encoding.shape[0], -1)

        return position_encoding


    def get_feature_dim(self):
        return self.get_feature_vector().shape[1]

    def add_densification_stats(self, viewspace_point_tensor_grad, visibility_filter):
        self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor_grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify(self, extent, max_grad=0.0002):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_clone(grads, max_grad, extent)
        self.densify_split(grads, max_grad, extent)

    def densify_split(self, grads, max_grad, scene_extent, N=2):

        n_init_points = self.get_xyz.shape[0]

        # Extract points that satisfy the grad_threshold
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False) # -> selected_pts_mask一个布尔张量

        # # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if not selected_pts_mask.any():
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N, 1)
        # print('new_xyz.shape', new_xyz.shape)
        # print('new_scaling.shape', new_scaling.shape)
        # print('new_rotation.shape', new_rotation.shape)
        # print('new_features_dc.shape', new_features_dc.shape)
        # print('new_features_rest.shape', new_features_rest.shape)
        # print('new_opacity.shape', new_opacity.shape)

        self.add_new_gaussian_points(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_clone(self, grads, max_grad, scene_extent):
        ''' According to the gradients and scalar factor clone existing gaussian points
        '''

        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self.xyz[selected_pts_mask]
        new_features_dc = self.features_dc[selected_pts_mask]
        new_features_rest = self.features_rest[selected_pts_mask]
        new_opacity = self.opacity[selected_pts_mask]
        new_scaling = self.scaling[selected_pts_mask]
        new_rotation = self.rotation[selected_pts_mask]

        self.add_new_gaussian_points(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def add_new_gaussian_points(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation):
        ''' add new gaussian point into Gaussian model '''

        self.xyz = nn.Parameter(torch.cat([self.xyz, new_xyz.to(device)], dim=0), requires_grad=True)
        self.features_dc = nn.Parameter(torch.cat([self.features_dc, new_features_dc.to(device)], dim=0), requires_grad=True)
        self.features_rest = nn.Parameter(torch.cat([self.features_rest, new_features_rest.to(device)], dim=0), requires_grad=True)
        self.scaling = nn.Parameter(torch.cat([self.scaling, new_scaling.to(device)], dim=0), requires_grad=True)
        self.rotation = nn.Parameter(torch.cat([self.rotation, new_rotation.to(device)], dim=0), requires_grad=True)
        self.opacity = nn.Parameter(torch.cat([self.opacity, new_opacity.to(device)], dim=0), requires_grad=True)

        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device=device)

    def reset_opacity(self):
        new_opacity_value = torch.full_like(self.opacity.data, 0.01)
        self.opacity = torch.nn.Parameter(new_opacity_value)

    def pruning_points(self):
        opacities = self.opacity.abs()
        pass




from arguments import ModelParams, DnerfParams

import open3d as o3d
import numpy as np
import argparse


def setup_parser():
    parser = argparse.ArgumentParser()
    return parser

def filter_by_neighbors(pcd, radius=0.05, min_neighbors=10):

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    indices_to_keep = []

    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        if len(idx) >= min_neighbors:
            indices_to_keep.append(i)

    filtered_pcd = pcd.select_by_index(indices_to_keep)
    return filtered_pcd


def apply_poisson_surface_reconstruction(pcd, depth=6, width=0, scale=1.1, linear_fit=False):

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pcd.orient_normals_consistent_tangent_plane(30)

    normals = np.asarray(pcd.normals)
    if np.all(normals == 0):
        return None

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)[0]

    return poisson_mesh

def main():

    # Parameters
    parser = setup_parser()

    dnerf_configs = DnerfParams(parser)
    scene_configs = ModelParams(parser)

    args = parser.parse_args()

    dnerf_params = dnerf_configs.extract(args)
    scene_params = scene_configs.extract(args)

    scene_params.source_path = f'data/{dnerf_params.dataset}/'

    gaussian_path = f'output/result/{dnerf_params.dataset}{dnerf_params.identifier}/gaussian.ply'
    model_path = f"output/result/{dnerf_params.dataset}{dnerf_params.identifier}/model.ckpt"

    vsl_pcd = False
    vsl_mesh = True
    ps = False
    ap = True

    pcd_original = o3d.io.read_point_cloud(gaussian_path)
    filtered_pcd = filter_by_neighbors(pcd_original, radius=0.05, min_neighbors=20)

    if vsl_pcd:
        o3d.visualization.draw_geometries([filtered_pcd])

    if vsl_mesh:
        if ap:
            alpha = 0.1
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(filtered_pcd, alpha)

        if ps:
            mesh = apply_poisson_surface_reconstruction(filtered_pcd)

        if mesh:
            mesh.paint_uniform_color([1.0, 1.0, 1.0])  # 设置网格颜色为白色

            vis = o3d.visualization.Visualizer()
            vis.create_window()

            vis.add_geometry(mesh)

            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])  # 设置背景颜色为黑色
            opt.mesh_show_wireframe = True
            opt.mesh_show_back_face = True

            vis.run()
            vis.destroy_window()
        else:
            print("Error!!")


if __name__ == "__main__":
    main()


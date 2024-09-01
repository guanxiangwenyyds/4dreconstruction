from scene.scene_read import searchForMaxIteration, add_points,readColmapSceneInfo, readPanopticSportsinfos, readHyperDataInfos
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from scene.read_dnerf import readNerfSyntheticInfo
from scene.read_multiviewer import readPanopticSportsinfos
from arguments import ModelParams

import os


class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel):

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = readNerfSyntheticInfo(args.source_path, args.white_background, args.eval, args.extension)
            dataset_type = "blender"
        else:
            assert False, "Could not find dataset "

        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     print('Reading Colmap Scene Info...')
        #     # --> SceneInfo Class: pcd，cams，ply path，maxtime
        #     scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval, args.llffhold)
        #     dataset_type="colmap"
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = readNerfSyntheticInfo(args.source_path, args.white_background, args.eval, args.extension)
        #     dataset_type = "blender"
        # else:
        #     assert False, "Could not recognize scene type! Only colmap projects are supported "

        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        print("Dataset type: ", dataset_type)
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)


        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.11111111111111111111111111111111111111111111111111")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud)

    # save point cloud and deformation
    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
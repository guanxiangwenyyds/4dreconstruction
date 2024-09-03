from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process = False
        self.add_points = False
        self.extension = ".png"
        self.llffhold=8
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class DnerfParams(ParamGroup):
    def __init__(self, parser):

        self.dataset = 'mutant'
        self.identifier = '001'

        # parameters for deformation model
        self.depth = 3
        self.common_width = 128
        self.deformation_lr = 0.001

        # parameters for gaussian point group
        self.max_points = 50000
        self.pe_t = True            # positional encoding
        self.pe_xyz = True
        self.only_xyz_t = True      # if False, all attributes of gaussian(opacity,color,etc)
                                    # will be used as feature input to deformation model.
        self.if_diff_lr = True
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.xyz_lr = 0.001

        # parameters for training
        self.num_epoch = 250   # total number of epochs
        self.num_epoch_4_init = 20
        self.densification_from_iteration = 1500  # Each epoch has 150 iterations(150 images in train set).
        self.densification_till_iteration = 15000
        self.densification_inter_iteration = 300  # densification interval

        super().__init__(parser, "Dnerf Parameters")


class PanopticParams(ParamGroup):
    def __init__(self, parser):

        self.dataset = 'basketball'
        self.identifier = '001'

        # parameters for deformation model
        self.depth = 3
        self.common_width = 256

        # parameters for gaussian point group
        self.pe_t = True
        self.pe_xyz = True
        self.only_xyz_t = True
        self.if_diff_lr = None

        # parameters for training
        self.skip_init_gaussian = None
        self.num_epoch_4_init = 500  # number of epochs for initial gaussian training
        self.gaussian_lr_scheduler = 40
        self.gaussian_lr_init = 0.001
        self.gaussian_lr = 0.0001
        self.fix_gaussian = None
        self.train_deformation_model = True
        self.deformation_lr_scheduler = 1
        self.deformation_lr = 0.00016
        self.num_epoch_4_deformation = 3  # number of epochs for deformation training
        # weights for loss
        self.w_l1 = 1.0
        self.w_rg = 0.0
        self.w_ssim = 0.0
        self.w_mse = 0.0
        # We found that using densification on this dataset does not get improvement
        # but increases the time significantly, so we don't use by default.
        self.densification_inter_iteration = 30
        self.densification_from_epoch_in_coarse = 0
        self.densification_till_epoch_in_coarse = 0
        self.max_points = 200000

        # parameters for evaluation
        self.coarse_model = None
        self.test_or_train = 'test'
        # render image
        self.start_frame = 0
        self.num_frames = 150
        self.intervals_frame = 1
        self.cam_index = 0  # There are 4 diff cameras' view in test set
        super().__init__(parser, "Panoptic Parameters")
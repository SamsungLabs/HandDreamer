# Adapted from https://github.com/IDEA-Research/DreamWaltz/blob/main/core/prompt/smpl_prompt.py
import math
import os
import os.path as osp
from typing import Iterable

import cv2
import imageio
import numpy as np
import open3d as o3d
import smplx
import torch
import torch.utils.dlpack
from PIL import Image
import matplotlib
from .aist import AIST
from .point3d import *
from .pose import NeRF_data_to_standard, SE3_Mat2RT, index2pose


def mano_to_openpose(m, n_joints_per_finger=4):
    """
    convert joints from MANO format to openpose format
    """
    finger_o2m = {0: 4, 1: 0, 2: 1, 3: 3, 4: 2}
    finger_m2o = {v: k for k,v in finger_o2m.items()}
    o = np.zeros((5*n_joints_per_finger+1, 3))
    o[0] = m[0][0]
    for mfidx in range(5):
        for jidx in range(n_joints_per_finger):
            midx = 1 + mfidx*4 + jidx
            oidx = 1 + finger_m2o[mfidx]*n_joints_per_finger + jidx
            o[oidx] = m[0][midx]
    return np.expand_dims(o,axis=0)

def add_fingertips(joints, vertices):

    fingertip_idxs = [333, 444, 672, 555, 745]
    out = [joints[0][0]]
    for fidx in range(5):
      for jidx in range(4):
        if jidx < 3:
          idx = 1 + fidx*3 + jidx
          out.append(joints[0][idx])
        else:
          out.append(vertices[0][fingertip_idxs[fidx]])

    out = np.array([out])
    return out


class MySMPL(object):
    def __init__(
        self, batch_size, model_type, scaling_factor=1.0, model_path="", gender="neutral"
    ) -> None:
        # Build SMPL Model
        assert model_type in ("smpl", "smplh", "smplx", "mano", "mano_flat")
        self.scaling_factor = scaling_factor
        smpl_cfgs = {
            "model_path": model_path,
            "model_type": model_type,
            "gender": gender,
            "batch_size": batch_size,
            "num_betas": 10,
            "ext": "npz",
            "use_face_contour": False,
        }


        mano_cfgs = {
            'model_path' : model_path,
            'model_type' : 'mano',
            'use_pca' : False,
            'is_rhand' : False,
            'flat_hand_mean' : False
        }

        mano_flat_cfgs = {
            'model_path' : model_path,
            'model_type' : 'mano',
            'use_pca' : False,
            'is_rhand' : False,
            'flat_hand_mean' : True
        }

        self.model_type = model_type
        print('Model Path: %s' %model_path, self.model_type, self.scaling_factor)
        if self.model_type == 'smpl':
            self.model = smplx.create(**smpl_cfgs)
        elif self.model_type == 'mano':
            self.model = smplx.create(**mano_cfgs)
        elif self.model_type == 'mano_flat':
            self.model = smplx.create(**mano_flat_cfgs)

        self.batch_size = batch_size
        # Build Pose Prior
        #self.vp = build_human_body_prior()

    def sample_body_pose(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        body_pose = self.vp.sample_poses(num_poses=batch_size)[
            "pose_body"
        ]  # tensor with shape of (N, 21, 3)
        if self.model_type == "smpl":
            body_pose_hands = torch.zeros((batch_size, 2, 3)).to(body_pose.device)
            body_pose = torch.cat((body_pose, body_pose_hands), dim=1)
        body_pose = (
            body_pose.contiguous().view(batch_size, -1).cpu()
        )  # body_pose shape = (N, 63)
        return body_pose

    def normalize(
        self, vertices, joints, keypoints, scale=0.5, transl_mode="pelvis"
    ):  # scale=0.5
        """
        Input & Return:
            vertices: np.array, [N, 10475, 3]
            joints: np.array, [N, 127, 3]
            keypoints: np.array, [N, 18, 3]
        """
        assert vertices.ndim == 3 and vertices.shape[-1] == 3
        assert joints.ndim == 3 and joints.shape[-1] == 3
        assert keypoints.ndim == 3 and keypoints.shape[-1] == 3
        # Translation
        if transl_mode == "pelvis":
            center = joints[
                :,
                [
                    0,
                ],
                :,
            ]
        else:
            selected_indices = {
                "body": np.asarray([1, 8, 11]),
                "hip": np.asarray([8, 11]),
                "all": np.asarray([i for i in range(18)]),
            }
            center = keypoints[:, selected_indices[transl_mode], :].mean(
                axis=(0, 1), keepdims=True
            )  # shape = [1, 1, 3]
        vertices -= center
        joints -= center
        keypoints -= center
        # Scale
        if scale is not None:
            scale /= np.max(np.linalg.norm(keypoints, ord=2, axis=-1))
            vertices *= scale
            joints *= scale
            keypoints *= scale
        return vertices, joints, keypoints

    def __call__(self, body_pose=None, random_pose=True, **kwargs):
        """
        Input:
        Return:
            vertices: np.array, shape = (N, V, 3)
            joints: np.array, shape = (N, J, 3)
            keypoints: np.array, shape = (N, 18, 3)
        """
        # SMPL param sampling
        batch_size = self.batch_size
        if body_pose is None and random_pose:
            body_pose = self.sample_body_pose(batch_size=batch_size)
        # SMPL model inference

        #output = self.model(body_pose=body_pose, return_verts=True, **kwargs)
        output = self.model(return_verts=True, **kwargs)
        vertices = (
            output.vertices.detach().cpu().numpy()
        )  # Vertices shape = (N, 10475, 3)
        joints = output.joints.detach().cpu().numpy()  # Joints shape = (N, 127, 3)
        # Select keypoints
        centroid = np.mean(vertices, axis=1, keepdims=True)
        vertices = (vertices - centroid)*self.scaling_factor
        joints = (joints - centroid)*self.scaling_factor
        if (self.model_type == 'mano') or (self.model_type == 'mano_flat'):
            joints_with_tips = add_fingertips(joints, vertices)
            keypoints = mano_to_openpose(joints_with_tips)
            # print(keypoints.shape)

        
        # Normalize
        # vertices, joints, keypoints = self.normalize(vertices, joints, keypoints)
        return vertices, joints, keypoints


class _HumanScene(object):
    face_indices = (0, 14, 15, 16, 17)
    body_indices = [i for i in range(18) if i not in (0, 14, 15, 16, 17)]
    face_keypoints = ("nose", "r_eye", "l_eye", "r_ear", "l_ear")

    def __init__(self, model_type, offset_y=0.25) -> None:
        self.model_type = model_type
        self.num_joints = 23 if model_type == "smpl" else 21
        self.offset_y = offset_y

    def add_offset_to_smpl_params(self, smpl_params):
        if "transl" in smpl_params:
            transl_offset = torch.zeros_like(smpl_params["transl"])
            transl_offset[..., 1] += self.offset_y
            smpl_params["transl"] += transl_offset  # [B, 3]
        else:
            transl_offset = torch.zeros_like(smpl_params["body_pose"])[..., :3]
            transl_offset[..., 1] += self.offset_y
            smpl_params["transl"] = transl_offset
        return smpl_params

    def build_scene(self):
        meshs = []
        ray_casting_scene = o3d.t.geometry.RaycastingScene()
        for each_vertices in self.vertices:
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(each_vertices),
                triangles=o3d.utility.Vector3iVector(self.triangles),
            )
            mesh.compute_vertex_normals()
            meshs.append(mesh)
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            ray_casting_scene.add_triangles(mesh_t)
        return meshs, ray_casting_scene

    def export_depth_map(
        self, intrinsics, extrinsic, width=512, height=512, inverse=True, normalize=True
    ):
        """
        Input:
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        """
        # Rays are 6D vectors with origin and ray direction.
        # Here we use a helper function to create rays for a pinhole camera.
        rays = self.ray_casting_scene.create_rays_pinhole(
            intrinsics, extrinsic, width_px=width, height_px=height
        )

        # Compute the ray intersections.
        ans = self.ray_casting_scene.cast_rays(rays)
        depth = ans["t_hit"].numpy()

        # Inverse and Normalize
        if inverse:
            depth = 1.0 / depth
        if normalize:
            depth -= np.min(depth)
            depth /= np.max(depth)
        image = np.asarray(depth * 255.0, np.uint8)
        image = np.stack([image, image, image], axis=2)
        return Image.fromarray(image)

    def export_pose_map(
        self, intrinsics, extrinsic, width=512, height=512, occlusion_culling=True, model_type='mano', counter=0
    ):
        """
        Input:
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        Variable:
            self.keypoints: np.array, [N, K, 3]
        """
        # Init
        N, K, _ = self.keypoints.shape # 21x3
        R, T = SE3_Mat2RT(extrinsic) # 4x4 to 3x3 and 3x1
        # Transform
        kp_world = self.keypoints.reshape(-1, 3)
        kp_camera = transform_keypoints_to_novelview(kp_world, None, None, R, T)
        kp_image = project_camera3d_to_2d(kp_camera, intrinsics)  # [N*18, 2]
        kp_image = kp_image.reshape(N, K, 2)  # [N, 18, 2]

        # print("keypoints 2D: ")
        # print(kp_image)

        # Mesh Visualization code
        # mesh_v = self.vertices[0]
        # mesh_v_camera = transform_keypoints_to_novelview(mesh_v, None, None, R, T)
        # mesh_v_image = project_camera3d_to_2d(mesh_v_camera, intrinsics)
        # mesh_v_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # for key in mesh_v_image:
        #     if np.isnan(key).any():
        #         continue
        #     x, y = key[0], key[1]
        #     x = int(x)
        #     y = int(y)
        #     cv2.circle(mesh_v_canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        # mesh_v_canvas = Image.fromarray(mesh_v_canvas)
        # mesh_v_canvas.save('debug/check_pose_%d_mesh.png' %(counter))

        # Keypoint Dumping Code
        # db = {}
        # db["hand_kps"] = kp_camera.tolist()
        # path = os.path.join(os.getcwd(), 'kps_dump', 'kps_'+str(counter).zfill(4)+'.json')
        # with open(os.path.join('kps_dump', 'kps_'+str(counter).zfill(4)+'.json'), 'w') as f:
        #     json.dump(db, f)


        # Occlusion
        if occlusion_culling:
            CAM_world = np.dot(np.linalg.inv(R), -T)
            occluded = self.detect_occlusion(CAM_world)  # [N, 18]
            kp_image[occluded, :] = None
        # Draw
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if 'mano' in model_type:
            # print("calling draw_handpose")
            image = self.draw_handpose(canvas, kp_image[0])  #Assumong batch size = 1 as assumed throughout
        else:
            image = self.draw_bodypose(canvas, kp_image)
        return image

    def export_mesh_map(
        self, intrinsics, extrinsic, width=512, height=512, device=None
    ):
        import pytorch3d
        import pytorch3d.renderer
        from scipy.spatial.transform import Rotation

        """ Render the mesh under camera coordinates
        vertices: (N_v, 3), vertices of mesh
        faces: (N_f, 3), faces of mesh
        translation: (3, ), translations of mesh or camera
        focal: float, focal length of camera
        height: int, height of image
        width: int, width of image
        device: "cpu"/"cuda:0", device of torch
        :return: the rgba rendered image
        """

        if device is None:
            device = torch.device("cuda")

        vertices = torch.from_numpy(self.vertices[0]).to(device)
        faces = torch.from_numpy(self.triangles.astype(np.int64)).to(device)
        # translation = self.smpl_params["transl"].to(device)

        # print(vertices.shape)     torch.Size([6890, 3])
        # print(faces.shape)        torch.Size([13776, 3])
        # print(translation.shape)  torch.Size([1, 3])

        # add the translation
        # vertices = vertices + translation

        # upside down the mesh
        # rot = Rotation.from_euler("z", 180, degrees=True).as_matrix().astype(np.float32)
        # rot = torch.from_numpy(rot).to(device)

        # vertices = torch.matmul(rot, vertices.T).T

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
        textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
        mesh = pytorch3d.structures.Meshes(
            verts=[vertices], faces=[faces], textures=textures
        )

        focal = intrinsics[0][0].item()

        R = torch.from_numpy(extrinsic[np.newaxis, :3, :3]).to(device)
        T = torch.from_numpy(extrinsic[np.newaxis, :3, 3]).to(device)
        R = R.transpose(1, 2)  # column
        R[:, :, 0:2] = -R[:, :, 0:2]  # x = -x, y = -y
        T[:, 0:2] = -T[:, 0:2]  # x = -x, y = -y

        if not hasattr(self, "mesh_renderer"):
            # Define the settings for rasterization and shading.
            raster_settings = pytorch3d.renderer.RasterizationSettings(
                # image_size=(height, width),   # (H, W)
                image_size=height,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Define the material
            materials = pytorch3d.renderer.Materials(
                ambient_color=((1, 1, 1),),
                diffuse_color=((1, 1, 1),),
                specular_color=((1, 1, 1),),
                shininess=64,
                device=device,
            )

            # Place a directional light in front of the object.
            lights = pytorch3d.renderer.DirectionalLights(
                device=device, direction=((0, 2, 3),)
            )
            # lights = pytorch3d.renderer.AmbientLights(ambient_color=((1.0, 1.0, 1.0),), device=device)

            # Create a phong renderer by composing a rasterizer and a shader.
            renderer = pytorch3d.renderer.MeshRenderer(
                rasterizer=pytorch3d.renderer.MeshRasterizer(
                    raster_settings=raster_settings
                ),
                shader=pytorch3d.renderer.SoftPhongShader(
                    device=device, lights=lights, materials=materials
                ),
            )

            self.mesh_renderer = renderer

        # Initialize a camera.
        # R: Rotation matrix of shape (N, 3, 3)
        # T: Translation matrix of shape (N, 3)
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=(
                (2 * focal / min(height, width), 2 * focal / min(height, width)),
            ),
            R=R,
            T=T,
            image_size=((height, width),),
            device=device,
        )

        # Do rendering
        color_batch = self.mesh_renderer(mesh, cameras=cameras)  # [1, 512, 512, 4]

        # To Image
        valid_mask_batch = color_batch[:, :, :, [-1]] > 0
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        input_img = np.zeros_like(color[:, :, :3])
        alpha = 1.0
        image_vis = (
            alpha * color[:, :, :3] * valid_mask
            + (1 - alpha) * input_img * valid_mask
            + (1 - valid_mask) * input_img
        )

        image_vis = image_vis.astype(np.uint8)

        image = Image.fromarray(image_vis, mode="RGB")
        return image

    def export_distance(self, query_points: torch.Tensor, signed=True):
        """
        Input:
            query_points: torch.Tensor, [..., 3]
        Return:
            distances: torch.Tensor, [...]
        """
        if isinstance(query_points, torch.Tensor):
            device = query_points.device
            query_points = o3d.core.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(query_points.detach().cpu())
            )
        if signed:
            distances = self.ray_casting_scene.compute_signed_distance(query_points)
        else:
            distances = self.ray_casting_scene.compute_distance(query_points)
        distances = torch.utils.dlpack.from_dlpack(distances.to_dlpack()).to(device)
        return distances

    def export_density(self, query_points: torch.Tensor, a=0.001):
        def inv_softplus(bias):
            """Inverse softplus function.
            Args:
                bias (float or tensor): the value to be softplus-inverted.
            """
            is_tensor = True
            if not isinstance(bias, torch.Tensor):
                is_tensor = False
                bias = torch.tensor(bias)
            out = bias.expm1().clamp_min(1e-6).log()
            if not is_tensor and out.numel() == 1:
                return out.item()
            return out

        distances = self.export_distance(query_points)
        # print(torch.min(distances), torch.mean(distances), torch.max(distances))
        density = torch.sigmoid(-distances / a) / a  # [0, 1000]
        # t = torch.sigmoid(- distances / a) / a  # [0, 1000]
        # density = torch.clamp(inv_softplus(t), min=0.0, max=1/a)

        return density

    def detect_occlusion(self, CAM_world, thres=0.02):
        """
        Input:
            CAM_world: np.array, [3, 1], camera position in world coordinates
        Return:
            occluded: np.array, bool, [N, 18]
        """
        KP_world = self.keypoints  #world kps                                # [N, 21, 3]
        CAM_world = np.broadcast_to(CAM_world.T, (*(KP_world.shape[:-1]), 3))  # [N, 21, 3]
        t_far = np.linalg.norm(KP_world - CAM_world, ord=2, axis=2)            # [N, 21]

        O_xyz = CAM_world
        D_xyz = KP_world - CAM_world
        D_xyz /= np.linalg.norm(D_xyz, ord=2, axis=2, keepdims=True)  # [N, 21, 3]
        rays = np.concatenate((O_xyz, D_xyz), axis=2)   # [N, 21, 6]  ## (origin, direction)

        outputs = self.ray_casting_scene.cast_rays(np.asarray(rays, dtype=np.float32))
        t_hit = outputs['t_hit'].numpy()                # [N, 21]
        geometry_ids = outputs['geometry_ids'].numpy()  # [2, 21], int
        occluded = (t_far - t_hit) > thres      # [N, 21], bool
        return occluded

    @staticmethod
    def draw_bodypose(canvas, keypoints_2d):
        """
        canvas = np.zeros_like(input_image), np.array, [H x W x 3]
        keypoints_2d: np.array, [N, 18, 2], N is the number of people
        """
        stickwidth = 4
        limbSeq = [
            [2, 3],
            [2, 6],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [2, 9],
            [9, 10],
            [10, 11],
            [2, 12],
            [12, 13],
            [13, 14],
            [2, 1],
            [1, 15],
            [15, 17],
            [1, 16],
            [16, 18],
            [3, 17],
            [6, 18],
        ]
        colors = [
            [255, 0, 0],
            [255, 85, 0],
            [255, 170, 0],
            [255, 255, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [85, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [255, 0, 170],
            [255, 0, 85],
        ]
        assert keypoints_2d.shape[1] == 18 and keypoints_2d.ndim in (2, 3)
        if keypoints_2d.ndim == 2:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
        N = keypoints_2d.shape[0]
        for p in range(N):
            # draw points
            for i in range(18):
                x, y = keypoints_2d[p, i]
                if is_nan(x) or is_nan(y):
                    continue
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
            # draw lines
            for i in range(17):
                indices = np.array(limbSeq[i]) - 1
                cur_canvas = canvas.copy()
                X = keypoints_2d[p, indices, 1]
                Y = keypoints_2d[p, indices, 0]
                if is_nan(Y[0]) or is_nan(Y[1]) or is_nan(X[0]) or is_nan(X[1]):
                    continue
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)),
                    (int(length / 2), stickwidth),
                    int(angle),
                    0,
                    360,
                    1,
                )
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return Image.fromarray(canvas)

    def export_geometry(self, plot_joints=True):
        # Export Geometry for Visualization
        geometry = self.meshs.copy()
        if plot_joints:
            for joints in self.joints:
                joints_pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(joints))
                joints_pcl.paint_uniform_color([1.0, 0.0, 0.0])
                geometry.append(joints_pcl)
        return geometry
        # import open3d.web_visualizer
        # open3d.web_visualizer.draw(geometry)

    def export_mesh_to_file(self, filename):
        o3d.io.write_triangle_mesh(
            str(filename), self.meshs[0], write_triangle_uvs=False
        )

    @staticmethod
    def draw_handpose(canvas, keypoints):
       
        eps = 0.01
        H, W, C = canvas.shape
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
                 [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
        for ie, (e1, e2) in enumerate(edges):
            k1 = keypoints[e1]
            k2 = keypoints[e2]
            if np.isnan(k1).any() or np.isnan(k2).any():
                continue
            
            x1 = int(k1[0])
            y1 = int(k1[1])
            x2 = int(k2[0])
            y2 = int(k2[1])
            #if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)
        for keypoint in keypoints:
            if np.isnan(keypoint).any():
                continue
            x, y = keypoint[0], keypoint[1]
            x = int(x)
            y = int(y)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        return Image.fromarray(canvas)


class CanonicalScene(_HumanScene):
    def __init__(self, scene, model_type="mano", scaling_factor=1.0, mano_path="", **kwargs) -> None:
        super().__init__(model_type=model_type, **kwargs)
        # Load data from 3DPW dataset
        self.num_person = 1
        self.scaling_factor = scaling_factor
        self.mano_path = mano_path
        smpl_params = {
            "betas": torch.zeros((1, 10)),
            "body_pose": torch.zeros((1, self.num_joints * 3)),
            "global_orient": torch.zeros((1, 3)),
        }
        mano_flat_params = {
            "betas": torch.zeros([1, 10]),
            "body_pose": torch.zeros((1, self.num_joints * 3)),
            "hand_pose": torch.zeros([1,45]),
            "global_orient": torch.tensor([[0.0, 0.0, 1.57]]),   #0,0,-0.34 - slightly tilted, pi/2 for uright for upright
            "transl":torch.zeros([1,3])
        }
        
        finger_index_dict = {
                'index': [0,3,6],
                'middle': [9,12,15],
                'ring': [27,30,33],
                'pinky': [18,21,24],
                'thumb':[36,39,42]
                }
        #mano_flat_params["hand_pose"][0,finger_index_dict['index'][0] + 2] = -0.34
        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][1] + 2] = -1.4
        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][2] + 2] = -0.48

        mano_flat_params["hand_pose"][0,finger_index_dict['thumb'][0] + 0] = 1.2
        mano_flat_params["hand_pose"][0,finger_index_dict['thumb'][2] + 2] = -0.47


        #mano_flat_params["hand_pose"][0,finger_index_dict['middle'][0] + 2] = -1.57
        #mano_flat_params["hand_pose"][0,finger_index_dict['middle'][1] + 1] = -0.52
        #mano_flat_params["hand_pose"][0,finger_index_dict['middle'][1] + 2] = -1.57
 
        #mano_flat_params["hand_pose"][0,finger_index_dict['ring'][0] + 1] = -0.25
        #mano_flat_params["hand_pose"][0,finger_index_dict['ring'][0] + 2] = -1.57
        #mano_flat_params["hand_pose"][0,finger_index_dict['ring'][1] + 0] = -0.87
        #mano_flat_params["hand_pose"][0,finger_index_dict['ring'][1] + 1] = -0.26
        #mano_flat_params["hand_pose"][0,finger_index_dict['ring'][1] + 2] = -1.22

        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][0] + 1] = -1.04
        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][0] + 2] = -1.57
        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][1] + 0] = -1.04
        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][1] + 1] = -1.04
        #mano_flat_params["hand_pose"][0,finger_index_dict['pinky'][1] + 2] = -1.04

        mano_flat_params["hand_pose"][0,finger_index_dict['index'][0] + 2] = -1.0
        mano_flat_params["hand_pose"][0,finger_index_dict['index'][1] + 2] = -1.0


        if scene == "canonical-T":
            body_pose = smpl_params["body_pose"].reshape(1, self.num_joints, 3)
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +1.0])
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -1.0])
            smpl_params["body_pose"] = body_pose.reshape(1, 63)
        elif scene == "canonical-A":
            body_pose = smpl_params["body_pose"].reshape(1, self.num_joints, 3)
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi / 4])
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +np.pi / 4])
            smpl_params["body_pose"] = body_pose.reshape(1, -1)
        # Set SMPL Params
        smpl_params = self.add_offset_to_smpl_params(smpl_params)
        self.smpl_params = smpl_params
        # SMPL Model
        smpl = MySMPL(batch_size=self.num_person, model_type=model_type, scaling_factor=self.scaling_factor, model_path=self.mano_path)
        self.smpl = smpl
        self.triangles = smpl.model.faces
        if model_type == "mano_flat":
            self.vertices, self.joints, self.keypoints = smpl(**mano_flat_params)
        else:
            self.vertices, self.joints, self.keypoints = smpl(**smpl_params)  #this is for both default mano and smpl
        # Build Scene
        self.meshs, self.ray_casting_scene = self.build_scene()

    def set_frame_index(self, frame_idx):
        pass


# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #
class SMPLPrompt:
    def __init__(
        self,
        cond_type,
        scene="canonical-A",
        smpl_offset_y=0.25,
        num_person=1,
        height=512,
        width=512,
        model_type='mano',
        mesh_scaling_factor=1.0,
        mano_root=""
    ):
        # Init
        self.cond_type = cond_type
        self.height, self.width = height, width
        self.model_type = model_type
        self.occlusion_culling = False
        print('Using model type {}'.format(self.model_type))
        # Build Scene
        if scene.startswith("canonical"):
            self.hs = CanonicalScene(scene=scene, model_type=model_type, scaling_factor=mesh_scaling_factor, mano_path=mano_root, offset_y=smpl_offset_y)
        else:
            assert 0, scene
        self.num_person = self.hs.num_person
        self.callCtr = 0

    def __call__(self, intrinsics, cam2world, cond_type=None, frame_idx=None):
        """
        Input:
            intrinsics: shape = [4, ]
            cam2world: shape = [N, 4, 4]
            cond_type: List[str]
        Return:
            cond_images: list of [PIL.Image]
        """
        if cond_type is None:
            cond_type = self.cond_type
        if isinstance(cond_type, str):
            cond_type = [
                cond_type,
            ]
        intrinsics, extrinsic = NeRF_data_to_standard(
            intrinsics, cam2world, H=self.height, W=self.width
        )
        cond_images = []
        depth_images = []
        for _cond in cond_type:
            if _cond == 'pose' or 'pose' in _cond:
                if (self.model_type == 'mano') or (self.model_type == 'mano_flat'):
                    cond_image = self.hs.export_pose_map(intrinsics, extrinsic, width=self.width, height=self.height, occlusion_culling=self.occlusion_culling, model_type=self.model_type, counter=self.callCtr)
                    depth_image = self.hs.export_depth_map(intrinsics, extrinsic, width=self.width, height=self.height)
                else:
                    cond_image = self.hs.export_pose_map(intrinsics, extrinsic, width=self.width, height=self.height)
            elif _cond == "depth":
                depth_image = self.hs.export_depth_map(intrinsics, extrinsic, width=self.width, height=self.height)
                cond_image = self.hs.export_pose_map(intrinsics, extrinsic, width=self.width, height=self.height, occlusion_culling=self.occlusion_culling, model_type=self.model_type, counter=self.callCtr)
            elif _cond == "mesh":
                cond_image = self.hs.export_mesh_map(intrinsics, extrinsic, width=self.width, height=self.height)
                depth_image = self.hs.export_depth_map(intrinsics, extrinsic, width=self.width, height=self.height)
            else:
                assert 0, _cond
            cond_images.append(cond_image)
            depth_images.append(depth_image)

        if (self.model_type == 'mano') or (self.model_type == 'mano_flat'):
            return cond_images, depth_images
        else:
            return cond_images

    def write_video(
        self,
        save_dir="./",
        save_image="output.png",
        save_video="output.mp4",
        cond_type=None,
    ):
        import os
        import os.path as osp

        os.makedirs(save_dir, exist_ok=True)
        images = []
        for i in range(100):
            intrinsics, cam2world = index2pose(i, H=self.height, W=self.width)
            image = self(intrinsics, cam2world, cond_type=cond_type)[0]
            if i == 0:
                image.save(osp.join(save_dir, save_image))
            images.append(np.array(image))
        imageio.mimsave(
            osp.join(save_dir, save_video),
            np.array(images),
            fps=25,
            quality=8,
            macro_block_size=1,
        )
        return image




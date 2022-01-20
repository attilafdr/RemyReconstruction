#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Attila Fodor
# Date: 14 Jan 2022

"""Creates a reconstruction of the scene by reprojecting the image-depthmap pairs using the camera calibration
 parameters and camera position information.

The code makes several attempts to improve the depthmaps, the camera positions and the final alignment of the
 pointclouds, with (very) limited success. These techniques are based on some form of block matching that don't
 work particularly well in the example scene, due to the low camera resolution, featureless surface, glossy object
 and generally low quality images.

While this was anticipated, not starting with these steps makes little sense, as more sophisticated methods don't
 guarantee success either. To rule out the possibility that this was due to bad implementation I used Colmap to
 create a baseline reconstruction, which was not better at all, as Colmap failed to align about half of the camera
 images. In a practical application the scene can be improved by better lighting, using a higher resolution camera(s),
 or using learned approaches.

I haven't explored these as it's difficult to estimate how much time it takes to get them to work on a custom dataset,
 even though these might vastly improve the reconstruction accuracy. One notable work is [RoutedFusion: Learning
 Real-time Depth Map Fusion](https://www.silvanweder.com/publications/routed-fusion/) that proposes the fusion of
 depth maps loaded with non-gaussian noise, which could prepare the scene for an ICP step very well. Learned 3D
 reconstruction from images has improved a lot on the past few years as well, from
 [Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction](https://arxiv.org/pdf/1706.07036.pdf)
 to [3D Reconstruction of Novel Object Shapes from Single Images](https://arxiv.org/pdf/2006.07752.pdf), which could
 work exceptionally well on simple, featureless shapes compared to classical reconstruction methods.

The main() function demonstrates the usage of the code, and the class/function docstrings explain the reasoning behind
 the chosen approaches and their implementation details.
"""
import cv2
import open3d as o3d
import numpy as np

from copy import copy
from time import sleep
from itertools import combinations
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tqdm import tqdm

import numpy.typing as npt
from typing import List, Tuple


class PointCloudAligner(object):
    """
    Takes color images, depth maps and camera positions and creates an aligned reprojection of pointcloud fragments
    """
    def __init__(self, cam_calib: o3d.camera.PinholeCameraIntrinsic) -> None:
        """
        Inits a PointCloudAligner with camera calibration parameters.

        Args:
            cam_calib: Intrinsic camera calibration parameters without distortions
        """
        self.cam_calib = cam_calib

        self.vis = None
        self.pcd = o3d.geometry.PointCloud()

    def reproject(self, image: npt.NDArray[np.uint8], depth_map: npt.NDArray[np.uint16],
                  translation: npt.NDArray[np.float64] = None, rotation: npt.NDArray[np.float64] = None,
                  transformation: npt.NDArray[np.float64] = None) -> o3d.geometry.PointCloud:
        """
        Reproject a depth map and image to a colored 3D pointcloud, rotates it to position.

        Reprojection uses the object camera calibration parameters.
        If transformation parameter is provided, rotation and translation has no effect. If neither transformation,
         rotation or translation is provided, assumes identity transformation.
        Both stores the pointcloud in self and returns it as a new object - both can be used depending on the
         visualisation requirements (self supports updating, new object supports adding)

        Args:
            image: RGB color image
            depth_map: Depth map corresponding to the color image
            transformation: 4x4 combined rotation and translation matrix
            translation: 3x1 translation vector
            rotation: 3x1 rotation vector in axis-angle representation. Rotation angle equals to vector norm.

        Returns:
            Reprojected and transformed pointcloud.
        """
        _image = o3d.geometry.Image(image)
        _depth_map = o3d.geometry.Image(depth_map)
        depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(image),
                                                                         depth=o3d.geometry.Image(depth_map),
                                                                         depth_scale=1000, depth_trunc=0.6,
                                                                         convert_rgb_to_intensity=True)

        # Build transformation matrix
        if transformation is not None:
            t_mtx = transformation
        else:
            t_mtx = np.identity(4)
            if rotation is not None:
                t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation)
            if translation is not None:
                t_mtx[:3, 3] = translation

        _pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=depth_image,
                                                              intrinsic=self.cam_calib)

        # Apply transformation
        _pcd.transform(t_mtx)

        # Need to track the pcd in the same pointer otherwise the visualisation update doesn't work...
        self.pcd.points = _pcd.points
        self.pcd.colors = _pcd.colors

        return _pcd

    def visualise(self, pcd: o3d.geometry.PointCloud = None, update: bool = False) -> None:
        """
        Opens a window and visualises the pointcloud.

        Args:
            pcd: The pointcloud to visualise.
            update: If True, updates the visualisation of the pointcloud stored in self.pcd
        """
        if not self.vis:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
        else:
            if pcd:
                self.vis.add_geometry(pcd)
            if update:
                self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self) -> None:
        """
        Convenience function to close the window if required
        """
        self.vis.destroy_window()


def reconstruct(pose_mat: npt.NDArray[np.float64], rot_mat: npt.NDArray[np.float64], color_img_list: List[str],
                depth_img_list: List[str], cam_calib: o3d.geometry.PointCloud,
                visualise: bool = False) -> o3d.geometry.PointCloud:
    """
    Feeds images and corresponding position data to a PointCloudAligner object to reconstruct a scene using the
     available data.

    Args:
        pose_mat: nx3 array of translation vectors
        rot_mat: nx3 array of rotation vectors in axis-angle representation. Rotation angle equals to vector norm
        color_img_list: List of strings to the paths of the color images
        depth_img_list: List of strings to the paths of the depth images
        cam_calib: Intrinsic camera calibration parameters without distortions
        visualise: Open a window and visualise the resulting pointcloud

    Returns:
        The aligned pointcloud
    """
    aligner = PointCloudAligner(cam_calib=cam_calib)
    pcd = o3d.geometry.PointCloud()

    for i, (color_path, depth_path) in enumerate(zip(color_img_list, depth_img_list)):
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        pcd = aligner.reproject(image=color, depth_map=depth, rotation=rot_mat[i, :], translation=pose_mat[i, :])
        if visualise:
            aligner.visualise(pcd)

    if visualise:
        aligner.vis.run()

    return pcd


def color_icp():
    def get_remy_pcd(i, cam_calib, pose_mat, rot_mat):
        t_mtx = np.identity(4)
        t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_mat[i, :])
        t_mtx[:3, 3] = pose_mat[i, :]
        color = cv2.imread(f'/home/attila/Datasets/Remy/rgb-{str(i).zfill(5)}.png', cv2.IMREAD_COLOR)
        depth = cv2.imread(f'/home/attila/Datasets/Remy/depth-{str(i).zfill(5)}.png', cv2.IMREAD_UNCHANGED)
        image = o3d.geometry.Image(color)
        depth_map = o3d.geometry.Image(depth)
        depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(image),
                                                                         depth=o3d.geometry.Image(depth_map),
                                                                         depth_scale=1000, depth_trunc=0.6,
                                                                         convert_rgb_to_intensity=True)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=depth_image,
                                                             intrinsic=cam_calib)
        pcd.transform(t_mtx)

        return pcd

    """Register each new RGBD frame with colored ICP to the existing pointcloud"""
    cam_calib = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                  fx=613.688, fy=614.261,
                                                  cx=323.035, cy=242.229)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)

    pose_data = np.genfromtxt('/home/attila/Datasets/Remy/scanner.log', delimiter=' ', usecols=range(2, 8))
    rot_mat = pose_data[:, 3:]
    pose_mat = pose_data[:, 0:3]

    # Load im 0 as target
    start = 0
    pcd_target = get_remy_pcd(start, cam_calib, pose_mat, rot_mat)
    icp_transforms = [np.identity(4)] * 195
    vis.add_geometry(pcd_target, reset_bounding_box=True)

    # Load each image sequentially and calculate the transformation with ICP
    for i in range(start, 195):
        pcd_move = get_remy_pcd(i, cam_calib, pose_mat, rot_mat)
        #pcd_move.transform(icp_transforms[i-1])
        try:
            for scale in range(3):
                iter = max_iter[scale]
                radius = voxel_radius[scale]
                source_down = pcd_move.voxel_down_sample(radius)
                target_down = pcd_target.voxel_down_sample(radius)
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                      relative_rmse=1e-6,
                                                                      max_iteration=iter))
                current_transformation = result_icp.transformation
        except RuntimeError:
            # Open3d raises a RuntimeError exception if there are not enough points to perform the ICP.
            # In this case we default to an identity transformation
            current_transformation = np.identity(4)

        # Convert the rotation matrix to axis-angle and calculate the rotation angle in degrees.
        # ICP should only slightly modify the angle, therefore if the calculated rotation angle is large,
        # it's probably incorrect and better to discard it entirely.
        angle = np.linalg.norm(Rotation.from_matrix(copy(current_transformation)[:3, :3]).as_rotvec(degrees=True))
        print(angle)
        if angle < 4:
            pcd_move.transform(current_transformation)
            icp_transforms[i] = current_transformation @ icp_transforms[i - 1]

        pcd_target = pcd_move

        vis.add_geometry(pcd_move, reset_bounding_box=True)
        vis.poll_events()
        vis.update_renderer()
        sleep(0.01)

    vis.run()


class KeypointTracker(object):
    """

    """
    def __init__(self, ) -> None:

        self.mappoints = np.empty(shape=(0, 3))
        self.observations = np.empty(shape=(0, 4))

    @staticmethod
    def _track_keypoints(img_list: List[str]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """

        :param img_list:
        :return:
        """
        # Generate key points and descriptors with ORB for all images and store in a list of dictionaries
        length = len(img_list)
        detector = cv2.ORB_create()
        min_n_features = 300

        mappoints = np.empty(shape=(0, 3))
        observations = np.empty(shape=(0, 4))
        kps_ref = np.empty(shape=(0, 2), dtype=np.float32)

        # Read first image
        img_ref = cv2.imread(img_list[0], cv2.IMREAD_COLOR)

        for ref_img_no, img_path in enumerate(tqdm(img_list[1:], total=length - 1,
                                                   desc='Keypoint extraction'),
                                              start=0):
            img_track = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # If the number of tracked keypoints is under the set threshold,
            # generate new ones and append to the existing tracked keypoints
            if kps_ref.shape[0] < min_n_features:
                kps_new = detector.detect(image=img_ref, mask=None)
                kps_new = np.array([x.pt for x in kps_new], dtype=np.float32)
                kps_ref = np.vstack([kps_ref, kps_new])

            # Calculate sparse optical flow for the reference keypoints
            kps_track, status, error = cv2.calcOpticalFlowPyrLK(prevImg=img_ref,
                                                                nextImg=img_track,
                                                                prevPts=kps_ref,
                                                                nextPts=None,
                                                                winSize=(21, 21),
                                                                criteria=(cv2.TERM_CRITERIA_EPS |
                                                                          cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            # Keep the valid keypoints pairs and eliminate any duplications
            kps_ref = kps_ref[status.ravel() == 1]
            _, ref_unique_idx = np.unique(kps_ref, return_index=True, axis=0)
            _, track_unique_idx = np.unique(kps_track, return_index=True, axis=0)
            kps_ref = kps_ref[np.intersect1d(ref_unique_idx, track_unique_idx)]
            kps_track = kps_track[np.intersect1d(ref_unique_idx, track_unique_idx)]

            """
            At this point a VO would be looking for the Essential/Fundamental matrix and recover the pose change,
            but we already have that approximate information, so skip this step and log the observations and mappoints.
            """

            # For each keypoint find the mappoint if it already has one associated with it when it was tracked before,
            # or create a new
            for kp_ref, kp_track in zip(kps_ref, kps_track):
                mappoint_id = observations[np.where((observations[:, 1] == ref_img_no) *
                                                    (observations[:, 2] == kp_ref[0]) *
                                                    (observations[:, 3] == kp_ref[1])), 0]

                if mappoint_id.size > 0:
                    # If the ref keypoint is in the list of existing observations,
                    # add the track keypoint only to the same mappoint

                    # Mappoint id, reference image id, x, y
                    observations = np.vstack([observations,
                                              np.array([float(mappoint_id), ref_img_no + 1,
                                                        kp_track[0], kp_track[1]], ndmin=2)])

                else:
                    # Add a new mappoint and the two observations
                    observations = np.vstack([observations,
                                              np.array([mappoints.shape[0], ref_img_no, kp_ref[0], kp_ref[1]], ndmin=2),
                                              np.array([mappoints.shape[0], ref_img_no + 1, kp_track[0], kp_track[1]],
                                                       ndmin=2)
                                              ])

                    mappoints = np.vstack([mappoints, np.zeros(shape=(1, 3))])

            observations = np.unique(np.array(sorted(observations, key=lambda x: (x[0]))), axis=0)

            # Step the references
            img_ref = img_track
            kps_ref = kps_track

        return observations, mappoints

    @staticmethod
    def _extract_keypoints(img_list):
        # Generate key points and descriptors with ORB for all images and store in a list of dictionaries
        length = len(img_list)
        orb = cv2.ORB_create()

        # features = [{'key_points': [],
        #             'descriptors': np.empty((0, 32))} for i in range(length)]  # Initialise a list of dictionaries

        # im_no, x, y, des
        keypoints = np.empty(shape=(0, 35))
        kp_starts = [0]  # Store the first keypoint index for the corresponding image no.

        for i in tqdm(range(0, length), desc='Keypoint extraction'):
            color = cv2.imread(img_list[i], cv2.IMREAD_COLOR)
            kp = orb.detect(color, None)
            kp, des = orb.compute(color, kp)

            # features[i]['key_points'] = list(kp)

            keypoints = np.vstack([keypoints, np.hstack([i * np.ones(shape=(len(kp), 1)),
                                                         np.array([p.pt for p in kp]),
                                                         des])])
            kp_starts.append(kp_starts[-1] + len(kp))

        return keypoints, kp_starts

    @staticmethod
    def _match_keypoints(keypoints, kp_starts, show_matches=False):
        length = len(kp_starts) - 1

        # Match keypoints exhaustively with a brute force matcher
        mappoints = np.empty(shape=(0, 3))  # Store the initial point position estimates
        # Store the observed mappoint id and observing image no with keypoint x, y coordinates
        observations = np.empty(shape=(0, 4))
        for i in tqdm(range(0, length - 10), desc='Keypoint matching'):
            for j in range(i + 1, i + 11):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(queryDescriptors=keypoints[kp_starts[i]: kp_starts[i + 1], 3:].astype(np.uint8),
                                   trainDescriptors=keypoints[kp_starts[j]: kp_starts[j + 1], 3:].astype(np.uint8))
                matches = sorted(matches, key=lambda x: x.distance)

                for match in matches[:50]:
                    # Add an empty mappoint
                    mappoints = np.vstack([mappoints, np.zeros(shape=(1, 3))])
                    # Add two observations
                    observations = np.vstack([observations,
                                              np.hstack([mappoints.shape[0] - 1,  # Mappoint id
                                                         np.array([i]),  # Reference image id
                                                         keypoints[kp_starts[i] + match.queryIdx][1:3]]),  # x, y
                                              np.hstack([mappoints.shape[0] - 1,  # Same mappoint id
                                                         np.array([j]),  # Compared to image id
                                                         keypoints[kp_starts[j] + match.trainIdx][1:3]])])  # x, y

        # observations = np.array(sorted(observations, key=lambda x: (x[0])))  # Key: mappoint id
        observations = np.array(sorted(observations, key=lambda x: (x[1], x[2], x[3])))
        return observations, mappoints

    @staticmethod
    def _triangulate_mappoints(observations, cam_calib, mappoints, pose_mat, rot_mat):

        for mappoint in tqdm(np.unique(observations[:, 0]), desc='Triangulating mappoints'):
            position = np.empty(shape=(0, 3))
            for obs1, obs2 in combinations(observations[observations[:, 0] == mappoint], r=2):
                t_mtx = np.zeros(shape=(3, 4))
                t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_mat[int(obs1[1])])
                t_mtx[:3, 3] = pose_mat[int(obs1[1])]
                proj_1 = cam_calib.intrinsic_matrix @ t_mtx
                points_1 = obs1[2:4].transpose()

                t_mtx = np.zeros(shape=(3, 4))
                t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_mat[int(obs2[1])])
                t_mtx[:3, 3] = pose_mat[int(obs2[1])]
                proj_2 = cam_calib.intrinsic_matrix @ t_mtx
                points_2 = obs2[2:4].transpose()

                res = cv2.triangulatePoints(projMatr1=proj_1, projMatr2=proj_2, projPoints1=points_1,
                                            projPoints2=points_2)

                position = np.vstack([position, cv2.convertPointsFromHomogeneous(res.transpose()).flatten()])

            mappoints[int(mappoint), :] = np.mean(position, axis=0)

        return mappoints

    def compute_points(self, pose_mat: npt.NDArray[np.float64], rot_mat: npt.NDArray[np.float64],
                       color_img_list: List[str], cam_calib: o3d.camera.PinholeCameraIntrinsic, method: str,
                       visualise: bool = False) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """

        :param pose_mat:
        :param rot_mat:
        :param color_img_list:
        :param cam_calib:
        :param method:
        :param visualise:
        :return:
        """
        # Extract keypoints
        if method == 'local_tracking':
            _observations, _mappoints = self._track_keypoints(img_list=color_img_list)
        elif method == 'global_matching':
            _keypoints, _kp_starts = self._extract_keypoints(img_list=color_img_list)
            _observations, _mappoints = self._match_keypoints(keypoints=_keypoints, kp_starts=_kp_starts)
        else:
            raise NotImplementedError('Method not implemented. Please choose either \'local_tracking\' or '
                                      '\'global_matching\'')

        # Triangulate mappoints
        _mappoints = self._triangulate_mappoints(observations=_observations, cam_calib=cam_calib, mappoints=_mappoints,
                                          pose_mat=pose_mat, rot_mat=rot_mat)

        self.mappoints = _mappoints
        self.observations = _observations

        return self.observations, self.mappoints


class BundleAdjustment(object):
    def __init__(self):
        self.pose_mat = np.empty(shape=(0, 3))
        self.rot_mat = np.empty(shape=(0, 3))

    @staticmethod
    def _jacobian_sparsity(observations):
        """
        Calculate which elements of the Jacobian matrix are non-zero

        :param n_cameras:
        :param n_points:
        :param camera_indices:
        :param point_indices:
        :return:
        """

        n_cameras = np.max(observations[:, 1], axis=0) + 1
        n_points = np.max(observations[:, 0], axis=0) + 1
        n_keypoints = observations.shape[0]

        m = int(n_keypoints * 2)
        n = int(n_points*3 + n_cameras*6)
        sparsity_matrix = lil_matrix((m, n), dtype=int)

        for i, keypoint in enumerate(observations):
            kp_row_id = i*2
            point_col_id = int(keypoint[0]*3)
            camera_col_id = int(n_points*3 + keypoint[1]*6)

            # Set mappoint involved in the observation
            sparsity_matrix[kp_row_id: kp_row_id+2, point_col_id: point_col_id+3] = 1
            # Set the camera involved in the observation
            sparsity_matrix[kp_row_id: kp_row_id+2, camera_col_id: camera_col_id+6] = 1

        return sparsity_matrix

    @staticmethod
    def _project_points(x_params, observations, cam_calib):
        """This function calculates the residual projection error values for each mappoint-image observation pair

        x_params is an (n,) shaped input array containing the parameters to be optimised, which are
          - 3D positions of the mappoints
          - 6DOF positions of the cameras

        n_cameras and n_points parameters are used to parse x_params, while obsrv contains the measured observations
         of the mappoints that are used to calculate the residual values.
        """

        # For each camera, project the observed mappoints and calculate the residual values.
        # OpenCV handles the necessary rotation for the projection

        # Splot the free param array to points and cameras and reshape them to their proper shapes
        n_points = int(np.max(observations[:, 0], axis=0)) + 1
        n_cameras = int(np.max(observations[:, 1], axis=0)) + 1

        points, cameras = x_params[:n_points*3], x_params[n_points*3:]

        mappoints = points.reshape(-1, 3)
        cameras = cameras.reshape(-1, 6)

        projected_points = np.empty(shape=(0, 2))
        measured_points = np.empty(shape=(0, 2))

        for image_id in range(n_cameras):  # Iterate over the camera ids
            observing_keypoints = observations[observations[:, 1] == image_id, :]
            observed_mappoints = observing_keypoints[:, 0].astype(int)

            new_points, jacobian = cv2.projectPoints(objectPoints=mappoints[observed_mappoints].transpose(),
                                                     rvec=cameras[image_id, 3:],
                                                     tvec=cameras[image_id, :3],
                                                     cameraMatrix=cam_calib.intrinsic_matrix,
                                                     # No distortion parameters available so assumed undistorted
                                                     distCoeffs=np.zeros(5))

            projected_points = np.vstack([projected_points, new_points.squeeze()])
            measured_points = np.vstack([measured_points, observing_keypoints[:, 2:]])

        # Return a flattened array of difference between projection coordinates
        ret = (projected_points - measured_points).ravel()
        return ret

    def refine_poses(self, pose_mat: npt.NDArray[np.float64], rot_mat: npt.NDArray[np.float64],
                     cam_calib: o3d.camera.PinholeCameraIntrinsic, mappoints: npt.NDArray[np.float64],
                     observations: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Use bundle adjustment to refine the initial camera pose estimates. Bunde adjustment requires initial estimates of
         the independent variables, which are the
           - 6DOF Camera position estimates and calibration parameters
           - 3D positions of the mappoints

         - Observations of the mappoints as keypoints in images

        To obtain an initial estimate of the mappoints, keypoints are detected in the images and matched by descriptor
        values. The matches are then merged to create a list of points observed by a list of keypoints each. For each
        observation the mappoint position is calculated using the depth information and is averaged (excluding 0.0 values
        meaning depth information is not present).

        In the bundle adjustment stage, mappoints are projected onto the images that observed that given mappoint.
        A residual value is calculated for each observation by subtracting the projection coordinates from
        the measured (keypoint) coordinates. The residual values are stored in a sparse matrix, for which the Jacobians
        (first order partial derivatives) are calculated. We use Scipy's least squares optimisation function to optimize
        the inputs, the camera parameters and the mappoint locations.

        :param rot_mat:
        :param pose_mat:
        :param color_img_list:
        :param cam_calib:
        :return:
        """
        # Calculate jacobian sparsity structure to be able to fit the Jacobian matrix in memory
        sp_mtx = self._jacobian_sparsity(observations=observations)

        # Perform LSE optimisation
        x_params = np.hstack([mappoints.ravel(), np.hstack([pose_mat, rot_mat]).ravel()])
        res = least_squares(fun=self._project_points, x0=x_params, jac_sparsity=sp_mtx, loss='linear', verbose=2,
                            x_scale='jac', ftol=1e-4, method='trf', max_nfev=100, jac='2-point',
                            args=(observations, cam_calib))

        # Parse and return the results
        new_camera_poses = res.x[mappoints.shape[0] * 3:].reshape(-1, 6)
        _pose_mat, _rot_mat = new_camera_poses[:, :3], new_camera_poses[:, 3:]

        self.pose_mat = _pose_mat
        self.rot_mat = _rot_mat

        return self.pose_mat, self.rot_mat


def get_better_depthmaps(imgpairs, baseline):
    """According to https://www.mouser.com/pdfdocs/Intel_D400_Series_Datasheet.pdf the Intel D435 has
    50mm stereo baseline"""

    for left_path, right_path in imgpairs:
        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        depth_map = cv2.imread('/home/attila/Datasets/Remy/depth-00050.png', cv2.IMREAD_UNCHANGED)
        depth_map[depth_map > 1e+3] = 0

        c = cv2.vconcat([left, right])

        #cv2.imshow('disparity', c)
        #cv2.imshow('disparity', right)
        #cv2.waitKey(100000)

        fx = 1.93  # lense focal length
        baseline = 50  # distance in mm between the two cameras
        disparities = 128  # num of disparities to consider
        block = 31  # block size to match
        units = 0.001  # depth units

        stereo = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)
        disparity = stereo.compute(left, right)
        depth = np.zeros(shape=left.shape).astype(float)
        depth[disparity > 0] = (fx * baseline) / (units * disparity[disparity > 0])
        depth[depth > 1e+3] = 0


        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # Remove horizontal space between axes
        #fig.subplots_adjust(vspace=0)

        # Plot each graph, and manually set the y tick values
        axs[0].imshow(depth_map)
        axs[1].imshow(depth)
        plt.show()


        #disparity = ((disparity + np.min(disparity)) / np.max(disparity) * 255).astype(np.uint8)
        #cv2.imshow('disparity', disparity)
        #cv2.waitKey(1000)


def main():
    # Data dir with '/' at the end
    data_path = '/home/attila/Datasets/Remy'

    # Change dataset limits or stride if needed
    start = 0
    end = 195
    step = 1

    # Parse scanner log and load position and rotation data
    pose_data = np.genfromtxt(f'{data_path}/scanner.log', delimiter=' ', usecols=range(2, 8))
    rot_mat = pose_data[start:end:step, 3:]
    pose_mat = pose_data[start:end:step, 0:3]

    # Create image path lists
    color_img_list = [f'{data_path}/rgb-{str(i).zfill(5)}.png' for i in range(start, end, step)]
    depth_img_list = [f'{data_path}/depth-{str(i).zfill(5)}.png' for i in range(start, end, step)]
    ir1_img_list = [f'{data_path}/ir1-{str(i).zfill(5)}.png' for i in range(start, end, step)]
    ir2_img_list = [f'{data_path}/ir2-{str(i).zfill(5)}.png' for i in range(start, end, step)]

    # Calibration data
    cam_calib = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                  fx=613.688, fy=614.261,
                                                  cx=323.035, cy=242.229)

    # Perform an initial reconstruction with the provided data and inspect results
    #reconstruct(pose_mat, rot_mat, color_img_list, depth_img_list, cam_calib=cam_calib, visualise=True)

    # Process stereo images, create better depth maps and inspect results
    #depth_img_list = get_better_depthmaps(zip(ir1_img_list, ir2_img_list), baseline=0.05)
    #reconstruct(pose_mat, rot_mat, color_img_list, depth_img_list, visualise=True)

    # Refine camera poses with ORB features and bundle adjustment
    tracker = KeypointTracker()
    observations, mappoints = tracker.compute_points(pose_mat=pose_mat, rot_mat=rot_mat, color_img_list=color_img_list,
                                                     cam_calib=cam_calib, method='global_matching', visualise=False)

    ba = BundleAdjustment()
    new_pose_mat, new_rot_mat = ba.refine_poses(pose_mat=pose_mat, rot_mat=rot_mat, mappoints=mappoints,
                                                observations=observations, cam_calib=cam_calib)

    print(np.linalg.norm(pose_mat - new_pose_mat, axis=1))
    print(np.linalg.norm(rot_mat - new_rot_mat, axis=1))

    reconstruct(new_pose_mat, new_rot_mat, color_img_list, depth_img_list, cam_calib, visualise=True)

    # Perform color ICP on the final pointcloud to improve the model
    #color_icp()


if __name__ == '__main__':
    main()
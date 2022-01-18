import numpy as np
import cv2
import open3d as o3d

from copy import copy
from time import sleep

from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import matplotlib.pyplot as plt
from tqdm import tqdm


class PointCloudAligner(object):
    def __init__(self):
        self.cam_calib = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                           fx=613.688, fy=614.261,
                                                           cx=323.035, cy=242.229)

        self.vis = None
        self.pcd = None

    def reproject(self, image, depth_map, translation=None, rotation=None, transformation=None):
        _image = o3d.geometry.Image(image)
        _depth_map = o3d.geometry.Image(depth_map)
        depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(image),
                                                                         depth=o3d.geometry.Image(depth_map),
                                                                         depth_scale=1000, depth_trunc=0.6,
                                                                         convert_rgb_to_intensity=True)

        if transformation is not None:
            t_mtx = transformation
        else:
            t_mtx = np.identity(4)
            if rotation is not None:
                t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation)
            if translation is not None:
                t_mtx[:3, 3] = translation

        # Need to track the pcd in the same pointer otherwise the visualisation update doesn't work...
        _pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=depth_image,
                                                              intrinsic=self.cam_calib)

        _pcd.transform(t_mtx)

        try:
            self.pcd.points = _pcd.points
            self.pcd.colors = _pcd.colors
        except AttributeError:
            self.pcd = _pcd

        return _pcd

    def visualise(self, pcd):
        if not self.vis:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            self.ctr = self.vis.get_view_control()
        else:
            self.vis.add_geometry(pcd)
            #self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()


def align_scanner(pose_mat, rot_mat, color_img_list, depth_img_list):
    """Reproject the depth maps to 3D and align the fragments
    using the camera poses/orientations from the scanner log"""

    length = len(color_img_list)

    vis = PointCloudAligner()

    for i, (color_path, depth_path) in enumerate(zip(color_img_list, depth_img_list)):
        color = cv2.imread(color_img_list[i], cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_img_list[i], cv2.IMREAD_UNCHANGED)

        pcd = vis.reproject(image=color, depth_map=depth, rotation=rot_mat[i, :], translation=pose_mat[i, :])
        vis.visualise(pcd)

    vis.vis.run()


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


def refine_camera_poses(rot_mat, pose_mat, color_img_list, cam_calib):
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

    def jacobian_sparsity(observations):
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

        for row in observations:
            print(row)

        for i, keypoint in enumerate(observations):
            kp_row_id = i*2
            point_col_id = int(keypoint[0]*3)
            camera_col_id = int(n_points*3 + keypoint[1]*6)

            # Set mappoint involved in the observation
            sparsity_matrix[kp_row_id: kp_row_id+2, point_col_id: point_col_id+3] = 1
            # Set the camera involved in the observation
            sparsity_matrix[kp_row_id: kp_row_id+2, camera_col_id: camera_col_id+6] = 1

        print(sparsity_matrix[:, :3])

        return sparsity_matrix

    def project_points(x_params, observations, cam_calib):
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
                                                     distCoeffs=np.array([]))

            projected_points = np.vstack([projected_points, new_points.squeeze()])
            measured_points = np.vstack([measured_points, observing_keypoints[:, 2:]])

        # Return a flattened array of difference between projection coordinates
        ret = (projected_points - measured_points).ravel()
        return ret

    """
    
    Calculate Observations
    
    """
    # Generate key points and descriptors with ORB for all images and store in a list of dictionaries
    length = len(color_img_list)
    orb = cv2.ORB_create()
    features = [{'key_points': [],
                 'descriptors': np.empty((0, 32))} for i in range(length)]  # Initialise a list of dictionaries
    # im_no, x, y, des
    keypoints = np.empty(shape=(0, 35))
    kp_start = [0]  # Store the first keypoint index for the corresponding image no.

    for i in tqdm(range(0, length), desc='Keypoint extraction'):
        color = cv2.imread(color_img_list[i], cv2.IMREAD_COLOR)
        kp = orb.detect(color, None)
        kp, des = orb.compute(color, kp)

        features[i]['key_points'] = list(kp)

        keypoints = np.vstack([keypoints, np.hstack([i * np.ones(shape=(len(kp), 1)),
                                                     np.array([p.pt for p in kp]),
                                                     des])])
        kp_start.append(kp_start[-1] + len(kp))

    # Match keypoints exhaustively with a brute force matcher - we optimise for reconstruction accuracy now, not runtime
    # Only the top 10 matches are stored
    mappoints = np.empty(shape=(0, 3))  # Store the initial point position estimates
    # Store the observed mappoint id and observing image no with keypoint x, y coordinates
    observations = np.empty(shape=(0, 4))
    for i in tqdm(range(0, length-1), desc='Keypoint matching'):
        for j in range(i+1, length):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(queryDescriptors=keypoints[kp_start[i]: kp_start[i+1], 3:].astype(np.uint8),
                               trainDescriptors=keypoints[kp_start[j]: kp_start[j+1], 3:].astype(np.uint8))
            matches = sorted(matches, key=lambda x: x.distance)

            for match in matches[:30]:
                # Add an empty mappoint
                mappoints = np.vstack([mappoints, np.zeros(shape=(1, 3))])
                # Add two observations
                observations = np.vstack([observations,
                                          np.hstack([mappoints.shape[0]-1,
                                                     np.array([i]),
                                                     keypoints[kp_start[i]+match.queryIdx][1:3]]),
                                          np.hstack([mappoints.shape[0]-1,
                                                     np.array([j]),
                                                     keypoints[kp_start[j] + match.trainIdx][1:3]])])

            '''
            # Show matches
            img1 = cv2.imread(color_img_list[i], cv2.IMREAD_COLOR)
            img2 = cv2.imread(color_img_list[j], cv2.IMREAD_COLOR)
            img3 = cv2.drawMatches(img1, features[i]['key_points'], img2, features[j]['key_points'],
                                   matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', img3), cv2.waitKey(1)
            '''
    observations = np.array(sorted(observations, key=lambda x: (x[0])))  # Key: mappoint id

    for keypoint_pair in observations.reshape(-1, 8):
        #print(keypoint_pair)
        t_mtx = np.zeros(shape=(3, 4))
        t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_mat[int(keypoint_pair[1])])
        t_mtx[:3, 3] = pose_mat[int(keypoint_pair[1])]
        proj_1 = cam_calib.intrinsic_matrix @ t_mtx
        points_1 = keypoint_pair[2:4].transpose()

        t_mtx = np.zeros(shape=(3, 4))
        t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_mat[int(keypoint_pair[5])])
        t_mtx[:3, 3] = pose_mat[int(keypoint_pair[5])]
        proj_2 = cam_calib.intrinsic_matrix @ t_mtx
        points_2 = keypoint_pair[6:8].transpose()

        res = cv2.triangulatePoints(projMatr1=proj_1, projMatr2=proj_2, projPoints1=points_1, projPoints2=points_2)
        mappoints[int(keypoint_pair[0]), :] = cv2.convertPointsFromHomogeneous(res.transpose()).flatten()

    '''
    Merge mappoints - Keypoints have been discovere and triangulated pair-wise.
    # Merge observations - Some keypoints may have been registered to multiple mappoints
    observations = np.array(sorted(observations, key=lambda x: (x[1], x[2], x[3])))  # Keys: image ID, x, y
    # Return unique values -
    unique, unique_idx = np.unique(observations[:, 1:], axis=0, return_index=True)
    '''

    observations = np.array(sorted(observations, key=lambda x: (x[1], x[2], x[3])))  # Keys: image ID, x, y
    unique, unique_idx = np.unique(observations[:, 1:], axis=0, return_index=True)  # Unique keypoints regardless mappoint id
    final_mappoint_id_lookup = np.zeros(mappoints.shape[0])
    for u, unique_id in enumerate(unique_idx[:-1]):
        i = 0
        # For each non-unique element mappoint id, change all occurrences of that mappoint id to the
        # unique element mappoint id.
        while unique_id+i < observations.shape[0] and unique_id+i < unique_idx[u+1]:
            current_row = observations[unique_id+i]
            # Mark into which mappoint is the current mappoint merged
            final_mappoint_id_lookup[int(current_row[0])] = observations[unique_id, 0]
            # Set the value of the mappoint id of all observations that share the same value with the current row
            # to the mappoint id of the last unique row, which is the mappoint it is being merged into
            observations[observations[:, 0] == current_row[0], 0] = observations[unique_id, 0]
            i += 1

    # Average mappoint coordinates over the observed keypoints
    unique_mappoint_ids = np.unique(observations[:, 0], return_index=False)
    final_mappoints = np.empty(shape=(0, 3))
    for id in unique_mappoint_ids:  # Iterate through the merged mappoint ids
        coords = mappoints[final_mappoint_id_lookup == id]  # Pull all mappoint coords with the merged id
        # Add the final mappoint with the mean coordinates
        final_mappoints = np.vstack([final_mappoints, np.mean(coords, axis=0)])

    mappoints = final_mappoints

    # Replace the old mappoint ids in the observations with the new ones
    # There's probably a more numpy way of doing this than a for loop that I can't think of right now
    for new_id, old_id in enumerate(unique_mappoint_ids):
        observations[observations[:, 0] == old_id, 0] = new_id

    # Discard duplicate observations of the same merged mappoint in images
    unique, unique_ids = np.unique(observations[:, :2], return_index=True, axis=0)
    observations = observations[unique_ids]
    observations = np.array(sorted(observations, key=lambda x: (x[1], x[2], x[3])))

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mappoints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd, reset_bounding_box=True)
    vis.run()
    '''

    """
    Calculate Jaccard sparsity structure
    """

    sparsity_mtx = jacobian_sparsity(observations)

    """
    Perform optimisation
    """

    # Optimize the
    x_params = np.hstack([mappoints.ravel(), np.hstack([pose_mat[:length], rot_mat[:length]]).ravel()])
    res = least_squares(fun=project_points, x0=x_params, jac_sparsity=sparsity_mtx, loss='cauchy', verbose=2, x_scale='jac', ftol=1e-4,
                        method='trf', args=(observations, cam_calib))

    new_camera_poses = res.x[mappoints.shape[0] * 3:].reshape(-1, 6)
    pose_mat, rot_mat = new_camera_poses[:, :3], new_camera_poses[:, 3:]

    return pose_mat, rot_mat


def main():

    start = 0
    end = 50

    pose_data = np.genfromtxt('/home/attila/Datasets/Remy/scanner.log', delimiter=' ', usecols=range(2, 8))
    rot_mat = pose_data[start:end, 3:]
    pose_mat = pose_data[start:end, 0:3]

    color_img_list = [f'/home/attila/Datasets/Remy/rgb-{str(i).zfill(5)}.png' for i in range(start, end)]
    depth_img_list = [f'/home/attila/Datasets/Remy/depth-{str(i).zfill(5)}.png' for i in range(start, end)]

    cam_calib = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                  fx=613.688, fy=614.261,
                                                  cx=323.035, cy=242.229)

    align_scanner(pose_mat, rot_mat, color_img_list, depth_img_list)
    #color_icp()
    pose_mat, rot_mat = refine_camera_poses(rot_mat=rot_mat, pose_mat=pose_mat,
                                            color_img_list=color_img_list, cam_calib=cam_calib)

    align_scanner(pose_mat, rot_mat, color_img_list, depth_img_list)


if __name__ == '__main__':
    main()
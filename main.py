import numpy as np
import cv2
import open3d as o3d


class PointCloudAligner(object):
    def __init__(self):
        self.cam_calib = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                           fx=613.688, fy=614.261,
                                                           cx=323.035, cy=242.229)

        self.vis = None
        self.pcd = None

    def reproject(self, image, depth_map, translation, rotation):
        _image = o3d.geometry.Image(image)
        _depth_map = o3d.geometry.Image(depth_map)
        depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(image),
                                                                         depth=o3d.geometry.Image(depth_map),
                                                                         depth_scale=1000, depth_trunc=0.6,
                                                                         convert_rgb_to_intensity=True)

        t_mtx = np.identity(4)
        t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation)
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
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()


def align_scanner():
    """Reproject the depth maps to 3D and align the fragments
    using the camera poses/orientations from the scanner log"""
    pose_data = np.genfromtxt('/home/attila/Datasets/Remy/scanner.log', delimiter=' ', usecols=range(2, 8))
    rot_mat = pose_data[:, 3:]
    pose_mat = pose_data[:, 0:3]

    vis = PointCloudAligner()

    for i in range(0, 195):
        color = cv2.imread(f'/home/attila/Datasets/Remy/rgb-{str(i).zfill(5)}.png', cv2.IMREAD_COLOR)
        depth = cv2.imread(f'/home/attila/Datasets/Remy/depth-{str(i).zfill(5)}.png', cv2.IMREAD_UNCHANGED)

        pcd = vis.reproject(image=color, depth_map=depth, rotation=rot_mat[i, :], translation=pose_mat[i, :])
        vis.visualise(pcd)

    vis.vis.run()


def tsdf_integration():
    """Open3d supports simple integration with Truncated Signed Distance Field depth map fusion,
    so trying it was logical. Unfortunately it doesn't perform very well,
    due to depth map noise around the base of the cup."""

    pose_data = np.genfromtxt('/home/attila/Datasets/Remy/scanner.log', delimiter=' ', usecols=range(2, 8))
    rot_mat = pose_data[:, 3:]
    pose_mat = pose_data[:, 0:3]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=1.0 / 512.0,
        sdf_trunc=0.4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    cam_calib = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                  fx=613.688, fy=614.261,
                                                  cx=323.035, cy=242.229)

    for i in range(30):
        t_mtx = np.identity(4)
        t_mtx[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_mat[i, :])
        t_mtx[:3, 3] = pose_mat[i, :]

        color = cv2.imread(f'/home/attila/Datasets/Remy/rgb-{str(i).zfill(5)}.png', cv2.IMREAD_COLOR)
        depth = cv2.imread(f'/home/attila/Datasets/Remy/depth-{str(i).zfill(5)}.png', cv2.IMREAD_UNCHANGED)
        image = o3d.geometry.Image(color)
        depth_map = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            image, depth_map, depth_trunc=0.6, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            np.linalg.inv(t_mtx))

    pcd = volume.extract_point_cloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd, reset_bounding_box=True)

    vis.run()


def main():
    #align_scanner()
    tsdf_integration()


if __name__ == '__main__':
    main()
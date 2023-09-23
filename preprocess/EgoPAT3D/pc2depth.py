"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pcd_file = '../../data/sequences/bathroomCabinet/bathroomCabinet_2/pointcloud/300.ply'

    pcd = o3d.io.read_point_cloud(pcd_file)
    pointxyz = np.asarray(pcd.points)
    pointcolor = np.asarray(pcd.colors)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # camera intrinsics and exterinsics
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.extrinsic = np.array([[1,0,0,0], 
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])
    cam_param.intrinsic.set_intrinsics(width=3840, height=2160, fx=1.80820276e+03, fy=1.80794556e+03, cx=1.94228662e+03, cy=1.12382178e+03)

    # get depth
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    viewer.run()
    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(cam_param)
    depth = viewer.capture_depth_float_buffer()

    plt.imshow(np.asarray(depth))
    plt.imsave("testing_depth.png", np.asarray(depth), dpi = 1)

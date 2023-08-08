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

import os
import sys
PYK4A_ROOT = os.path.join(os.path.dirname(__file__), '../../third_party/pyk4a')
sys.path.append(os.path.join(PYK4A_ROOT, 'example'))
from helpers import convert_to_bgra_if_required, colorize

import pyk4a
from pyk4a import PyK4APlayback
import cv2
import numpy as np


def read_by_whileloop(mkvraw_file, rgb_path, depth_path):
    playback = PyK4APlayback(mkvraw_file)
    playback.open()
    num=1
    depth_range = [0, 1000]
    while True:
        capture = playback.get_next_capture()
        if capture.color is not None:
            rgb = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
            cv2.imwrite(os.path.join(rgb_path, '{}.jpg'.format(num)), rgb)
            if capture.depth is not None:
              depth = pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe)
              if depth is not None:
                if np.max(depth) > depth_range[1]:
                    depth_range[1] = np.max(depth)
                depth = colorize(depth, tuple(depth_range))
                cv2.imwrite(os.path.join(depth_path, '{}.png'.format(num)), depth)
                print('{}'.format(num))
            num += 1
            if num >= 1000:
                break
    cv2.destroyAllWindows()
    playback.close()


def get_specified_frame(playback, pointer, i):
    rgb, depth = None, None
    while True:
        capture = playback.get_next_capture()
        if (pointer == i) and (capture.color is not None) and (capture.depth is not None):
            rgb = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
            depth = pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe)
            break
        if pointer < i:
            pointer += 1
        else:
            break
    return rgb, depth, pointer


def read_by_frameid(mkvraw_file, rgb_path, depth_path):
    playback = PyK4APlayback(mkvraw_file)
    playback.open()
    # fps_dict = {'FPS.FPS_5': 5, 'FPS.FPS_15': 15, 'FPS.FPS_30': 30}
    # fps = fps_dict[str(playback.configuration['camera_fps'])]
    # offset = playback.configuration["start_timestamp_offset_usec"]
    pointer = 0
    depth_range = [0, 1000]
    for i in range(20, 1001, 10):
        print('{}'.format(i))
        rgb, depth, pointer = get_specified_frame(playback, pointer, i)
        if (rgb is not None) and (depth is not None):
            cv2.imwrite(os.path.join(rgb_path, '{}.jpg'.format(i)), rgb)
            depth = colorize(depth, tuple(depth_range))
            cv2.imwrite(os.path.join(depth_path, '{}.png'.format(i)), depth)
    
        # pos = int(i / fps * 1000000)
        # playback.seek(pos, origin=SeekOrigin.BEGIN)
        # capture = playback.get_next_capture()
        # rgb = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
        # depth = pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe)



def main():
    scene_id = '1'
    scene_name = 'bathroomCabinet'
    record_id = '2'
    record_name = '{}_{}'.format(scene_name, record_id)

    root_path = os.path.join(os.path.dirname(__file__), '../../data')
    record_path = os.path.join(root_path, 'EgoPAT3D-complete', scene_id, record_name)  # './1/bathroomCabinet_2'

    mkvraw_file = os.path.join(record_path, '{}.mkv'.format(record_name))
    # mkvraw_file = '../{}.mkv'.format(record_name)

    result_path = '../{}'.format(record_name)
    rgb_path = os.path.join(result_path, 'color')
    depth_path = os.path.join(result_path, 'depth')
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    read_by_whileloop(mkvraw_file, rgb_path, depth_path)

    # read_by_frameid(mkvraw_file, rgb_path, depth_path)



if __name__ == '__main__':

    main()

    
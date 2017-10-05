from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

from pixutils import *


# from .BridgeIt import *

def video_meta_data(cam, source='stream'):
    if source == 'stream':
        # f_width, f_height, fps = [cam.get(i) for i in [3, 4, 5]]
        f_width, f_height,fps = 1200, 800,  30
        return int(f_width), int(f_height), int(fps), np.inf
    else:
        f_width, f_height, fps, total_frames = [cam.get(i) for i in [3, 4, 5, 7]]
        return int(f_width), int(f_height), int(fps), int(total_frames)


def get_cam_matrix((cam_w,cam_h)):
    c_x, c_y = cam_h / 2, cam_w / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
    return camera_matrix


def get_lens_distortion():
    # Assuming no lens distortion
    return np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
import argparse
import cv2
import numpy as np
import os
import time

from renderer import ObjectRenderer
from utils import make_transform


def parse_args():
    parser = argparse.ArgumentParser(prog='3d model viewer')
    parser.add_argument(
        '--model', type=str,
        default=os.path.join('models', 'gear.obj'),
        help='Camera FOV (Y-axis) in degrees',
    )
    parser.add_argument(
        '--camera_y_fov', type=float, default=40.,
        help='Camera FOV (Y-axis) in degrees',
    )
    parser.add_argument(
        '--camera_x_res', type=int, default=1920,
        help='Camera X resolution',
    )
    parser.add_argument(
        '--camera_y_res', type=int, default=1080,
        help='Camera Y resolution',
    )
    parser.add_argument(
        '--rotation_speed_x', type=float, default=0.,
        help='Rotation speed around X-axis (deg/s)'
    )
    parser.add_argument(
        '--rotation_speed_y', type=float, default=0.,
        help='Rotation speed around Y-axis (deg/s)'
    )
    parser.add_argument(
        '--rotation_speed_z', type=float, default=0.,
        help='Rotation speed around Y-axis (deg/s)'
    )
    parser.add_argument(
        '--distance', type=float, default=100,
        help='Distance from the camera (mm)',
    )
    return parser.parse_args()


def show(args):
    renderer = ObjectRenderer(
        args.model,
        camera_y_fov=np.deg2rad(args.camera_y_fov),
        render_x_res=args.camera_x_res,
        render_y_res=args.camera_y_res,
        light_rgb=(0.2, 0.2, 1.),
        light_intensity=500.,
        transform=make_transform((90, 0, 0), (0, 0, 0)),
    )

    fps = 2.
    diff_scale = 1. / fps
    interval_ms = int(1. / fps)

    distance = args.distance
    x_angle = 0.
    y_angle = 0.
    z_angle = 0.


    while True:
        start_tstamp = time.time()
        x_angle += diff_scale * args.rotation_speed_x
        y_angle += diff_scale * args.rotation_speed_y
        z_angle += diff_scale * args.rotation_speed_z

        transform = make_transform((x_angle, y_angle, z_angle), (0, 0, distance))
        image, _ = renderer.render(transform)
        cv2.imshow('Model', image)

        to_wait = start_tstamp + 1. / fps - time.time()
        to_wait_ms = max(1, int(to_wait * 1000))
        if cv2.waitKey(to_wait_ms) & 0xFF == ord('c'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    show(parse_args())

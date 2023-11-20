import argparse
import cv2
import math
import numpy as np
import os
import yaml

from renderer import ObjectRenderer
from utils import profile, fuse_images


def parse_args():
    parser = argparse.ArgumentParser(prog='AR demo')
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'run.yaml'),
    )
    return parser.parse_args()


def run(args):
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)

    cap = cv2.VideoCapture(config['source'])
    _, img = cap.read()

    pattern_size = (config['pattern']['x_size'], config['pattern']['y_size'])
    cell_size_mm = config['pattern']['cell_size_mm']

    intrinsics = np.load(config['intrinsics']['path'])
    camera_matrix, dist_coeff = intrinsics['cameraMatrix'], intrinsics['distCoeffs']

    model_path = config['model']['path']
    initial_transform = np.asarray(config['model']['transform'])

    y_res, x_res, _ = img.shape
    fov_y = 2 * math.atan((y_res / (2 * camera_matrix[1, 1])))

    renderer = ObjectRenderer(
        path=model_path,
        camera_y_fov=fov_y,
        render_x_res=x_res,
        render_y_res=y_res,
        light_rgb=(0.1, 1.0, 0.1),
        light_intensity=50.,
        transform=initial_transform,
    )

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp[:, 0] -= 0.5 * (pattern_size[0] - 1)
    objp[:, 1] -= 0.5 * (pattern_size[1] - 1)
    objp[:, :2] *= cell_size_mm

    writer = None
    if config['output']['enable']:
        writer = cv2.VideoWriter(
            config['output']['path'],
            cv2.VideoWriter_fourcc(*'mp4v'),
            config['output']['fps'],
            (x_res, y_res)
        )

    while True:
        print('-' * 30)
        with profile('Main loop'):
            with profile('Image capturing'):
                ok, img = cap.read()
                if not ok:
                    break
                vis = img.copy()
                transform = None

            with profile('Detecting marker'):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ok, corners = cv2.findChessboardCorners(
                    gray, pattern_size, cv2.CALIB_CB_FAST_CHECK
                )
                if ok:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_fine = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )

            if ok:
                with profile('Pose estimation'):
                    ok, rvecs, tvecs = cv2.solvePnP(objp, corners_fine, camera_matrix, dist_coeff)
                    if ok:
                        rvecs[1][0] *= -1
                        rvecs[2][0] *= -1
                        rotation_matrix, _ = cv2.Rodrigues(rvecs)
                        transform = np.eye(4)
                        transform[:3, :3] = rotation_matrix
                        transform[2, -1] = -tvecs[2]
                        transform[1, -1] = -tvecs[1]
                        transform[0, -1] = tvecs[0]

            if transform is not None:
                with profile('Rendering & fusing'):
                    render, mask = renderer.render(transform)
                    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                    vis = fuse_images(src=vis, subst_img=render, subst_mask=mask)

            with profile('Visualization'):
                cv2.imshow('Stream', vis)

            if writer is not None:
                with profile('Recording'):
                    writer.write(vis)

            if cv2.waitKey(10) & 0xFF==ord('c'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    run(parse_args())

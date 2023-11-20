import argparse
import cv2
import numpy as np
import os
import time

from utils import profile


def parse_args():
    parser = argparse.ArgumentParser(prog='Camera calibrator')
    parser.add_argument('--source', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--interval_sec', type=float, default=1.)
    parser.add_argument('--target_frames', type=int, default=100)
    parser.add_argument('--patterm_xzs', type=int, default=6)
    parser.add_argument('--patterm_yzs', type=int, default=9)
    parser.add_argument('--output', type=str, default=os.path.join('config', 'intrinsics.npz'))
    return parser.parse_args()


def detect(args):
    cap = cv2.VideoCapture(args.source)


    objp = np.zeros((args.patterm_xzs * args.patterm_yzs, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.patterm_xzs, 0:args.patterm_yzs].T.reshape(-1,2)
    objpoints, imgpoints = [], []

    frames_cntr, warmup_cntr = 0, 0
    last_detection_tstamp = time.time()
    while frames_cntr < args.target_frames:
        ok, img = cap.read()
        vis = img.copy()

        warmup_cntr += 1
        if warmup_cntr >= args.warmup and time.time() > last_detection_tstamp + args.interval_sec:
            with profile('Processing'):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ok, corners = cv2.findChessboardCorners(
                    img,
                    (args.patterm_xzs, args.patterm_yzs),
                    cv2.CALIB_CB_FAST_CHECK
                )
                if ok:
                    objpoints.append(objp)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_fine = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners_fine)
                    last_detection_tstamp = time.time()
                    frames_cntr += 1
                    print(f'Captured calibration frames: {frames_cntr} / {args.target_frames}')

                    cv2.drawChessboardCorners(vis, (args.patterm_xzs, args.patterm_yzs), corners_fine, ok)

        cv2.imshow('Stream', vis)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

    ok, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    np.savez(args.output, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
    print(f'Intrinsics have been saved @ {args.output}')

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    detect(parse_args())

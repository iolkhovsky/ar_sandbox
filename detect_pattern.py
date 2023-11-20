import argparse
import cv2
import numpy as np

from utils import profile


def parse_args():
    parser = argparse.ArgumentParser(prog='Marker detector')
    parser.add_argument('--source', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--patterm_xzs', type=int, default=6)
    parser.add_argument('--patterm_yzs', type=int, default=9)
    return parser.parse_args()


def detect(args):
    cap = cv2.VideoCapture(args.source)

    frames_cntr = 0
    while True:
        ok, img = cap.read()
        vis = img.copy()
        frames_cntr += 1
        if frames_cntr >= args.warmup:
            with profile('Processing'):
                objp = np.zeros((args.patterm_xzs * args.patterm_yzs, 3), np.float32)
                objp[:, :2] = np.mgrid[0:args.patterm_xzs, 0:args.patterm_yzs].T.reshape(-1,2)
                objpoints, imgpoints = [], []

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
                    cv2.drawChessboardCorners(vis, (args.patterm_xzs, args.patterm_yzs), corners_fine, ok)

        cv2.imshow('Stream', vis)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    detect(parse_args())

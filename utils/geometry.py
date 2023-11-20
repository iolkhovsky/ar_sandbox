import numpy as np
from scipy.spatial.transform import Rotation as R


def make_transform(rotation, translation, scale=1., degrees=True):
    rot = R.from_euler(seq='XYZ', angles=rotation, degrees=degrees).as_matrix()
    trans = np.asarray(translation).T   
    transform = np.eye(4, 4)
    transform[:3, :3] = rot.copy()
    transform[:3, -1] = trans
    return transform

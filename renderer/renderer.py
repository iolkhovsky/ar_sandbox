import numpy as np
import pyrender
from pyrender.constants import RenderFlags
import numpy as np
import trimesh
from typing import Union, List, Tuple

from utils import profile


class ObjectRenderer:
    def __init__(
        self, path: str,
        camera_y_fov: float,
        render_x_res: int,
        render_y_res: int,
        light_rgb: Union[List[float], Tuple[float]] = (1., 1., 1.),
        light_intensity: float = 1.,
        transform: np.ndarray = np.eye(4),
    ):
        self._trimesh = trimesh.load(path)
        if transform is not None:
            self._trimesh.apply_transform(transform)

        self._camera = pyrender.PerspectiveCamera(
            yfov=camera_y_fov,
            aspectRatio=render_x_res / render_y_res
        )
        self._light = pyrender.DirectionalLight(
            color=np.asarray(light_rgb),
            intensity=light_intensity,
        )
        self._renderer = pyrender.OffscreenRenderer(render_x_res, render_y_res)

    @profile('Renderer')
    def render(self, transform: np.ndarray = np.eye(4)):
        trimesh = self._trimesh.copy()
        trimesh.apply_transform(transform)

        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(trimesh)
        scene.add(mesh)
    
        scene.add(self._camera, pose=np.eye(4))
        scene.add(self._light, pose=np.eye(4))

        flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
        image, depth = self._renderer.render(scene, flags)
        mask = depth > 0
        return image, mask
